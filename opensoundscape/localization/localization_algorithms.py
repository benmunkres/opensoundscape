"""Tools for localizing audio events from synchronized recording arrays"""

import warnings
import numpy as np
from scipy.optimize import least_squares
import itertools
import functools
from scipy.signal import hilbert
import librosa

# define defaults for physical constants
SPEED_OF_SOUND = 343  # m/s


def calc_speed_of_sound(temperature=20):
    """
    Calculate speed of sound in air, in meters per second

    Calculate speed of sound for a given temperature
    in Celsius (Humidity has a negligible
    effect on speed of sound and so this functionality
    is not implemented)

    Args:
        temperature: ambient air temperature in Celsius

    Returns:
        the speed of sound in air in meters per second
    """
    return 331.3 * np.sqrt(1 + float(temperature) / 273.15)


def lorentz_ip(u, v=None):
    """
    Compute Lorentz inner product of two vectors

    For vectors `u` and `v`, the
    Lorentz inner product for 3-dimensional case is defined as

        u[0]*v[0] + u[1]*v[1] + u[2]*v[2] - u[3]*v[3]

    Or, for 2-dimensional case as

        u[0]*v[0] + u[1]*v[1] - u[2]*v[2]

    Args:
        u: vector with shape either (3,) or (4,)
        v: vector with same shape as x1; if None (default), sets v = u

    Returns:
        float: value of Lorentz IP"""
    if v is None:
        v = u

    if len(u) == 3 and len(v) == 3:
        c = [1, 1, -1]
        return sum([u[i] * v[i] * c[i] for i in range(len(u))])
    elif len(u) == 4 and len(v) == 4:
        c = [1, 1, 1, -1]
        return sum([u[i] * v[i] * c[i] for i in range(len(u))])

    return ValueError(f"length of x should be 3 or 4, was{len(u)}")


def travel_time(source, receiver, speed_of_sound):
    """
    Calculate time required for sound to travel from a souce to a receiver

    Args:
        source: cartesian location [x,y] or [x,y,z] of sound source, in meters
        receiver: cartesian location [x,y] or [x,y,z] of sound receiver, in meters
        speed_of_sound: speed of sound in m/s

    Returns:
        time in seconds for sound to travel from source to receiver
    """
    distance = np.linalg.norm(np.array(source) - np.array(receiver))
    return distance / speed_of_sound


def localize(receiver_locations, tdoas, algorithm, speed_of_sound):
    """
    Perform TDOA localization on a sound event. If there are not enough receivers to localize the event, return None.
    Args:
        receiver_locations: a list of [x,y,z] locations for each receiver
            locations should be in meters, e.g., the UTM coordinate system.
        tdoas: a list of TDOA times (onset times) for each recorder
            The times should be in seconds.
        speed_of_sound: speed of sound in m/s
        algorithm: the algorithm to use for localization
            Options: 'soundfinder', 'gillette', 'least_squares'.
            See the documentation for the functions soundfinder_localize, gillette_localize, and least_squares_localize for more detail on how each algorithm works.
    Returns:
        The estimated source location in meters, in the same number of dimensions as the receiver locations.
    """
    # check that there are enough receivers to localize the event
    ndim = len(receiver_locations[0])
    if len(receiver_locations) < ndim + 1:
        warnings.warn(
            f"Only {len(receiver_locations)} receivers. Need at least {ndim+1} to localize in {ndim} dimensions."
        )
        return None
    if algorithm == "soundfinder":
        estimate = soundfinder_localize(receiver_locations, tdoas, speed_of_sound)
    elif algorithm == "gillette":
        estimate = gillette_localize(receiver_locations, tdoas, speed_of_sound)
    elif algorithm == "least_squares":
        estimate = least_squares_localize(receiver_locations, tdoas, speed_of_sound)
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. Implemented for 'soundfinder', 'gillette' and 'least_squares'."
        )
    return estimate


def least_squares_localize(receiver_locations, arrival_times, speed_of_sound):
    """
    Use a least squares optimizer to perform TDOA localization on a sound event, based on a set of TDOA times and receiver locations.
    Args:
        receiver_locations: a list of [x,y,z] locations for each receiver
            locations should be in meters, e.g., the UTM coordinate system.
        arrival_times: a list of TDOA times (onset times) for each recorder
            The first receiver is the reference receiver, with arrival time 0.
        speed of sound: speed of sound in m/s
    Returns:
        The solution (x,y,z) in meters. In the same number of dimensions as the receiver locations.
    """

    # check that these delays are with reference to one receiver (the reference receiver).
    # We do this by checking that one of the arrival times is within float precision
    # of 0 (i.e. arrival at the reference)
    if not np.isclose(np.min(np.abs(arrival_times)), 0):
        raise ValueError(
            "Arrival times must be relative to a reference receiver. Therefore one arrival"
            " time must be 0 (corresponding to arrival at the reference receiver) None of your "
            "TDOAs are zero. Please check your arrival_times."
        )

    # make sure our inputs follow consistent format
    receiver_locations = np.array(receiver_locations).astype("float64")
    arrival_times = np.array(arrival_times).astype("float64")

    # find which is the reference receiver and reorder, so reference receiver is first
    ref_receiver = np.argmin(abs(arrival_times))
    ordered_receivers = np.roll(receiver_locations, -ref_receiver, axis=0)
    ordered_tdoas = np.roll(arrival_times, -ref_receiver, axis=0)
    # define the function to minimize, i..e. the TDOA residuals

    def fun(x, ordered_receivers, ordered_tdoas):
        # calculate the predicted TDOAs for each receiver
        time_of_flight = np.linalg.norm(ordered_receivers - x, axis=1) / speed_of_sound
        # the observed TDOAs are all relative to the first receiver, so minus the first receiver's TDOA
        predicted_tdoas = np.array(time_of_flight) - time_of_flight[0]

        # calculate the difference between the predicted and observed TDOAs (i.e. TDOA residuals)
        return np.array(predicted_tdoas) - ordered_tdoas

    # initial guess for the source location is the mean of the receiver locations
    x0 = np.mean(ordered_receivers, axis=0)

    # run the optimization
    res = least_squares(fun, x0, args=(ordered_receivers, ordered_tdoas))

    return res.x


def soundfinder_localize(
    receiver_locations,
    arrival_times,
    speed_of_sound,
    invert_alg="gps",  # options: 'gps'
    center=True,  # True for original Sound Finder behavior
    pseudo=True,  # False for original Sound Finder
):
    """
    Use the soundfinder algorithm to perform TDOA localization on a sound event
    Localize a sound event given relative arrival times at multiple receivers.
    This function implements a localization algorithm from the
    equations described in [1]. Localization can be performed in a global coordinate
    system in meters (i.e., UTM), or relative to recorder locations
    in meters.

    This implementation follows [2] with corresponding variable names.

    Args:
        receiver_locations: a list of [x,y,z] locations for each receiver
          locations should be in meters, e.g., the UTM coordinate system.
        arrival_times: a list of TDOA times (onset times) for each recorder
          The times should be in seconds.
        speed of sound: speed of sound in m/s
        invert_alg: what inversion algorithm to use (only 'gps' is implemented)
        center: whether to center recorders before computing localization
          result. Computes localization relative to centered plot, then
          translates solution back to original recorder locations.
          (For behavior of original Sound Finder, use True)
        pseudo: whether to use the pseudorange error (True) or
          sum of squares discrepancy (False) to pick the solution to return
          (For behavior of original Sound Finder, use False. However,
          in initial tests, pseudorange error appears to perform better.)
    Returns:
        The solution (x,y,z) in meters.

    [1]  Wilson, David R., Matthew Battiston, John Brzustowski, and Daniel J. Mennill.
    “Sound Finder: A New Software Approach for Localizing Animals Recorded with a Microphone Array.”
    Bioacoustics 23, no. 2 (May 4, 2014): 99–112. https://doi.org/10.1080/09524622.2013.827588.

    [2] Global locationing Systems handout, 2002
    http://web.archive.org/web/20110719232148/http://www.macalester.edu/~halverson/math36/GPS.pdf
    """

    # make sure our inputs follow consistent format
    receiver_locations = np.array(receiver_locations).astype("float64")
    arrival_times = np.array(arrival_times).astype("float64")

    # The number of dimensions in which to perform localization
    dim = receiver_locations.shape[1]
    assert dim in [2, 3], "localization only works in 2 or 3 dimensions"

    ##### Shift coordinate system to center receivers around origin #####
    if center:
        p_mean = np.mean(receiver_locations, 0)
        receiver_locations = np.array([p - p_mean for p in receiver_locations])

    ##### Compute B, a, and e #####
    # these correspond to [2] and are defined directly after equation 6

    # Find the pseudorange, rho, for each recorder
    # pseudorange (minus a constant) ~= distances from source to each receiver
    rho = np.array([arrival_times * (-1 * speed_of_sound)]).T

    # Concatenate the pseudorange column with x,y,z location to form matrix B
    B = np.concatenate((receiver_locations, rho), axis=1)

    # e is a vector of ones
    e = np.ones(receiver_locations.shape[0])

    # a is a 1/2 times a vector of squared Lorentz norms
    a = 0.5 * np.apply_along_axis(lorentz_ip, axis=1, arr=B)

    # choose between two algorithms to invert the matrix
    if invert_alg != "gps":
        raise NotImplementedError
        # original implementation of lstsq:
        # Compute B+ * a and B+ * e
        # using closest equivalent to R's solve(qr(B), e)
        # Bplus_e = np.linalg.lstsq(B, e, rcond=None)[0]
        # Bplus_a = np.linalg.lstsq(B, a, rcond=None)[0]

    else:  # invert_alg == 'gps' ('special' falls back to 'lstsq')
        ## Compute B+ = (B^T \* B)^(-1) \* B^T
        # B^T * B

        to_invert = np.matmul(B.T, B)

        try:
            inverted = np.linalg.inv(to_invert)

        except np.linalg.LinAlgError as err:
            # for 'gps' algorithm, simply fail
            # if invert_alg == "gps":
            warnings.warn("4")
            if "Singular matrix" in str(err):
                warnings.warn("5")
                warnings.warn(
                    "Singular matrix. Were recorders linear or on same plane? Exiting with NaN outputs",
                    UserWarning,
                )
                return [[np.nan]] * (dim)
            else:
                warnings.warn("6")
                raise

            # for 'special' algorithm: Fall back to lstsq algorithm
            # elif invert_alg == "special":  #
            #     warnings.warn("7")
            #     Bplus_e = np.linalg.lstsq(B, e, rcond=None)[0]
            #     Bplus_a = np.linalg.lstsq(B, a, rcond=None)[0]

        else:  # inversion of the matrix succeeded
            # B+ is inverse(B_transpose*B) * B_transpose
            # Compute B+ * a and B+ * e
            Bplus = np.matmul(inverted, B.T)
            Bplus_a = np.matmul(Bplus, a)
            Bplus_e = np.matmul(Bplus, e)

    ###### Solve quadratic equation for lambda #####

    # Compute coefficients
    cA = lorentz_ip(Bplus_e)
    cB = 2 * (lorentz_ip(Bplus_e, Bplus_a) - 1)
    cC = lorentz_ip(Bplus_a)

    # Compute discriminant
    disc = cB**2 - 4 * cA * cC
    # If discriminant is negative, set to zero to ensure
    # we get an answer, albeit not a very good one
    if disc < 0:
        disc = 0
        warnings.warn(
            "Discriminant negative--set to zero. Solution may be inaccurate. Inspect final value of output array",
            UserWarning,
        )

    # Compute options for lambda
    lamb = (-cB + np.array([-1, 1]) * np.sqrt(disc)) / (2 * cA)

    # Find solution u0 and solution u1
    ale0 = np.add(a, lamb[0] * e)
    u0 = np.matmul(Bplus, ale0)
    ale1 = np.add(a, lamb[1] * e)
    u1 = np.matmul(Bplus, ale1)

    # print('Solution 1: {}'.format(u0))
    # print('Solution 2: {}'.format(u1))

    ##### Return the better solution #####

    # Re-translate points
    if center:
        shift = np.append(p_mean, 0)  # 0 for b=error, which we don't need to shift
        u0 += shift
        u1 += shift

    # Select and return quadratic solution
    if pseudo:
        # Return the solution with the lower estimate of b, error in pseudorange
        # drop the estimate of b (error in pseudorange) from the return values,
        # returning just the location vector
        if abs(u0[-1]) <= abs(u1[-1]):
            return u0[0:-1]
        else:
            return u1[0:-1]

    else:
        # use the sum of squares discrepancy to choose the solution
        # This was the return method used in the original Sound Finder,
        # but it gives worse performance

        # Compute sum of squares discrepancies for each solution
        s0 = float(np.sum((np.matmul(B, u0) - np.add(a, lamb[0] * e)) ** 2))
        s1 = float(np.sum((np.matmul(B, u1) - np.add(a, lamb[1] * e)) ** 2))

        # Return the solution with lower sum of squares discrepancy
        # drop the final value, which is the estimate of b, error in the pseudorange,
        # returning just the location vector
        if s0 < s1:
            return u0[0:-1]
        else:
            return u1[0:-1]


def gillette_localize(receiver_locations, arrival_times, speed_of_sound):
    """
    Uses the Gillette and Silverman [1] localization algorithm to localize a sound event from a set of TDOAs.
    Args:
        receiver_locations: a list of [x,y] or [x,y,z] locations for each receiver
            locations should be in meters, e.g., the UTM coordinate system.
        arrival_times: a list of TDOA times (arrival times) for each receiver
            The times should be in seconds.
        speed_of_sound: speed of sound in m/s
    Returns:
        coords: a tuple of (x,y,z) coordinates of the sound source


    Algorithm from:
    [1] M. D. Gillette and H. F. Silverman, "A Linear Closed-Form Algorithm for Source Localization
    From Time-Differences of Arrival," IEEE Signal Processing Letters
    """

    # check that these delays are with reference to one receiver (the reference receiver).
    # We do this by checking that one of the arrival times is within float precision
    # of 0 (i.e. arrival at the reference)
    if not np.isclose(np.min(np.abs(arrival_times)), 0):
        raise ValueError(
            "Arrival times must be relative to a reference receiver. Therefore one arrival"
            " time must be 0 (corresponding to arrival at the reference receiver) None of your "
            "TDOAs are zero. Please check your arrival_times."
        )

    # make sure our inputs follow consistent format
    receiver_locations = np.array(receiver_locations).astype("float64")
    arrival_times = np.array(arrival_times).astype("float64")

    # The number of dimensions in which to perform localization
    dim = receiver_locations.shape[1]

    # find which is the reference receiver and reorder, so reference receiver is first
    ref_receiver = np.argmin(abs(arrival_times))
    ordered_receivers = np.roll(receiver_locations, -ref_receiver, axis=0)
    ordered_tdoas = np.roll(arrival_times, -ref_receiver, axis=0)

    # Gillette silverman solves Ax = w, where x is the solution vector, A is a matrix, and w is a vector
    # Matrix A according to Gillette and Silverman (2008)
    A = np.zeros((len(ordered_tdoas) - 1, dim + 1))
    for column in range(dim + 1):
        if column < dim:
            A[:, column] = ordered_receivers[0, column] - ordered_receivers[1:, column]
        elif column == dim:
            A[:, column] = ordered_tdoas[1:] * speed_of_sound

    # Vector w according to Gillette and Silverman (2008)
    # w = 1/2 (dm0^2 - xm^2 - ym^2 - zm^2 + x0^2 + y0^2 + z0^2)
    X02 = np.sum(ordered_receivers[0] ** 2)  # x0^2 + y0^2 + z0^2
    dmx = ordered_tdoas[1:] * speed_of_sound
    XM2 = np.sum(ordered_receivers**2, axis=1)[1:]

    vec_w = 0.5 * (dmx + X02 - XM2)

    answer = np.linalg.lstsq(A, vec_w.T, rcond=None)
    coords = answer[0][:dim]
    # pseudorange = answer[0][dim]
    # residuals = answer[1]

    return coords

def corr_sum(point,
             receiver_locations, pair_cross_corellations,
             speed_of_sound, samplerate,
             receiver_pairs, xc_length):
    """
    Evaluate the correlation sum at a given point
    potentially more efficient ways to do this...
    """
    sum_val = 0.0
    xc_midpoint = xc_length/2
    receiver_lag_idxs = samplerate/speed_of_sound * np.linalg.norm(receiver_locations - point, axis = 1)
    for i, j in receiver_pairs:
        xc_index = np.round(xc_midpoint + receiver_lag_idxs[i] - receiver_lag_idxs[j])
        if xc_index < 0 or xc_index >= xc_length:
            pass
        else:
            sum_val += pair_cross_corellations[(i, j)][xc_index]

    return sum_val

def correlation_sum_localize(receiver_locations, pair_cross_correlations: dict,
                  samplerate, speed_of_sound,
                  grid = (None, None, 1), apply_envelope: bool = True,
                  return_points = None, return_grid = False):
    """
    Localize an audio event using the Correlation Sum method presented in Birchfield and Gillmor. [1],
    applied to bioacoustic arrays in [2]
    Args:
        receiver_locations:
            locations (in 2d or 3d) of the recorder locations, as list of [x, y] or [x, y, z],
            or as Nx2 or Nx3 ndarray with N being the number of recorders
        pair_cross_correlations: Cross correlations between pairs of recorders as a dict with keys of recorder number pairs (as tuples), and values as the cross-correlations as ndarrays. Note that key pairs are sorted
        samplerate: The audio samplerate (fs) in Hz (samples/second)
        speed_of_sound: The speed of sound for the given recording conditions (in m/s)
        grid_params: grid starting point, ending point, and spacing for each dimension (in m)
            in the form (starting coordinates, ending coords, dimension spacing)
        apply_envelope: Whether a Hilbert envelope be applied to the cross-correlations (as is done in [2])
        return_pts: Number of possible locations to return (automatic if None)
        return_grid: Should the entire evaluated grid be returned

    Returns:
        Tuple containing coordinates of predicted sound event location(s),
        relative values of each location,
        and possibly the evaluated grid of coordinates if return_grid is true
        in the form: ((coords, value), ((gridx, gridy, ...), evaluated_values))

    [1] S. T. Birchfield and D. K. Gillmor, “Fast Bayesian acoustic localization,” in 2002 IEEE International Conference on Acoustics, Speech, and Signal Processing, May 2002, p. II-1793-II–1796. doi: 10.1109/ICASSP.2002.5744971.

    [2] T. C. Collier, A. N. G. Kirschel, and C. E. Taylor, “Acoustic localization of antbirds in a Mexican rainforest using a wireless sensor network,” The Journal of the Acoustical Society of America, vol. 128, no. 1, pp. 182–189, Jul. 2010, doi: 10.1121/1.3425729.

    """
    # Get inputs into correct format (and format check them)
    receiver_locations = np.asarray(receiver_locations).astype(np.float64)
    n_receivers = receiver_locations.shape[0]
    # check dims TODO

    num_recorders = receiver_locations.shape[0]
    assert(pair_cross_correlations.shape[0] == num_recorders)
    assert(pair_cross_correlations.shape[1] == num_recorders)

    # create an iterator of all unique recorder pairs (ignoring pairs with the same numbers)
    recorder_pairs = itertools.combinations(range(0, N), 2)

    # apply a hilbert envelope to each cross-correlation if requested
    if apply_envelope:
        for pair in recorder_pairs:
            # apply the hilbert envelope
            x_h = hilbert(pair_cross_correlations[pair])
            pair_cross_correlations[pair] = np.real(x_h) + np.imag(x_h)

    # create a grid (as an iterator instead of using ndarrays)
    start_pt, end_pt, spacing = grid_params
    # if start and end locations are not specified, grid will enclose all recorders
    if start_pt is None:
        start_pt = receiver_locations.min(axis=0)
    if end_pt is None:
        end_pt = receiver_locations.max(axis=0)

    # reshuffle start and end pts if needed so all start values are <= end values
    pt_range = np.vstack((start_pt, end_pt))
    start_pt = pt_range.min(axis=0)
    end_pt = pt_range.max(axis=0)

    # number of dimensions the grid has (can still have 2d array and 3d grid)
    n_grid_dims = start_pt.len()
    assert(n_grid_dims == end_pt.len())
    # augment recorder locations if needed
    if receiver_locations.shape[1] != 3:
        # add zero as last coord in each recorder location,
        # eg go from [[x1, y1], [x2, y2]] -> [[x1, y1, 0], [x2, y2, 0]]
        receiver_locations = np.concatenate((receiver_locations, np.zeros(n_receivers)[:, np.newaxis]),
                                            axis=1)

    if type(spacing) is not tuple:
        # copy spacing for each dimension
        spacing = (spacing,)*n_grid_dims

    # create a vector of points along each dim to evaluate at
    grid_x = np.linspace(start_pt[0], end_pt[0], num = (end_pt[0] - start_pt[0])/spacing[0])
    grid_y = np.arange(start_pt[1], end_pt[1], spacing[1])
    grid_z = np.zeros(1) if (n_grid_dims == 2) else np.arange(start_pt[2], end_pt[2], spacing[2])

    # create actual "grid" as an iterator by taking cartesian product of all axes
    grid_iter = itertools.product(grid_x, grid_y, grid_z)

    # upsample (potentially enveloped) cross-correlations to an appropriate spatial resolution
    # we want the temporal resolution of the cross-correlations to be at least Nx the
    # spacing
    resolution_multiplier = 2 # trying 2x multiplier ??
    min_spacing = min(spacing) # minimum spacing

    # to calculate the samplerate we need for a given spatial resolution (approx)
    # target_samplerat = speed_of_sound [m/s]/min_spacing[m] * resolution_multiplier = [Hz]
    goal_samplerate = speed_of_sound/min_spacing * resolution_multiplier

    # upsample to target samplerate
    if goal_samplerate < samplerate:
        for pair in recorder_pairs:
            x = pair_cross_correlations[pair]
            if apply_envelope:
                # if we have an envelope we can more likely get away with simpler interpolation as it's smoother
                pair_cross_correlations[pair] = librosa.resample(x, samplerate,
                                                                 goal_samplerate,
                                                                 res_type='soxr_qq')
            else:
                pair_cross_correlations[pair] = librosa.resample(x, samplerate,
                                                                 goal_samplrate,
                                                                 res_type='soxr_hq')

    samplerate = goal_samplerate
    xc_length = pair_cross_correlations[(0, 1)].shape[0]

    # create function handle to calculate it based only on point (can potentially parallelize with starmap later) - we can call this on only the point (also can use starmap on this)
    corr_sum_handle = functools.partial(corr_sum, receiver_locations=receiver_locations,
                                        pair_cross_correlations=pair_cross_correlations,
                                        speed_of_sound=speed_of_sound, samplerate=samplerate
                                        receiver_pairs=recorder_pairs, xc_length=xc_length)

    # final grid of evaluated points
    x_size = grid_x.len()
    y_size = grid_y.len()
    z_size = grid_z.len()
    xc_sum_grid = np.zeros((x_size, y_size, z_size))

    index_iter = itertools.product(range(x_size), range(y_size), range(z_size))
    # can potentially replace with mp.Array and a pool to parallelize
    for (x_i, y_i, z_i), point in zip(index_iter, grid_iter):
        xc_sum_grid[x_i, y_i, z_i] = corr_sum_handle(np.asarray(point))

    # find all local maxima in the resulting grid... TODO
    # find global maxima of grid:
    i_max = xc_sum_grid.argmax()
    p_max = (grid_x[i_max[0]], grid_y[i_max[1]], grid_z[i_max[2]]) # max point
    v_max = xc_sum_grid.max() # maximum value

    if return_grid:
        return (p_max, v_max), ((grid_x, grid_y, grid_z), xc_sum_grid)
    else:
        return pmax, vmax
