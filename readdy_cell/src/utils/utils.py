import readdy
import numpy as np
from skimage.feature import canny
from skimage.morphology import skeletonize

ut = readdy.units

def from_vec3(vec):
    if not isinstance(vec, np.ndarray):
        vec = np.array([vec[0], vec[1], vec[2]])
    return vec

def get_projection_coordinate(pos, v1, v2):
    """
    Calculates the projection of the vector from v1 to pos onto the vector defined by points v1 and v2,
    ensuring the coordinate lies along the segment between v1 and v2.

    Parameters:
    pos (np.ndarray): Position to project.
    v1 (np.ndarray): Starting point of the vector defining the segment.
    v2 (np.ndarray): Ending point of the vector defining the segment.

    Returns:
    np.ndarray: The projected coordinate constrained to lie between v1 and v2.
    """
    a = pos - v1  # Vector from v1 to pos
    b = v2 - v1  # Vector from v1 to v2 (the direction of the line segment)
    proj = np.dot(a, b) / np.dot(b, b) * b
    proj_clipped = np.clip(proj, 0, np.linalg.norm(b)) * (b / np.linalg.norm(b))
    coordinate = v1 + proj_clipped
    return coordinate

def rate_to_probability(rate, timestep, **kwargs):
    time_unit = kwargs.get("time_unit", "s")

    assert type(rate) in [float, ut.Quantity], "Rate must be a float, or Quantity object."
    assert type(timestep) in [float, ut.Quantity], "Timestep must be a float, or Quantity object."

    if isinstance(rate, ut.Quantity):
        rate = rate.to(f"{time_unit}^-1").magnitude

    if isinstance(timestep, ut.Quantity):
        timestep = timestep.to(time_unit).magnitude
    # print(f"(rate_to_probability) Rate: {rate}, Timestep: {timestep}")
    return 1 - np.exp(-rate * timestep)

def calculate_motor_step_probability(target_velocity, step_length, timestep, **kwargs):
    """ Calculates the probability of a motor taking a step based on the target velocity. """
    if not isinstance(target_velocity, ut.Quantity):
        target_velocity = target_velocity * ut.um / ut.s
    if not isinstance(step_length, ut.Quantity):
        step_length = step_length * ut.um

    # Calculate the step rate based on the target velocity
    step_rate = target_velocity / step_length
    p_step = rate_to_probability(step_rate, timestep, **kwargs)
    return p_step

def reformat_vec3_trajectory(vec):
    vec_new = np.zeros(shape=(len(vec), len(vec[0]), 3), dtype=np.float32)
    for i in range(len(vec)):
        for j in range(len(vec[i])):
            vec_new[i][j] = np.array([vec[i][j][0], vec[i][j][1], vec[i][j][2]])
    return vec_new

def convert_units(d, time_unit='s', length_unit='um'):
    for key, value in d.items():
        if isinstance(value, dict):
            convert_units(value)
        elif isinstance(value, ut.Quantity):
            if value.dimensionality == ut.Unit(f'1/{time_unit}').dimensionality:
                d[key] = value.to(f'1/{time_unit}').magnitude
            elif value.dimensionality == ut.Unit(length_unit).dimensionality:
                d[key] = value.to(length_unit).magnitude
        else:
            continue
    return d

def permute_array(data):
    return [np.copy(data), np.rot90(data, axes=(0, 1)), np.rot90(data, axes=(0, 2))]

def combine_permuted_arrays(data):
    data[1] = np.rot90(data[1], axes=(1, 0))
    data[2] = np.rot90(data[2], axes=(2, 0))
    return np.logical_or.reduce(data)

def canny_3d(data, permute_axes=False):
    """An algorithm which performs edge detection in 3D using the canny algorithm."""
    canny_arrays = []
    data_arrays = permute_array(data) if permute_axes else [data]
    for array in data_arrays:
        edges = np.zeros(array.shape)
        for i in range(0, array.shape[0]):
            edges[i, :, :] = canny(array[i, :, :],
                                    sigma=1,
                                    low_threshold=0.1,
                                    high_threshold=0.2).astype(np.uint8)
        canny_arrays.append(edges)
    edge_data = combine_permuted_arrays(canny_arrays) if permute_axes else canny_arrays[0]
    return edge_data

def skeletonize_3d(data):
    """An algorithm which performs skeletonization in 3D."""
    skeleton = np.zeros(shape=data.shape)
    for i in range(data.shape[0]):
        skeleton[i, :, :] = skeletonize(data[i, :, :])
    return skeleton

def fill_top_bottom(data):
    """ Fill holes in the top and bottom surfaces of a 3D array. """
    from scipy.ndimage import binary_fill_holes
    filled_data = np.copy(data)
    filled_data[0, :, :] = binary_fill_holes(data[0, :, :])
    filled_data[-1, :, :] = binary_fill_holes(data[-1, :, :])
    return filled_data

class RunningAverage:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.total += value
        self.count += 1
        return self.average()

    def average(self):
        return self.total / self.count if self.count != 0 else 0.0
