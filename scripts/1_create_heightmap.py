import logging
import math
import h5py
import numpy as np
import time
import pickle

from utils import helper
from config_and_logging import CONFIG, DEBUG_MODE, USE_FULL_VOLUME_LENGTH, CUSTOM_VOLUME_LENGTH, CUSTOM_VOLUME_START

# Read constants from the config file
RELATIVE_DIAMETER_OF_ELLIPSE_IN_VOLUME = CONFIG.getfloat('Height Map', 'relative_diameter_of_ellipse_in_volume')
RESOLUTION_FACTOR = CONFIG.getfloat('Height Map', 'resolution_factor')
STEP_LENGTH = CONFIG.getfloat('Height Map', 'step_length')
SCAN_RADIUS_MARGIN_FACTOR = eval(CONFIG.get('Height Map', 'scan_radius_margin_factor'))
THRESHOLD_FACTOR = CONFIG.getfloat('Height Map', 'threshold_factor')

# Read pre-examination pickle
pre_examination = pickle.load(open(helper.get_path_of_existing_file_from_config('pre-examination', CONFIG), 'rb'))
OTSU_THRESHOLD = pre_examination['otsu_threshold']
ROSTRUM_IS_POINTED_TO_THE_FRONT = pre_examination['rostrum_is_pointed_to_front']
AXIS_CENTER_X = pre_examination['axis_center_x']
AXIS_CENTER_Y = pre_examination['axis_center_y']


def is_inside_xy(current_point: tuple, volume_shape_xy: tuple) -> bool:
    """
    Check if a given point is inside the volume boundaries.

    Args:
        current_point (tuple): The coordinates of the 2D point.
        volume_shape_xy (tuple): The shape of the volume in the xy-plane.

    Returns:
        bool: True if the point is inside the volume, False otherwise.
    """
    return np.all((0 <= np.array(current_point)) & (np.array(current_point) < volume_shape_xy))


def calculate_ellipse_circumference(a: float, b: float) -> float:
    """
    Calculate the circumference of an ellipse given its semi-major and semi-minor axes.
    The formula is from https://stackoverflow.com/questions/42310956/how-to-calculate-the-perimeter-of-an-ellipse

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.

    Returns:
        float: Circumference of the ellipse.
    """
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def calculate_heightmap(volume: np.ndarray, vtk_helper: helper.VtkRostrumHelper, start: int, length: int) -> tuple:
    """
    Create a 2D height map from a 3D volume. The height map represents the distance from each point in the volume
    to a central point, measured along lines radiating out from the center at different angles.

    Args:
        volume (np.ndarray): The 3D volume, given as a 3D numpy array.
        vtk_helper (helper.VtkRostrumHelper): VTK helper object.
        start (int): The index of the first slice to be included in the height map.
        length (int): The length of the whole volume.

    Returns:
        tuple: A tuple containing the height map array (a 2D numpy array) and the ellipse circumference (a float).
    """
    ellipse_circumference = calculate_ellipse_circumference(volume.shape[2] / 2 * RELATIVE_DIAMETER_OF_ELLIPSE_IN_VOLUME,
                                                            volume.shape[1] / 2 * RELATIVE_DIAMETER_OF_ELLIPSE_IN_VOLUME)

    # Initialize an empty array for the height map with the desired resolution factor
    heightmap = np.zeros((int(volume.shape[0] / RESOLUTION_FACTOR),
                          int(ellipse_circumference / RESOLUTION_FACTOR)))
    logging.debug(f'Ellipse circumference: {ellipse_circumference}')
    logging.info(f'Creating height map with shape {heightmap.shape}. This may take a few minutes...')

    # # Calculate the center point of the volume by taking half of its dimensions
    # if USE_VOLUME_CENTER_AS_AXIS_CENTER_INSTEAD_OF_ROSTRUM_TIP:
    #     center_point_start = np.array([volume.shape[1] / 2, volume.shape[2] / 2])
    # else:
    #     center_point_start = np.array([AXIS_CENTER_Y, AXIS_CENTER_X])

    # Calculate the center point of the volume by taking half of its dimensions and the rostrum tip
    axis_center_point_start = np.array([volume.shape[1] / 2, volume.shape[2] / 2])
    axis_center_point_end = np.array([AXIS_CENTER_Y, AXIS_CENTER_X])

    volume_shape_xy = volume[0].shape
    max_possible_radius = max(volume.shape[1], volume.shape[2])
    factored_threshold = OTSU_THRESHOLD * THRESHOLD_FACTOR

    # Loop over the i_angle of the ellipse
    n_angles = heightmap.shape[1]
    for i_angle in range(n_angles):
        # Convert the i_angle index to an angle in radians by dividing it by the number of points and
        # multiplying by 2 * pi for value in radians
        angle_rad = i_angle / n_angles * 2 * np.pi

        # Calculate the unit vector along the angle by taking its cosine and sine components
        unit_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])

        # Initialize the current radius to the maximum possible value,
        # which is either the width or the height of the volume.
        most_current_radius = max(volume.shape[1], volume.shape[2])

        # Loop over the z-axis of the volume
        for z in range(heightmap.shape[0]):

            z_relative_progress_in_whole_volume = (start + z * RESOLUTION_FACTOR) / length
            axis_center_for_current_z = z_relative_progress_in_whole_volume ** 2 \
                * (axis_center_point_end - axis_center_point_start) + axis_center_point_start

            # if volume tip is pointed to the front, the z axis is inverted
            if ROSTRUM_IS_POINTED_TO_THE_FRONT:
                z = heightmap.shape[0] - z - 1

            start_point = axis_center_for_current_z + unit_vector * (
                    most_current_radius + max_possible_radius * SCAN_RADIUS_MARGIN_FACTOR)

            current_point = start_point
            point_exists = True
            counter_not_yet_inside = 0

            # If it's not inside yet, move the current point backwards along the vector until it is inside the volume.
            while not is_inside_xy(current_point, volume_shape_xy) and point_exists:
                current_point = current_point - unit_vector * STEP_LENGTH
                if counter_not_yet_inside > max_possible_radius:
                    point_exists = False
                    current_point = axis_center_for_current_z
                counter_not_yet_inside += 1

            # Move the current point backwards along the vector until it reaches a value above the threshold.
            while point_exists and volume[int(z * RESOLUTION_FACTOR), int(current_point[0]), int(current_point[1])] < factored_threshold:
                current_point = current_point - unit_vector * STEP_LENGTH

                if not is_inside_xy(current_point, volume_shape_xy) or np.linalg.norm(current_point - axis_center_for_current_z) <= STEP_LENGTH:
                    point_exists = False

            # If a valid point exists, move it forward along the vector by one step.
            if point_exists and is_inside_xy(current_point, volume_shape_xy):
                current_point = current_point + unit_vector * STEP_LENGTH
                while not is_inside_xy(current_point, volume_shape_xy):
                    current_point = current_point - unit_vector

                while volume[int(z * RESOLUTION_FACTOR), int(current_point[0]), int(current_point[1])] < factored_threshold:
                    current_point = current_point - unit_vector
                    if not is_inside_xy(current_point, volume_shape_xy):
                        point_exists = False
                        # logging.debug("Point outside volume.")
                        break

            if not point_exists:
                current_point = axis_center_for_current_z

            # Calculate the distance from the current point to the center and store it in the height map
            most_current_radius = np.linalg.norm(current_point - axis_center_for_current_z)
            heightmap[z, i_angle] = most_current_radius

            # Add a line from the current point to the center point to the VTK helper
            if DEBUG_MODE == 'high':
                color = 'blue' if i_angle == 0 \
                    else 'red' if i_angle == n_angles - 1 \
                    else 'black'
                vtk_helper.add_polylines_actor(
                    [[[int(current_point[1]), int(current_point[0]), int(z * RESOLUTION_FACTOR)],
                      [int(axis_center_for_current_z[1]), int(axis_center_for_current_z[0]), int(z * RESOLUTION_FACTOR)]]], color=color)
    return heightmap, ellipse_circumference


def run_heightmap_creation() -> None:
    """
    Run the height map creation process. This involves reading a volume from an hdf5 file,
    creating a height map from the volume, and saving the height map to a new hdf5 file.
    """
    # Load a volume from hdf5 file and read metadata
    with h5py.File(helper.get_path_of_existing_file_from_config('volume', CONFIG), "r") as f:
        volume_dset = f["volume"]
        meta_grp = f["meta"]
        length = int(meta_grp.attrs["length"])
        data_type = meta_grp.attrs["data type"]
        start = 0 if USE_FULL_VOLUME_LENGTH \
            else CUSTOM_VOLUME_START if not ROSTRUM_IS_POINTED_TO_THE_FRONT \
            else length - CUSTOM_VOLUME_START - CUSTOM_VOLUME_LENGTH
        stop = length if USE_FULL_VOLUME_LENGTH \
            else (CUSTOM_VOLUME_LENGTH + start) if not ROSTRUM_IS_POINTED_TO_THE_FRONT \
            else length - CUSTOM_VOLUME_START
        volume = volume_dset[start:stop]
        logging.debug(f"volume.shape: {volume.shape}")

    if DEBUG_MODE == 'high':
        vtk_helper = helper.VtkRostrumHelper()
        vtk_helper.add_volume_as_data(volume, data_type, OTSU_THRESHOLD)
    else:
        vtk_helper = None

    # Call the create_height_map function and store the result
    heightmap, ellipse_circumference = calculate_heightmap(volume, vtk_helper, start, length)

    with h5py.File(helper.create_path_for_new_file_from_config('heightmap', CONFIG), 'w') as outfile:
        dataset = outfile.create_dataset('data', data=heightmap)
        dataset.attrs['resolution_factor'] = RESOLUTION_FACTOR  # save with heightmap to avoid bugs when changing config
        dataset.attrs['volume.shape'] = volume.shape
        dataset.attrs['volume.whole_length'] = length
        dataset.attrs['volume.start'] = start
        dataset.attrs['ellipse_circumference'] = ellipse_circumference

    helper.show_2d_array_as_image(heightmap)

    if DEBUG_MODE == 'high':
        vtk_helper.start_interactor()


if __name__ == '__main__':
    tic = time.perf_counter()
    script_name = __file__.split('\\')[-1]
    logging.info(f'started {script_name}...\n')
    run_heightmap_creation()
    logging.info(f"finished. {time.perf_counter() - tic:1.1f}s")
    exit(0)
