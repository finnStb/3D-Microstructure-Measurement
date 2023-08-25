import configparser
import csv
import logging
import math
import os
import pickle
import time
from datetime import datetime

import h5py
import numpy as np
import vtk
import vtkmodules.vtkRenderingOpenGL2
from scipy.ndimage import gaussian_filter
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkConvexPointSet,
    vtkPolyData
)
from vtkmodules.vtkFiltersSources import vtkRegularPolygonSource, vtkSphereSource
from vtkmodules.vtkFiltersSources import vtkDiskSource
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkGlyph3DMapper
)

from helper import extract_cells_closest_to_point_from_polydata
from utils import helper
from config_and_logging import CONFIG, DEBUG_MODE, USE_FULL_VOLUME_LENGTH, CUSTOM_VOLUME_LENGTH, CUSTOM_VOLUME_START

# Read pre-examination pickle
pre_examination = pickle.load(open(helper.get_path_of_existing_file_from_config('pre-examination', CONFIG), 'rb'))
OTSU_THRESHOLD = pre_examination['otsu_threshold']
PERCENTILE_25 = pre_examination['percentile_25']
PERCENTILE_50 = pre_examination['percentile_50']
PERCENTILE_75 = pre_examination['percentile_75']
ROSTRUM_IS_POINTED_TO_THE_FRONT = pre_examination['rostrum_is_pointed_to_front']
AXIS_CENTER_X = pre_examination['axis_center_x']
AXIS_CENTER_Y = pre_examination['axis_center_y']

# Read constants from the config file
N_DIRECTIONS = CONFIG.getint('Scan', 'n_directions')
HORIZONTAL_RESOLUTION = CONFIG.getfloat('Scan', 'horizontal_resolution')
VERTICAL_RESOLUTION = CONFIG.getfloat('Scan', 'vertical_resolution')
MAX_EXPECTED_RADIUS = CONFIG.getint('Scan', 'max_expected_radius')
PERFECT_FIT_MINIMUM_SCORE = CONFIG.getfloat('Scan', 'perfect_fit_minimum_score')
GAUSSIAN_FILTER_SIGMA = CONFIG.getfloat('Scan', 'gaussian_filter_sigma')
INWARDS_SHIFT_MULTIPLIER = CONFIG.getfloat('Scan', 'inwards_shift_multiplier')
RADIUS_MULTIPLIER = CONFIG.getfloat('Scan', 'radius_multiplier')
ENDPOINTS_INIT_FACTOR = CONFIG.getfloat('Scan', 'endpoints_init_factor')
# S_VALUE_SIGMOID_COEFF_1 = CONFIG.getfloat('Scan', 's_value_sigmoid_coeff_1')
# S_VALUE_SIGMOID_OFFSET_1 = CONFIG.getfloat('Scan', 's_value_sigmoid_offset_1')
# S_VALUE_SIGMOID_COEFF_2 = CONFIG.getfloat('Scan', 's_value_sigmoid_coeff_2')
# S_VALUE_SIGMOID_OFFSET_2 = CONFIG.getfloat('Scan', 's_value_sigmoid_offset_2')
P_NEW_VAL_FACTOR_START = CONFIG.getfloat('Scan', 'p_new_val_factor_start')
P_NEW_VAL_FACTOR_END = CONFIG.getfloat('Scan', 'p_new_val_factor_end')
P_NORMAL_FACTOR_MULTIPLIER = CONFIG.getfloat('Scan', 'p_normal_factor_multiplier')
P_EMA_DIFF_ADJUST_FACTOR = CONFIG.getfloat('Scan', 'p_ema_diff_adjust_factor')
P_CURRENT_TOOTH_RADIUS_WEIGHT = CONFIG.getfloat('Scan', 'p_current_tooth_radius_weight')
P_PREVIOUS_EMA_RADIUS_WEIGHT = CONFIG.getfloat('Scan', 'p_previous_ema_radius_weight')
P_MEDIAN_TOOTH_RADIUS_WEIGHT = CONFIG.getfloat('Scan', 'p_median_tooth_radius_weight')
P_EXPECTED_RADIUS_WEIGHT = CONFIG.getfloat('Scan', 'p_expected_radius_weight')
C_SCAN_PROGRESS_THRESHOLD = CONFIG.getfloat('Scan', 'c_scan_progress_threshold')
C_MIN_LAYER_INDEX = CONFIG.getfloat('Scan', 'c_min_layer_index')
C_OFFSET_THRESHOLD = CONFIG.getfloat('Scan', 'c_offset_threshold')
C_ENDPOINT_PREDICTION_WEIGHT = CONFIG.getfloat('Scan', 'c_endpoint_prediction_weight')
C_EXPECTED_RADIUS_WEIGHT = CONFIG.getfloat('Scan', 'c_expected_radius_weight')
C_SMOOTHING_FACTOR_BASE = CONFIG.getfloat('Scan', 'c_smoothing_factor_base')
C_LAYER_RADIUS_MEDIAN_MULTIPLIER = CONFIG.getfloat('Scan', 'c_layer_radius_median_multiplier')
C_EXPECTED_RADIUS_MULTIPLIER = CONFIG.getfloat('Scan', 'c_expected_radius_multiplier')
C_CORRECTION_FACTOR_AT_START = CONFIG.getfloat('Scan', 'c_correction_factor_at_start')
C_CORRECTION_FACTOR_AT_END = CONFIG.getfloat('Scan', 'c_correction_factor_at_end')

# get the specific dataset settings for score calculation
VOLUME_FILENAME = CONFIG.get('General', 'volume_filename')
config_dataset_settings = configparser.ConfigParser()
file_path_dataset_settings = f"../settings/specific_dataset_scan_settings/{VOLUME_FILENAME}.ini"
if os.path.exists(file_path_dataset_settings):
    config_dataset_settings.read(file_path_dataset_settings)
else:
    S_INITIAL_VALUE_SIGMOID_COEFF_1 = CONFIG.getfloat('Scan', 's_initial_value_sigmoid_coeff_1')
    S_INITIAL_VALUE_SIGMOID_COEFF_2 = CONFIG.getfloat('Scan', 's_initial_value_sigmoid_coeff_2')
    S_INITIAL_CONTRAST_SIGMOID_COEFF = CONFIG.getfloat('Scan', 's_initial_contrast_sigmoid_coeff')
    S_INITIAL_CONTRAST_SIGMOID_OFFSET = CONFIG.getfloat('Scan', 's_initial_contrast_sigmoid_offset')
    S_INITIAL_PROXIMITY_SIGMOID_COEFF = CONFIG.getfloat('Scan', 's_initial_proximity_sigmoid_coeff')
    S_INITIAL_PROXIMITY_SIGMOID_OFFSET = CONFIG.getfloat('Scan', 's_initial_proximity_sigmoid_offset')
    S_INITIAL_VALUE_WEIGHT = CONFIG.getfloat('Scan', 's_initial_value_weight')
    S_INITIAL_CONTRAST_WEIGHT = CONFIG.getfloat('Scan', 's_initial_contrast_weight')
    S_INITIAL_PROXIMITY_WEIGHT = CONFIG.getfloat('Scan', 's_initial_proximity_weight')
    config_dataset_settings['DEFAULT'] = {
        's_value_sigmoid_coeff_1': S_INITIAL_VALUE_SIGMOID_COEFF_1,
        's_value_sigmoid_coeff_2': S_INITIAL_VALUE_SIGMOID_COEFF_2,
        's_contrast_sigmoid_coeff': S_INITIAL_CONTRAST_SIGMOID_COEFF,
        's_contrast_sigmoid_offset': S_INITIAL_CONTRAST_SIGMOID_OFFSET,
        's_proximity_sigmoid_coeff': S_INITIAL_PROXIMITY_SIGMOID_COEFF,
        's_proximity_sigmoid_offset': S_INITIAL_PROXIMITY_SIGMOID_OFFSET,
        's_value_weight': S_INITIAL_VALUE_WEIGHT,
        's_contrast_weight': S_INITIAL_CONTRAST_WEIGHT,
        's_proximity_weight': S_INITIAL_PROXIMITY_WEIGHT,
    }
    with open(file_path_dataset_settings, 'w') as configfile:
        config_dataset_settings.write(configfile)

S_VALUE_SIGMOID_COEFF_1 = config_dataset_settings.getfloat('DEFAULT', 's_value_sigmoid_coeff_1')
S_VALUE_SIGMOID_COEFF_2 = config_dataset_settings.getfloat('DEFAULT', 's_value_sigmoid_coeff_2')
S_CONTRAST_SIGMOID_COEFF = config_dataset_settings.getfloat('DEFAULT', 's_contrast_sigmoid_coeff')
S_CONTRAST_SIGMOID_OFFSET = config_dataset_settings.getfloat('DEFAULT', 's_contrast_sigmoid_offset')
S_PROXIMITY_SIGMOID_COEFF = config_dataset_settings.getfloat('DEFAULT', 's_proximity_sigmoid_coeff')
S_PROXIMITY_SIGMOID_OFFSET = config_dataset_settings.getfloat('DEFAULT', 's_proximity_sigmoid_offset')
S_VALUE_WEIGHT = config_dataset_settings.getfloat('DEFAULT', 's_value_weight')
S_CONTRAST_WEIGHT = config_dataset_settings.getfloat('DEFAULT', 's_contrast_weight')
S_PROXIMITY_WEIGHT = config_dataset_settings.getfloat('DEFAULT', 's_proximity_weight')

sum_of_weights = P_CURRENT_TOOTH_RADIUS_WEIGHT + P_PREVIOUS_EMA_RADIUS_WEIGHT \
                 + P_MEDIAN_TOOTH_RADIUS_WEIGHT + P_EXPECTED_RADIUS_WEIGHT
if sum_of_weights != 1.0:
    raise ValueError(f'The sum of the weights for the tooth radius prediction must be 1.0. The current sum is '
                     f'{sum_of_weights}. (p_..._weight constants in config.ini)')


def sigmoid(t) -> float:
    """Sigmoid function.

    sigmoid(t) = 1 / (1 + e^(-t))

    :param t: input value
    :return: sigmoid(t)
    """
    return 1 / (1 + math.exp(-t))


def get_closest_point_on_polydata_to_given_point(point, polydata, point_locator):
    """
    Get the closest point on a polydata object to a given point.

    This function first extracts the cells closest to the point from the given polydata. Then, it uses a cell
    locator to find the closest point on the triangle of the polydata to the given point.

    :param point: The reference point to which the closest point on the polydata is to be found.
    :type point: list or tuple or ndarray
    :param polydata: The polydata on which the closest point is to be found.
    :type polydata: vtk.vtkPolyData
    :param point_locator: A locator used to find the closest cells to the point in the polydata.
    :type point_locator: vtk.vtkPointLocator
    :return: The closest point on the polydata to the given point.
    :rtype: list
    """
    # Extract cells closest to the point from the polydata
    extracted_poly_data = extract_cells_closest_to_point_from_polydata(polydata, point, point_locator)

    # Create the cell locator
    cell_locator = vtk.vtkCellLocator()
    cell_locator.SetDataSet(extracted_poly_data)

    # Set the number of cells per bucket for the locator
    cell_locator.SetNumberOfCellsPerBucket(100)
    cell_locator.BuildLocator()

    # Initialize values for finding the closest point
    closest_point = [0] * 3
    cell_id = vtk.mutable(-1)
    sub_id = vtk.mutable(-1)
    dist_squared = vtk.mutable(0.0)

    # Find the closest point on the triangle to the given point
    cell_locator.FindClosestPoint(point, closest_point, cell_id, sub_id, dist_squared)

    return closest_point


def volume_chunk_around_tooth(volume, tooth_tip: np.array, base_point_under_tip: np.array):  # todo in 03 schieben
    """
    Extract a chunk of the volume around a tooth.

    This function returns a chunk of the given volume around the tooth. The chunk is as small as possible
    while still containing the entire tooth. The volume boundaries are determined based on the tooth tip and
    base point parameters.

    :param volume: The volume from which to extract the tooth chunk.
    :type volume: ndarray
    :param tooth_tip: A point at the tip of the tooth.
    :type tooth_tip: ndarray
    :param base_point_under_tip: A point at the base of the tooth, under the tip.
    :type base_point_under_tip: ndarray
    :return: A chunk of the volume containing the tooth and the bounds of the chunk in the volume.
    :rtype: tuple (ndarray, dict)
    """
    # todo config
    tip_multiplier = 0.4
    base_multiplier = 0.8

    # Initialize the bounds to the size of the volume
    x_bounds = [0, len(volume[0][0])]
    y_bounds = [0, len(volume[0])]
    z_bounds = [0, len(volume)]

    # Compute the expected height of the tooth based on the tip and base points
    expected_height = np.linalg.norm(tooth_tip - base_point_under_tip)

    # Compute the bounds around the tooth tip and base
    tip_bound_lower = tooth_tip - expected_height * tip_multiplier
    tip_bound_higher = tooth_tip + expected_height * tip_multiplier
    base_bound_lower = base_point_under_tip - expected_height * base_multiplier
    base_bound_higher = base_point_under_tip + expected_height * base_multiplier

    # Update the bounds to be within the volume and to encompass the tooth
    x_bounds = [max(x_bounds[0], min(tip_bound_lower[0], base_bound_lower[0])),
                min(x_bounds[1], max(tip_bound_higher[0], base_bound_higher[0]))]
    y_bounds = [max(y_bounds[0], min(tip_bound_lower[1], base_bound_lower[1])),
                min(y_bounds[1], max(tip_bound_higher[1], base_bound_higher[1]))]
    z_bounds = [max(z_bounds[0], min(tip_bound_lower[2], base_bound_lower[2])),
                min(z_bounds[1], max(tip_bound_higher[2], base_bound_higher[2]))]

    # Round the bounds to the nearest integer
    x_bounds = np.round(x_bounds).astype(int)
    y_bounds = np.round(y_bounds).astype(int)
    z_bounds = np.round(z_bounds).astype(int)

    # Create a dictionary to hold the bounds
    bounds = {"x": x_bounds, "y": y_bounds, "z": z_bounds}

    # Extract the chunk of the volume around the tooth
    chunk = volume[z_bounds[0]:z_bounds[1], y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]]

    return chunk, bounds


def contrast_adjusted_sigmoid(value, a, b):
    # Apply the linear transformation
    value = a * value + b
    # Apply the sigmoid function and return the result
    return sigmoid(value)


def volume_index(point):
    """
    Converts a 3D point into a reversed index for volume access.

    Parameters:
    point (tuple): The 3D point to be converted. Should be in the format (x, y, z).

    Returns:
    tuple: The reversed and rounded 3D point for volume indexing.
    """
    return round(point[2]), round(point[1]), round(point[0])


def average_value_in_volume_at_coords(volume, coord_list):
    """
    Calculates the average value in a 3D volume at the specified coordinates.

    Parameters:
    volume (ndarray): The 3D volume of data.
    coord_list (list): The list of 3D points to calculate the average value for.

    Returns:
    float: The average value at the specified coordinates in the volume.
    """
    values = [volume[volume_index(coord)] for coord in coord_list]
    return sum(values) / len(values)


def is_point_within_bounds(point, shape):
    """
    Checks if a 3D point is within the bounds of a volume.

    Parameters:
    point (tuple): The 3D point to be checked. Should be in the format (z, y, x).
    volume_shape (tuple): The shape of the volume to check the point against. Should be in the format (x, y, z).

    Returns:
    bool: True if the point is within the bounds of the volume, False otherwise.
    """
    return 0 <= round(point[0]) < shape[2] and 0 <= round(point[1]) < shape[1] and 0 <= round(point[2]) < shape[0]


# def get_expected_radius(relative_tooth_scan_progress, height):
#     return math.sqrt(relative_tooth_scan_progress) / 3 * height  # todo 3 config


def calculate_radial_direction_vectors(normal_vector, num_directions):
    """
    Calculate radial direction vectors in a plane orthogonal to a given normal vector.

    Args:
    normal_vector (np.array): The surface normal vector.
    num_directions (int): The number of radial directions to calculate.

    Returns:
    radial_direction_vectors (list): A list of radial direction vectors.
    """

    # Select an arbitrary vector that is not parallel to the normal vector
    tangent_vector = np.array([1, 0, 0])

    # If the chosen tangent vector is parallel to the normal vector, choose a different tangent
    if abs(np.dot(normal_vector, tangent_vector)) == 1:
        tangent_vector = np.array([0, 1, 0])

    # Calculate the cross product of the normal and tangent vectors to create a new vector orthogonal to both
    tangent_vector = np.cross(normal_vector, tangent_vector)
    tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)  # normalize to unit length

    # Calculate another vector orthogonal to both the normal and tangent vectors
    binormal_vector = np.cross(normal_vector, tangent_vector)
    binormal_vector = binormal_vector / np.linalg.norm(binormal_vector)

    # Prepare the calculation of the radial direction vectors
    radial_direction_vectors = []
    angle_step = 2 * np.pi / num_directions  # angle between each radial direction vector

    # Calculate the radial direction vectors
    for direction in range(num_directions):
        radial_vector = tangent_vector * np.cos(direction * angle_step) + binormal_vector * np.sin(
            direction * angle_step)
        radial_direction_vectors.append(radial_vector)

    return radial_direction_vectors


def get_gauss_value(point, volume_chunk, volume_chunk_bounds):
    """
    This function calculates the Gauss value for a given point inside a volume chunk.

    Args:
        point (numpy.ndarray): 3D coordinates of the point.
        volume_chunk (numpy.ndarray): 3D array representing the volume chunk.
        volume_chunk_bounds (dict): Dictionary containing the bounds of the volume chunk in 'x', 'y', and 'z' direction.

    Returns:
        value_gauss (int): The gauss value of the point inside the volume chunk. Returns 0 if the point is outside the volume chunk bounds.
    """
    # Extract the minimum bounds for x, y, and z from the volume chunk bounds
    x_min_chunk = volume_chunk_bounds["x"][0]
    y_min_chunk = volume_chunk_bounds["y"][0]
    z_min_chunk = volume_chunk_bounds["z"][0]

    # Calculate the indices in the volume chunk for the given point
    gauss_x = round(point[0]) - x_min_chunk
    gauss_y = round(point[1]) - y_min_chunk
    gauss_z = round(point[2]) - z_min_chunk

    # Check whether the indices are within the bounds of the volume chunk
    if is_point_within_bounds([gauss_x, gauss_y, gauss_z], volume_chunk.shape):
        # If the indices are within the bounds, get the value at the point in the volume chunk
        value_gauss = round(volume_chunk[gauss_z][gauss_y][gauss_x])
    else:
        # If the indices are outside the bounds, set the gauss value to 0
        value_gauss = 0

    return value_gauss


def calculate_score_for_endpoint(endpoints, direction, volume, volume_chunk, volume_chunk_bounds,
                                 radial_direction_vectors, endpoints_predicted, score_contrast_a, score_contrast_b):
    """
    Calculates a score for an endpoint based on its value, contrast, and proximity.
    The sigmoid function is used to require minimum values for each score parameter
    and to get a final score between 0 and 1 for better comparability.

    Parameters:
    endpoints (np.array): Array of endpoints.
    direction (int): The direction to consider for the endpoint.
    volume (np.array): The volume.
    volume_chunk (np.array): The smoothed volume chunk of the tooth.
    volume_chunk_bounds (np.array): The bounds of the smoothed volume chunk.
    radial_direction_vectors (np.array): Radial direction vectors.
    i_horizontal (int): Horizontal index.
    endpoints_predicted (np.array): Array of predicted endpoints.

    Returns:
    float: The calculated score for the endpoint.
    """

    # VALUE CALCULATION
    gauss_value_at_endpoint = get_gauss_value(endpoints[direction], volume_chunk, volume_chunk_bounds)
    endpoint_value = (volume[volume_index(endpoints[direction])] + gauss_value_at_endpoint) / 2  # with gauss to denoise
    sigmoid_endpoint_value = S_VALUE_SIGMOID_COEFF_1 if endpoint_value < OTSU_THRESHOLD else S_VALUE_SIGMOID_COEFF_2

    # CONTRAST CALCULATION
    # Calculate the point before the current one
    previous_point = endpoints[direction] - radial_direction_vectors[direction] * HORIZONTAL_RESOLUTION
    gauss_value_at_previous_point = get_gauss_value(previous_point, volume_chunk, volume_chunk_bounds)
    # Calculate the contrast as the difference in values between the current endpoint and the previous point
    contrast = gauss_value_at_endpoint - gauss_value_at_previous_point
    sigmoid_contrast = contrast_adjusted_sigmoid(contrast, score_contrast_a, score_contrast_b)

    # PROXIMITY CALCULATION
    proximity = np.linalg.norm(endpoints_predicted[direction] - endpoints[direction])
    sigmoid_proximity = sigmoid(proximity * S_PROXIMITY_SIGMOID_COEFF * MAX_EXPECTED_RADIUS + S_PROXIMITY_SIGMOID_OFFSET)

    # FINAL SCORE CALCULATION
    score = sigmoid(
        S_VALUE_WEIGHT * sigmoid_endpoint_value
        + S_CONTRAST_WEIGHT * sigmoid_contrast
        + S_PROXIMITY_WEIGHT * sigmoid_proximity
    )

    s_scores = {  # todo temp
        'value': sigmoid_endpoint_value,
        'contrast': sigmoid_contrast,
        'proximity': sigmoid_proximity
    }

    return score, s_scores


def calculate_next_layer_prediction(relative_progress, tooth_centers, tooth_center_direction_vector_ema,
                                    surface_normal_vector, tooth_radii_per_side, radii_diff_emas_per_side,
                                    radii_emas_per_side, expected_radius):
    """
    Calculate the predicted values for the next layer in tooth scan.

    Parameters:
    relative_progress (float): The relative progress of tooth scan.
    tooth_centers (np.array): Centers of the teeth.
    tooth_direction_ema (np.array): Direction of the tooth using EMA (Exponential Moving Average).
    surface_normal_vector (np.array): Surface normal vector.
    tooth_radii_per_side (np.array): Radii of the teeth per side.
    radii_diff_emas_per_side (np.array): EMA of the difference in tooth radii per side.
    radii_emas_per_side (np.array): EMA of the tooth radii per side.
    expected_radius (float): Expected radius of the tooth.

    Returns:
    np.array, np.array, np.array: Updated values for tooth direction EMA, radii difference EMA per side, and radii EMA per side.
    """

    # Compute factors
    new_val_factor \
        = -math.sqrt(relative_progress) * (P_NEW_VAL_FACTOR_START - P_NEW_VAL_FACTOR_END) + P_NEW_VAL_FACTOR_START
    normal_factor = math.sin(relative_progress * math.pi / 2) * P_NORMAL_FACTOR_MULTIPLIER
    old_val_factor = 1 - (new_val_factor + normal_factor)

    # Update tooth direction EMA
    tooth_center_direction_vector_ema \
        = new_val_factor * (tooth_centers[-1] - tooth_centers[-2]) \
        + old_val_factor * tooth_center_direction_vector_ema \
        + normal_factor * surface_normal_vector * VERTICAL_RESOLUTION

    # Update radii difference EMA per side
    radii_diff_emas_per_side = \
        P_EMA_DIFF_ADJUST_FACTOR * (tooth_radii_per_side[-1] - tooth_radii_per_side[-2]) + \
        (1 - P_EMA_DIFF_ADJUST_FACTOR) * radii_diff_emas_per_side

    # Update radii EMA per side
    radii_emas_per_side \
        = P_CURRENT_TOOTH_RADIUS_WEIGHT * tooth_radii_per_side[-1] \
        + P_PREVIOUS_EMA_RADIUS_WEIGHT * radii_emas_per_side \
        + P_MEDIAN_TOOTH_RADIUS_WEIGHT * np.median(tooth_radii_per_side[-1]) \
        + P_EXPECTED_RADIUS_WEIGHT * expected_radius

    return tooth_center_direction_vector_ema, radii_diff_emas_per_side, radii_emas_per_side


def scan_for_endpoints(volume, smoothed_volume_chunk, volume_chunk_bounds, radial_direction_vectors, endpoints,
                       endpoints_predicted, score_contrast_a, score_contrast_b):
    """
    Scans for the actual endpoints of a tooth layer in all directions.

    Parameters:
    volume (np.array): 3D array representing the volume in which the tooth is embedded.
    smoothed_volume_chunk (np.array): 3D array representing the smoothed chunk of the volume around the tooth.
    volume_chunk_bounds (np.array): Bounds of the volume chunk.
    radial_direction_vectors (np.array): Radial direction vectors around the tooth tip.
    endpoints (np.array): Initial endpoints for the layer.
    endpoints_predicted (np.array): Predicted endpoints for the layer.

    Returns:
    np.array: Updated endpoints.
    """
    # Loop over each direction
    for direction in range(N_DIRECTIONS):
        i_horizontal = 0
        perfect_fit_found = False
        #score = 0  # todo temp

        # Scan until we find a perfect fit or reach the maximum expected radius
        while i_horizontal < MAX_EXPECTED_RADIUS and not perfect_fit_found:
            # Check if the current endpoint is within the volume bounds
            if is_point_within_bounds(endpoints[direction], volume.shape):
                # If we are at the maximum expected radius, use the predicted endpoint
                if i_horizontal == MAX_EXPECTED_RADIUS - 1:
                    # endpoints[direction] = endpoints_predicted[direction]
                    endpoints[direction] = endpoints_predicted[direction]
                    perfect_fit_found = True
                else:
                    # Calculate the score for the current endpoint
                    score, s_scores = calculate_score_for_endpoint(
                        endpoints, direction, volume, smoothed_volume_chunk, volume_chunk_bounds, radial_direction_vectors,
                        endpoints_predicted, score_contrast_a, score_contrast_b)

                    # If the score is high enough, we have found a perfect fit for the endpoint
                    perfect_fit_found = score >= PERFECT_FIT_MINIMUM_SCORE

                    # If not, move the endpoint further in the current direction
                    if not perfect_fit_found:
                        endpoints[direction] += radial_direction_vectors[direction] * HORIZONTAL_RESOLUTION
            i_horizontal += 1
    return endpoints


def endpoint_correction(endpoints, endpoints_predicted, tooth_centers, radial_direction_vectors,
                        expected_radius, corrections, relative_tooth_scan_progress, i_layer,
                        radii_emas_per_side, radii_diff_emas_per_side):
    """
    Corrects the endpoints of a layer if they are too far from the predicted endpoints or from the expected radius.

    Parameters:
    endpoints (np.array): Current endpoints of the layer.
    endpoints_predicted (np.array): Predicted endpoints of the layer.
    tooth_centers (list): List of tooth center points for each layer.
    radial_direction_vectors (np.array): Radial direction vectors.
    expected_radius (float): Expected radius for this layer.
    corrections (dict): Dictionary keeping track of the number of corrections made.
    relative_tooth_scan_progress (float): Relative progress of the tooth scan.
    i_layer (int): Current layer index.
    radii_emas_per_side (np.array): Exponential moving averages of radii for each side.
    radii_diff_emas_per_side (np.array): Differences of exponential moving averages of radii for each side.

    Returns:
    np.array, dict: The corrected endpoints and the updated corrections dictionary.
    """

    # 1. Check each endpoint for proximity to predicted endpoint. If too far off, shift towards predicted endpoint.
    if True:
        for direction in range(endpoints.shape[0]):
            proximity_to_predicted = np.linalg.norm(endpoints_predicted[direction] - endpoints[direction])
            proximity_to_center = np.linalg.norm(endpoints_predicted[direction] - tooth_centers[-1])
            radius_layer = np.mean(radii_emas_per_side + radii_diff_emas_per_side)

            relative_offset = proximity_to_predicted / proximity_to_center \
                + abs(radius_layer - proximity_to_center) / proximity_to_center
            corrections['chances_to_correct_proximity'] += 1

            if relative_offset >= C_OFFSET_THRESHOLD:
                correction_factor = (1 + relative_tooth_scan_progress) / 2 * C_SMOOTHING_FACTOR_BASE
                endpoints[direction] = correction_factor * (
                        C_ENDPOINT_PREDICTION_WEIGHT * endpoints_predicted[direction]
                        + C_EXPECTED_RADIUS_WEIGHT
                        * (tooth_centers[-1] + expected_radius * radial_direction_vectors[direction])
                ) + (1 - correction_factor) * endpoints[direction]
                corrections['proximity_to_predicted'] += 1

    # 2. Smooth the endpoints using the predicted endpoints and the expected radius
    for direction in range(endpoints.shape[0]):
        correction_factor = (1 + relative_tooth_scan_progress) / 2 * C_SMOOTHING_FACTOR_BASE
        endpoints[direction] = correction_factor * (
                C_ENDPOINT_PREDICTION_WEIGHT * endpoints_predicted[direction]
                + C_EXPECTED_RADIUS_WEIGHT
                * (tooth_centers[-1] + expected_radius * radial_direction_vectors[direction])
        ) + (1 - correction_factor) * endpoints[direction]

    # Append the center point for this layer
    center_point = np.mean(endpoints, axis=0)

    # 3. Check if endpoints are too far off from the median radius of the layer or the expected radius.
    radii_per_direction = np.linalg.norm(endpoints - center_point, axis=1)
    layer_radius_median = np.median(radii_per_direction)
    for direction, radius in enumerate(radii_per_direction):
        if radius > layer_radius_median * C_LAYER_RADIUS_MEDIAN_MULTIPLIER \
            or radius < layer_radius_median / C_LAYER_RADIUS_MEDIAN_MULTIPLIER \
            or radius > expected_radius * C_EXPECTED_RADIUS_MULTIPLIER \
            or radius < expected_radius / C_EXPECTED_RADIUS_MULTIPLIER:
            correction_factor = (C_CORRECTION_FACTOR_AT_END - C_CORRECTION_FACTOR_AT_START) \
                                * relative_tooth_scan_progress + C_CORRECTION_FACTOR_AT_START
            endpoints[direction] = correction_factor * (
                    C_ENDPOINT_PREDICTION_WEIGHT * endpoints_predicted[direction]
                    + C_EXPECTED_RADIUS_WEIGHT * (center_point + expected_radius * radial_direction_vectors[direction])
                ) + (1 - correction_factor) * endpoints[direction]
            corrections['too far off'] += 1

    # Append the center point of the layer as the endpoints were corrected
    tooth_centers.append(np.mean(endpoints, axis=0))
    for direction in range(N_DIRECTIONS):
        radii_per_direction[direction] = np.linalg.norm(endpoints[direction] - tooth_centers[-1])  # with fixed center

    return endpoints, corrections, radii_per_direction, tooth_centers


def scan_tooth(tooth_tip: np.ndarray, base_surface_mesh: vtk.vtkPolyData, volume, score_contrast_a, score_contrast_b, point_locator, start, whole_volume_length):
    """
    Scans a tooth in a given volume starting from the tooth tip and moving towards the base surface.

    Parameters:
    tooth_tip (np.ndarray): Coordinates (x, y, z) of the tooth tip.
    base_surface_mesh (vtk.vtkPolyData): Mesh representing the base surface.
    volume (np.array): 3D array representing the volume in which the tooth is embedded.
    score_contrast_a (float): Contrast score a.
    score_contrast_b (float): Contrast score b.
    point_locator (vtk.vtkPointLocator): Point locator for the base surface mesh.
    start (float): Start position of the volume in the whole scan.
    whole_volume_length (float): Length of the whole volume.

    Returns:
    dict: A dictionary containing information about the scanned tooth.
    """
    tooth_centers = []
    tooth_endpoints = []
    tooth_radii_per_direction = []

    # Find the closest point on the base surface to the tooth tip
    closest_base_surface_point = get_closest_point_on_polydata_to_given_point(tooth_tip, base_surface_mesh, point_locator)

    # Get the volume chunk around the tooth
    smoothed_volume_chunk, volume_chunk_bounds = volume_chunk_around_tooth(volume, tooth_tip, closest_base_surface_point)
    smoothed_volume_chunk = gaussian_filter(smoothed_volume_chunk, sigma=GAUSSIAN_FILTER_SIGMA)

    # Compute the surface normal vector at the tooth tip
    surface_normal_vector = np.array(closest_base_surface_point - tooth_tip)
    max_scan_len = np.linalg.norm(surface_normal_vector * 2)  # twice the distance from tooth tip to base surface
    surface_normal_vector /= np.linalg.norm(surface_normal_vector)

    # Compute the radial direction vectors around the tooth tip
    radial_direction_vectors = calculate_radial_direction_vectors(surface_normal_vector, N_DIRECTIONS)
    inwards_shift_vectors = np.multiply(radial_direction_vectors, INWARDS_SHIFT_MULTIPLIER)

    # Compute the expected tooth height
    expected_tooth_height = np.linalg.norm(tooth_tip - closest_base_surface_point)

    # Initialize tooth parameters
    current_position = np.array(tooth_tip, dtype=float)
    # axis_center_point = np.array([AXIS_CENTER_X, AXIS_CENTER_Y, closest_base_surface_point[2]])
    axis_center_point_start = np.array([volume.shape[2] / 2, volume.shape[1] / 2])
    axis_center_point_end = np.array([AXIS_CENTER_X, AXIS_CENTER_Y])
    z_relative_progress_in_whole_volume = (start + closest_base_surface_point[2]) / whole_volume_length
    axis_center_point = z_relative_progress_in_whole_volume ** 2 \
        * (axis_center_point_end - axis_center_point_start) + axis_center_point_start
    axis_center_point = np.array([axis_center_point[0], axis_center_point[1], tooth_tip[2]])
    distance_to_axis_center = np.linalg.norm(axis_center_point - closest_base_surface_point)

    tooth_centers.append(current_position)
    endpoints = np.array([current_position] * N_DIRECTIONS)
    tooth_endpoints.append(endpoints)
    tooth_radii_per_direction.append(np.zeros(N_DIRECTIONS))
    tooth_center_direction_vector_ema = surface_normal_vector * VERTICAL_RESOLUTION

    radii_diff_emas_per_side = np.full(N_DIRECTIONS, 1)
    radii_emas_per_side = np.full(N_DIRECTIONS, 1)
    current_position = tooth_centers[0] + tooth_center_direction_vector_ema

    base_surface_reached = False
    tooth_goes_out_of_bounds = False
    tooth_endpoints_go_out_of_bounds = False
    max_scan_len_reached = False
    i_layer = 1
    corrections = {'proximity_to_predicted': 0, 'chances_to_correct_proximity': 0, 'too far off': 0}  # todo remove??!!

    # Start scanning the tooth
    while not (base_surface_reached or tooth_goes_out_of_bounds or max_scan_len_reached):
        # Calculate the relative progress of the tooth scan
        relative_tooth_scan_progress = np.linalg.norm(tooth_tip - current_position) / expected_tooth_height
        expected_radius = math.sqrt(relative_tooth_scan_progress) * RADIUS_MULTIPLIER * expected_tooth_height

        # Initialize endpoints for this layer
        endpoints = np.array([current_position] * N_DIRECTIONS) \
            + (tooth_endpoints[-1] - np.array([tooth_centers[-1]] * N_DIRECTIONS)) \
            * ENDPOINTS_INIT_FACTOR * math.sqrt(relative_tooth_scan_progress) \
            - inwards_shift_vectors

        # Predicted endpoints for this layer
        endpoints_predicted = np.array([current_position] * N_DIRECTIONS) \
            + radial_direction_vectors * (radii_emas_per_side + radii_diff_emas_per_side)[:, None]

        # Scan to find the actual endpoints for this layer
        endpoints = scan_for_endpoints(volume, smoothed_volume_chunk, volume_chunk_bounds, radial_direction_vectors,
                                                    endpoints, endpoints_predicted, score_contrast_a, score_contrast_b)

        # Correct endpoints if they are too far from the predicted endpoints or from the expected radius
        endpoints, corrections, radii_per_direction, tooth_centers \
            = endpoint_correction(endpoints, endpoints_predicted, tooth_centers, radial_direction_vectors,
                                  expected_radius, corrections, relative_tooth_scan_progress, i_layer,
                                  radii_emas_per_side, radii_diff_emas_per_side)

        # Append the endpoints and radii for this layer
        tooth_endpoints.append(endpoints)
        tooth_radii_per_direction.append(radii_per_direction)

        # Check if the tooth goes out of bounds
        if not is_point_within_bounds(tooth_centers[-1], volume.shape):
            tooth_goes_out_of_bounds = True
        for direction in range(N_DIRECTIONS):
            if not is_point_within_bounds(tooth_endpoints[-1][direction], volume.shape):
                tooth_endpoints_go_out_of_bounds = True

        # Check if the max scan length has been reached
        if np.linalg.norm(tooth_tip - tooth_centers[-1]) > max_scan_len:
            max_scan_len_reached = True
            # correct possible errors
            if np.linalg.norm(tooth_centers[-1] - closest_base_surface_point) > np.linalg.norm(tooth_tip - closest_base_surface_point):
                for direction in range(N_DIRECTIONS):
                    tooth_endpoints[-1][direction] = closest_base_surface_point
                    endpoints[direction] = closest_base_surface_point
                    tooth_centers[-1] = closest_base_surface_point

        # Update the direction vector and radii for the next layer
        tooth_center_direction_vector_ema, radii_diff_emas_per_side, radii_emas_per_side = calculate_next_layer_prediction(
            relative_tooth_scan_progress, tooth_centers, tooth_center_direction_vector_ema, surface_normal_vector,
            tooth_radii_per_direction, radii_diff_emas_per_side, radii_emas_per_side, expected_radius)

        # Update the current position for the next layer
        current_position = tooth_centers[i_layer] + tooth_center_direction_vector_ema

        # Check if the base surface has been reached
        base_surface_reached |= distance_to_axis_center > np.linalg.norm(axis_center_point - tooth_centers[i_layer])
        i_layer += 1

    # Check if all center points are in volume bounds  # todo improve
    if not tooth_goes_out_of_bounds:
        average_value_of_deeper_tooth_centers = average_value_in_volume_at_coords(volume, tooth_centers[-len(tooth_centers) // 3:])
    else:
        average_value_of_deeper_tooth_centers = 0.0

    deepest_center_point_base_surface = get_closest_point_on_polydata_to_given_point(
        tooth_centers[-1], base_surface_mesh, point_locator)

    # Return a dictionary containing information about the scanned tooth
    return {
        'tooth_is_valid': not(tooth_goes_out_of_bounds or tooth_endpoints_go_out_of_bounds)
                          and len(tooth_centers) > 3
                          and np.linalg.norm(tooth_centers[0] - tooth_centers[-1]) > expected_tooth_height / 2
                          and 2 * expected_tooth_height > np.mean(np.array(tooth_radii_per_direction[-1])) > 0,
        'corrections': corrections,
        'tooth_endpoints': tooth_endpoints,
        'tooth_centers': tooth_centers,
        'tooth_radii': np.array(tooth_radii_per_direction),
        'average_value_of_deeper_tooth_centers': average_value_of_deeper_tooth_centers,
        'tooth_tip': tooth_tip,
        'closest_point_on_base_surface': closest_base_surface_point,
        'deepest_center_point_base_surface': deepest_center_point_base_surface,
    }


def run_teeth_scan():
    """
    This function runs the teeth scan over a given volume. It loads the volume data from a hdf5 file,
    reads the base surface from a PLY file, and then iterates over a CSV file of labels to perform the
    tooth scan for each labeled tooth. The scanned tooth data is then saved to a pickle file.
    """

    # Load volume data and metadata from hdf5 file
    with h5py.File(helper.get_path_of_existing_file_from_config('volume', CONFIG), "r") as f:
        volume_dset = f["volume"]
        meta_grp = f["meta"]
        length = int(meta_grp.attrs["length"])
        data_type = meta_grp.attrs["data type"]
        voxel_length_in_micrometers = meta_grp.attrs["resolution"]
        start = 0 if USE_FULL_VOLUME_LENGTH \
            else CUSTOM_VOLUME_START if not ROSTRUM_IS_POINTED_TO_THE_FRONT \
            else length - CUSTOM_VOLUME_START - CUSTOM_VOLUME_LENGTH
        stop = length if USE_FULL_VOLUME_LENGTH \
            else (CUSTOM_VOLUME_LENGTH + start) if not ROSTRUM_IS_POINTED_TO_THE_FRONT \
            else length - CUSTOM_VOLUME_START
        volume = volume_dset[start:stop]

    # Load the base surface from a PLY file
    ply_reader = vtk.vtkPLYReader()
    ply_reader.SetFileName(helper.get_path_of_existing_file_from_config('base_surface', CONFIG))
    ply_reader.Update()
    base_surface_polydata = ply_reader.GetOutput()

    # Prepare the score calculation
    # Adjust median and quartile_75 using the coefficients and offsets
    median = S_CONTRAST_SIGMOID_COEFF * PERCENTILE_50 + S_CONTRAST_SIGMOID_OFFSET
    quartile_75 = S_CONTRAST_SIGMOID_COEFF * PERCENTILE_75 + S_CONTRAST_SIGMOID_OFFSET
    # Sigmoid inverse of 0.7 and 0.95 (used for calculating the coefficients and offsets for the adjusted sigmoid function)
    x_at_05 = np.log(0.5 / (1 - 0.5))
    x_at_075 = np.log(0.75 / (1 - 0.75))
    score_contrast_a = (x_at_075 - x_at_05) / (quartile_75 - median)
    score_contrast_b = x_at_05 - score_contrast_a * median
    logging.info(f"contrast sigmoid function: c(x) = 1 / (1 + exp(-({score_contrast_a} * x + {score_contrast_b})))")
    logging.info(f"value sigmoid function: v(x) = 1 / (1 + exp(-({S_VALUE_SIGMOID_COEFF_1} * ((x - {OTSU_THRESHOLD} + ({OTSU_THRESHOLD} - {PERCENTILE_25}) * {S_VALUE_SIGMOID_COEFF_2}) / ({OTSU_THRESHOLD} - {PERCENTILE_25})) ^ 2 + 1.5)))")
    logging.info(f"proximity sigmoid function: p(x) = 1 / (1 + exp(-(x * {S_PROXIMITY_SIGMOID_COEFF} * {MAX_EXPECTED_RADIUS} + {S_PROXIMITY_SIGMOID_OFFSET})))")

    # Find the point ID of the point to search for
    point_locator = vtk.vtkPointLocator()
    point_locator.SetDataSet(base_surface_polydata)
    point_locator.BuildLocator()

    # Open the CSV file containing the labels
    with open(helper.get_path_of_existing_file_from_config('labels', CONFIG)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # skip header line of csv
        tooth_dicts = []

        # Iterate over each row in the CSV file
        for i_tooth, row in enumerate(csv_reader):
            z_coordinate = int(row[2])

            # Check if the z_coordinate is within the volume range
            if start <= z_coordinate < stop:
                tooth_tip = np.array([int(row[0]), int(row[1]), z_coordinate - start])
                print(f"Scanning tooth with index {i_tooth}")
                # Perform the tooth scan and append the result to tooth_dicts
                tooth_scan = scan_tooth(tooth_tip, base_surface_polydata, volume, score_contrast_a, score_contrast_b, point_locator, start, length)
                tooth_dicts.append(tooth_scan)

    # Save the tooth data to a pickle file
    tooth_data = {
        'tooth_dicts': tooth_dicts,
        'n_directions': N_DIRECTIONS,
        'start': start,
        'stop': stop,
        'voxel_length_in_micrometers': voxel_length_in_micrometers,
    }

    with open(helper.create_path_for_new_file_from_config('full_teeth_scan', CONFIG), 'wb') as f:
        pickle.dump(tooth_data, f)


if __name__ == '__main__':
    tic = time.perf_counter()
    script_name = __file__.split('\\')[-1]
    logging.info(f'started {script_name}...\n')
    run_teeth_scan()
    logging.info(f"finished. {time.perf_counter() - tic:1.1f}s")
    exit(0)

