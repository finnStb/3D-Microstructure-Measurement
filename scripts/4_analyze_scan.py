import configparser
import logging
import math
import pickle
import time

import numpy as np
import pandas as pd
import scipy
import vtk
from vtkmodules.util import numpy_support

from utils import helper
from config_and_logging import CONFIG, DEBUG_MODE, USE_FULL_VOLUME_LENGTH, CUSTOM_VOLUME_LENGTH, CUSTOM_VOLUME_START

# Read constants from the config file
KD_TREE_RADIUS_FOR_HEATMAP = CONFIG.getfloat('Analysis', 'kd_tree_radius_for_heatmap')


def smooth_3d_vectors(vectors: np.ndarray, n_smooth: int = 1):
    """
    Smooth 3D vectors by averaging each vector with its neighbors.
    Parameters:
    vectors (np.ndarray): The input 3D vectors to be smoothed.
    n_smooth (int): The number of smoothing iterations to perform.
    Returns:
    np.ndarray: The smoothed 3D vectors.
    """
    # If no smoothing iterations are required, return the original vectors
    if n_smooth <= 0:
        return vectors

    # Initialize an array for the smoothed vectors
    smoothed_vectors = np.zeros_like(vectors)

    # Iterate over each vector in the input array
    for i_vector in range(vectors.shape[0]):
        n_neighbors = 0  # Initialize neighbor count

        # Calculate the start and end indices for the neighbors
        start_idx = max(0, i_vector - 1)
        end_idx = min(vectors.shape[0], i_vector + 2)

        # Iterate over each neighbor
        for i_neighbor in range(start_idx, end_idx):
            # Add the vector of the neighbor
            smoothed_vectors[i_vector] += vectors[i_neighbor]
            n_neighbors += 1  # Increment neighbor count

        # Average the vector by dividing by the number of neighbors
        smoothed_vectors[i_vector] /= n_neighbors

    # Perform the smoothing operation recursively for n_smooth - 1 times
    return smooth_3d_vectors(smoothed_vectors, n_smooth - 1)


def smooth_3d_points(points: np.ndarray | list, iterations: int = 2):
    """
    Smooth 3D points by calculating the vectors between points and then smoothing these vectors.

    Parameters:
    points (np.ndarray | list): The input 3D points to be smoothed.
    iterations (int): The number of smoothing iterations to perform.

    Returns:
    tuple: A tuple containing the smoothed 3D points and the smoothed vectors between the points.
    """
    # Calculate the vectors between the points
    vectors_between_points = np.diff(points, axis=0)

    # Smooth the vectors between the points
    smoothed_vectors = smooth_3d_vectors(vectors_between_points, iterations)

    # Initialize an array for the smoothed points
    smoothed_points = np.zeros_like(points)

    # The first point remains the same
    smoothed_points[0] = points[0]

    # Iterate over each smoothed vector
    for i_vector, vector in enumerate(smoothed_vectors):
        # The next smoothed point is the current smoothed point plus the current vector
        smoothed_points[i_vector + 1] = smoothed_points[i_vector] + vector

    return smoothed_points, smoothed_vectors


def calculate_smoothed_centers_upwards(tooth_endpoints: np.ndarray, n_directions: int, smooth_factor: float = 0.5):
    """
    Calculate the smoothed centers of the tooth endpoints in the upward direction.

    Parameters:
    tooth_endpoints (np.ndarray): The endpoints of the tooth to be smoothed.
    n_directions (int): The number of directions to consider for smoothing.
    smooth_factor (float): The factor for smoothing the endpoints. Defaults to 0.5.

    Returns:
    list: The smoothed centers of the tooth endpoints.
    """
    smoothed_centers_upwards = []

    # Swap the axes of the tooth_endpoints to make it easier to work with sides of the tooth separately
    tooth_endpoints = np.array(tooth_endpoints).swapaxes(0, 1)

    endpoints_smoothed = np.zeros_like(tooth_endpoints)

    # Smooth the endpoints for each side
    for i_side, endpoints_side in enumerate(tooth_endpoints):
        endpoints_side = np.flip(endpoints_side)
        endpoints_smoothed[i_side][0] = endpoints_side[0]
        vectors_between_points = np.diff(endpoints_side, axis=0)
        ema_vector = vectors_between_points[0] if len(vectors_between_points) > 0 else np.zeros(3)

        # Apply exponential moving average (EMA) smoothing
        for i_vector in range(len(vectors_between_points)):
            ema_vector = (1 - smooth_factor) * ema_vector + smooth_factor * vectors_between_points[i_vector]
            endpoints_smoothed[i_side][i_vector + 1] = endpoints_smoothed[i_side][i_vector] + ema_vector

        endpoints_smoothed[i_side] = np.flip(endpoints_smoothed[i_side])

    # Swap the axes back to the original order after smoothing the endpoints for each side
    endpoints_smoothed = endpoints_smoothed.swapaxes(0, 1)

    # Calculate smoothed centers by averaging the endpoints
    for endpoints in endpoints_smoothed:
        smoothed_centers_upwards.append(endpoints.sum(axis=0) / n_directions)

    return smoothed_centers_upwards


def analyse_scan(tooth_dicts: list[dict], n_directions: int, kd_tree: scipy.spatial.KDTree, voxel_length_in_micrometers):
    """
    Analyse a scan of teeth, calculating various properties of each tooth such as height, length,
    radii, angles, curvature, and density.

    Parameters:
    tooth_dicts (list[dict]): A list of dictionaries, where each dictionary contains data for a tooth.
    n_directions (int): The number of directions to consider for smoothing.
    kd_tree (scipy.spatial.KDTree): A KDTree for efficient nearest neighbor queries, used for density calculation.

    Returns:
    list[dict]: A list of dictionaries, where each dictionary contains analysed data for a tooth.
    """
    analyses = []

    # Iterate over each tooth
    for i_tooth, tooth_data in enumerate(tooth_dicts):
        # Calculate the smoothed centers of the tooth in the upward direction
        tooth_centers = calculate_smoothed_centers_upwards(tooth_data['tooth_endpoints'], n_directions, 0.5)

        # Calculate the tooth height from the base surface root to the tip
        tooth_height = np.linalg.norm(tooth_data['tooth_tip'] - tooth_data['closest_point_on_base_surface'])

        # Calculate the vectors used for various angle measurements
        normal_vector_under_tooth_tip = tooth_data['tooth_tip'] - tooth_data['closest_point_on_base_surface']
        tooth_vector = tooth_data['tooth_tip'] - tooth_data['deepest_center_point_base_surface']
        angle_around_tooth_root_based_on_rostrum_axis = helper.angle_between_vectors_circle(normal_vector_under_tooth_tip - tooth_vector, np.array([0, 0, 1]))

        # Calculate the relative angle of the tooth from the base to the tip
        tooth_angle_base_to_tip_rel = helper.angle_between_vectors(normal_vector_under_tooth_tip, tooth_vector)

        # Calculate the simplified curvature of the tooth if there are enough tooth centers
        simplified_curvature = 0.0
        if len(tooth_centers) >= 3:
            lower_half_vector = tooth_centers[len(tooth_centers) // 2] - tooth_centers[0]
            upper_half_vector = tooth_centers[-1] - tooth_centers[len(tooth_centers) // 2]
            simplified_curvature = helper.angle_between_vectors(lower_half_vector, upper_half_vector)

        # Calculate the mean radii per layer and per side
        radii_mean_per_layer = np.mean(tooth_data['tooth_radii'], axis=1)
        radii_mean_per_side = np.mean(tooth_data['tooth_radii'], axis=0)

        # Smooth the tooth centers and calculate the vectors between them
        smoothed_tooth_centers, smoothed_vectors_between_tooth_centers = smooth_3d_points(tooth_centers, 12)

        # Calculate the summed curvature and the tooth length along the route of the centers
        summed_curvature = 0.0
        if len(tooth_centers) >= 3:
            for i_vector in range(len(smoothed_vectors_between_tooth_centers) - 1):
                summed_curvature += helper.angle_between_vectors(smoothed_vectors_between_tooth_centers[i_vector],
                                                                   smoothed_vectors_between_tooth_centers[i_vector + 1])
        tooth_length_route = 0.0
        for i in range(len(smoothed_tooth_centers) - 1):
            tooth_length_route += np.linalg.norm(smoothed_tooth_centers[i] - smoothed_tooth_centers[i + 1])

        voxel_length_to_millimeters_factor = voxel_length_in_micrometers / 1000

        # Compile the analysis for this tooth into a dictionary
        analysis_of_one_tooth = {
            'height[mm]': tooth_height * voxel_length_to_millimeters_factor,
            'straight_line_length[mm]': np.linalg.norm(tooth_vector) * voxel_length_to_millimeters_factor,
            'route_length[mm]': tooth_length_route * voxel_length_to_millimeters_factor,
            'mean_radius_per_layer[mm]': radii_mean_per_layer * voxel_length_to_millimeters_factor,
            'mean_radius_per_side[mm]': radii_mean_per_side * voxel_length_to_millimeters_factor,
            'mean_radius[mm]': np.mean(tooth_data['tooth_radii'].ravel()) * voxel_length_to_millimeters_factor,
            'median_radius[mm]': np.median(tooth_data['tooth_radii'].ravel()) * voxel_length_to_millimeters_factor,
            'radius_std_deviation[mm]': np.std(tooth_data['tooth_radii'].ravel()) * voxel_length_to_millimeters_factor,
            'radius_std_deviation_per_side[mm]': np.std(radii_mean_per_side.ravel()) * voxel_length_to_millimeters_factor,
            'radius_std_deviation_per_layer[mm]': np.std(radii_mean_per_layer.ravel()) * voxel_length_to_millimeters_factor,
            'angle_around_tooth_root_based_on_rostrum_axis': angle_around_tooth_root_based_on_rostrum_axis,
            'root_to_tip_angle_x': helper.angle_between_vectors_circle(tooth_vector, np.array([1, 0, 0])),
            'root_to_tip_angle_y': helper.angle_between_vectors_circle(tooth_vector, np.array([0, 1, 0])),
            'root_to_tip_angle_z': helper.angle_between_vectors_circle(tooth_vector, np.array([0, 0, 1])),
            'base_to_tip_angle': tooth_angle_base_to_tip_rel,
            'simplified_curvature': simplified_curvature,
            'summed_curvature': summed_curvature,
            'id': i_tooth,
            'base_x_coordinate': tooth_data['deepest_center_point_base_surface'][0],
            'base_y_coordinate': tooth_data['deepest_center_point_base_surface'][1],
            'base_z_coordinate': tooth_data['deepest_center_point_base_surface'][2],
            'vector_under_tooth_tip': normal_vector_under_tooth_tip,
            'tooth_vector': tooth_vector,
            'tooth_is_valid': tooth_data['tooth_is_valid'],
        }

        # Add the analysis to the list of analyses
        analyses.append(analysis_of_one_tooth)

    return analyses


def build_polydata_from_tooth_dicts(tooth_dicts: list[dict], n_directions: int):
    """
    Build a VTK PolyData structure from tooth data dictionaries.

    Parameters:
    tooth_dicts (list[dict]): A list of dictionaries, where each dictionary contains data for a tooth.
    n_directions (int): The number of directions to consider for smoothing.

    Returns:
    tuple: A tuple containing the combined polydata of all teeth and a list of polydata for each individual tooth.
    """
    tooth_polydata_list = []
    append_filter = vtk.vtkAppendPolyData()

    # Iterate over each tooth
    for i_tooth, tooth_data in enumerate(tooth_dicts):
        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        tooth_endpoints: [[]] = tooth_data.get("tooth_endpoints")

        n_layers = len(tooth_endpoints)

        # Iterate over each layer and direction, adding points and creating triangles
        for i_layer in range(n_layers):
            for i_direction in range(n_directions):
                points.InsertNextPoint(tooth_endpoints[i_layer][i_direction])

                # Create triangles if not on the last layer and not on the last direction
                if i_layer < n_layers - 1 and i_direction < n_directions:
                    polys.InsertNextCell(3)  # triangle has 3 points
                    polys.InsertCellPoint(i_layer * n_directions + i_direction)
                    polys.InsertCellPoint(i_layer * n_directions + (i_direction + 1) % n_directions)
                    polys.InsertCellPoint((i_layer + 1) * n_directions + i_direction)

                    polys.InsertNextCell(3)  # triangle has 3 points
                    polys.InsertCellPoint(i_layer * n_directions + (i_direction + 1) % n_directions)
                    polys.InsertCellPoint((i_layer + 1) * n_directions + (i_direction + 1) % n_directions)
                    polys.InsertCellPoint((i_layer + 1) * n_directions + i_direction)

        # Create a polydata object for the tooth and add it to the list
        tooth_polydata = vtk.vtkPolyData()
        tooth_polydata.SetPoints(points)
        tooth_polydata.SetPolys(polys)
        tooth_polydata_list.append(tooth_polydata)

        # Add the tooth polydata to the append filter
        append_filter.AddInputData(tooth_polydata)

    append_filter.Update()

    # Convert the output of the append filter to a polydata object
    output_polydata = vtk.vtkPolyData()
    output_polydata.ShallowCopy(append_filter.GetOutput())

    return output_polydata, tooth_polydata_list


def compare_tooth_analyses(analyses: list[dict]):
    """
    Compare analyses of teeth by calculating the fraction of the maximum value of each statistic.

    Parameters:
    analyses (list[dict]): A list of dictionaries, where each dictionary contains analyzed data for a tooth.

    Returns:
    tuple: A tuple containing the list of comparisons for each tooth, the list of comparison combinations
           for each tooth, and a dictionary of minimum and maximum values for each statistic.
    """

    # Identify the numeric keys in the analysis
    numeric_keys = [key for key in analyses[0] if isinstance(analyses[0][key], (int, float, np.float64, np.float32))]

    # Set all numeric values in analyses dicts to nan for all teeth that are not valid
    for i_tooth, tooth_analysis in enumerate(analyses):
        if not tooth_analysis['tooth_is_valid']:
            for key in numeric_keys:
                tooth_analysis[key] = np.nan
            analyses[i_tooth] = tooth_analysis

    num_teeth = len(analyses)
    tooth_comparisons = [{} for _ in range(num_teeth)]
    comparison_combinations = [{} for _ in range(num_teeth)]

    min_values = {}
    max_values = {}

    # Find the minimum and maximum values for each numeric key
    for key in numeric_keys:
        values = [tooth_analysis[key] for tooth_analysis in analyses]
        min_values[key] = min(values)
        max_values[key] = max(values)

        for i_tooth, tooth_analysis in enumerate(analyses):
            # Normalize the value between min_value and max_value
            normalized_value = np.true_divide(tooth_analysis[key] - min_values[key], max_values[key] - min_values[key])
            tooth_comparisons[i_tooth][key] = normalized_value

    print(f"min_values: {min_values}")
    print(f"max_values: {max_values}")

    # Calculate the combinations of all numeric keys
    for key_dividend in numeric_keys:
        for key_divisor in numeric_keys:
            key_combined = f"{key_dividend}_divided_by_{key_divisor}"

            if key_dividend == key_divisor:
                min_values[key_combined] = max_values[key_combined] = 1
                for i_tooth in range(num_teeth):
                    comparison_combinations[i_tooth][key_combined] = 1
                continue

            values = [tooth_analysis[key_dividend] / tooth_analysis[key_divisor]
                      if tooth_analysis[key_divisor] != 0 else None for tooth_analysis in analyses]
            values = [value for value in values if value is not None]  # Remove None values
            if values:
                min_values[key_combined] = min(values)
                max_values[key_combined] = max(values)

                for i_tooth, value in enumerate(values):
                    # Normalize the value between min_value and max_value
                    normalized_value = np.true_divide(value - min_values[key_combined],
                                                      max_values[key_combined] - min_values[key_combined])
                    comparison_combinations[i_tooth][key_combined] = normalized_value

    min_max_values = {"min_values": min_values, "max_values": max_values}

    return tooth_comparisons, comparison_combinations, min_max_values


def prepare_heatmap_distance_dicts(tooth_dicts, base_surface_polydata, kd_tree):
    """
    Prepare heatmap distance dictionaries, which represent the distances from each base surface vertex to every tooth.

    Parameters:
    tooth_dicts (list[dict]): A list of dictionaries, each containing data of a tooth.
    base_surface_polydata (vtk.vtkPolyData): The rostrum base surface polydata.
    kd_tree (scipy.spatial.KDTree): The KDTree used for nearest-neighbor queries.

    Returns:
    list[dict]: A list of dictionaries, each representing the distances from a base surface vertex to every tooth.
    """

    distance_dicts = []
    base_surface_vertices = numpy_support.vtk_to_numpy(base_surface_polydata.GetPoints().GetData())

    for base_surface_vertex in base_surface_vertices:
        distance_dict = {}

        points_within_radius_indices = kd_tree.query_ball_point(base_surface_vertex, KD_TREE_RADIUS_FOR_HEATMAP)

        for index in points_within_radius_indices:
            tooth_root_position = tooth_dicts[index]['deepest_center_point_base_surface']
            distance = np.linalg.norm(base_surface_vertex - tooth_root_position)
            distance_dict[index] = distance

        distance_dicts.append(distance_dict)

    return distance_dicts


def run_analysis_scan():
    """
    Analyze the scanned data of teeth and save the results in various formats.

    This function performs several steps:
    1. Load the scan data from a pickle file.
    2. Create a KDTree for each tooth root point.
    3. Prepare heatmap distance dictionaries.
    4. Analyze the scan.
    5. Build polydata from the tooth data.
    6. Compare the analyses of the teeth.
    7. Save the analyses in a CSV file.
    8. Smooth the polydata of the teeth.
    9. Save the complete analysis and the teeth polydata in pickle and VTK files, respectively.
    """

    # Load the scan data from pickle
    with open(helper.get_path_of_existing_file_from_config('full_teeth_scan', CONFIG), 'rb') as f:
        data = pickle.load(f)

    tooth_dicts = data['tooth_dicts']
    n_directions = data['n_directions']

    # Load the base surface
    mesh_base = vtk.vtkPLYReader()
    mesh_base.SetFileName(helper.get_path_of_existing_file_from_config('base_surface', CONFIG))
    mesh_base.Update()
    polydata_base = mesh_base.GetOutput()

    # Prepare KDTree and heatmap
    tooth_base_points = [tooth_dict['deepest_center_point_base_surface'] for tooth_dict in tooth_dicts]
    kd_tree = scipy.spatial.KDTree(tooth_base_points)
    heatmap_distance_dicts = prepare_heatmap_distance_dicts(tooth_dicts, polydata_base, kd_tree)

    analyses = analyse_scan(tooth_dicts, n_directions, kd_tree, data['voxel_length_in_micrometers'])
    teeth_polydata, _ = build_polydata_from_tooth_dicts(tooth_dicts, n_directions)

    comparisons, comparison_combinations, min_max_values = compare_tooth_analyses(analyses)

    # Save analyses in a CSV file
    df = pd.DataFrame(analyses)
    df.to_csv(helper.create_path_for_new_file_from_config('pandas_DF', CONFIG))

    # Smooth the tooth endpoints
    for tooth_dict in tooth_dicts:
        tooth_dict['tooth_endpoints'], _ = smooth_3d_points(tooth_dict['tooth_endpoints'], 1)

    smoothed_teeth_polydata, _ = build_polydata_from_tooth_dicts(tooth_dicts, n_directions)

    # print("Analyses:", analyses)

    complete_analysis = {
        "analyses": analyses,
        "comparisons": comparisons,
        "comparison_combinations": comparison_combinations,
        "min_max_values": min_max_values,
        "heatmap_distance_dicts": heatmap_distance_dicts,
    }

    # Save the complete analysis in a pickle file
    with open(helper.create_path_for_new_file_from_config('analysis_pickle', CONFIG), "wb") as f:
        pickle.dump(complete_analysis, f)

    # Save the teeth polydata in a VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(teeth_polydata)
    writer.SetFileName(helper.create_path_for_new_file_from_config('teeth_polydata', CONFIG))
    writer.Write()

    # Save the smoothed teeth polydata in a VTK file
    writer.SetInputData(smoothed_teeth_polydata)
    writer.SetFileName(helper.create_path_for_new_file_from_config('smoothed_teeth_polydata', CONFIG))
    writer.Write()


if __name__ == '__main__':
    tic = time.perf_counter()
    script_name = __file__.split('\\')[-1]
    logging.info(f'started {script_name}...\n')
    run_analysis_scan()
    logging.info(f"finished. {time.perf_counter() - tic:1.1f}s")
    exit(0)
