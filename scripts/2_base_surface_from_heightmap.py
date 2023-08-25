import logging
import pickle
import time

import h5py
import numpy as np
import vtk
import cv2
from scipy.ndimage import gaussian_filter

from config_and_logging import CONFIG, DEBUG_MODE, USE_FULL_VOLUME_LENGTH, CUSTOM_VOLUME_LENGTH, CUSTOM_VOLUME_START
from utils import helper


# Read constants from the config file
# These constants provide the parameters for creating the base surface of the rostrum
EXTRAPOLATE_Z_FACTOR = CONFIG.getfloat('Base Surface', 'extrapolate_z_factor')
BASE_SURFACE_FACE_EDGE_LENGTH = CONFIG.getfloat('Base Surface', 'base_surface_face_edge_length')
MAX_CHUNK_SIZE = CONFIG.getfloat('Base Surface', 'max_chunk_size')
MIN_CHUNK_SIZE = CONFIG.getfloat('Base Surface', 'min_chunk_size')
GRADIENT_RELATIVE_LENGTH = CONFIG.getfloat('Base Surface', 'gradient_relative_length')
QUANTILE = CONFIG.getfloat('Base Surface', 'quantile')
BILATERAL_D = CONFIG.getint('Base Surface', 'bilateral_d')
BILATERAL_SIGMA_COLOR = CONFIG.getfloat('Base Surface', 'bilateral_sigma_color')
BILATERAL_SIGMA_SPACE = CONFIG.getfloat('Base Surface', 'bilateral_sigma_space')
BOX_KSIZE = CONFIG.getint('Base Surface', 'box_ksize')
GAUSSIAN_SIGMA = CONFIG.getfloat('Base Surface', 'gaussian_sigma')
VTK_VIEW_PARTS = CONFIG.getint('Base Surface', 'vtk_view_parts')
VTK_VIEW_RELATIVE_LENGTH = eval(CONFIG.get('Base Surface', 'vtk_view_relative_length'))


# Read the heightmap data from the hdf5 file
# The heightmap provides a representation of the rostrum's geometry in 2D
with h5py.File(helper.get_path_of_existing_file_from_config('heightmap', CONFIG), 'r') as f:
    dataset = f['data']
    HEIGHTMAP = dataset[:]
    RESOLUTION_FACTOR = dataset.attrs['resolution_factor']
    VOLUME_SHAPE = dataset.attrs['volume.shape']
    VOLUME_WHOLE_LENGTH = dataset.attrs['volume.whole_length']
    VOLUME_START = dataset.attrs['volume.start']
    ELLIPSE_CIRCUMFERENCE = dataset.attrs['ellipse_circumference']


# Read pre-examination pickle data
pre_examination = pickle.load(open(helper.get_path_of_existing_file_from_config('pre-examination', CONFIG), 'rb'))
OTSU_THRESHOLD = pre_examination['otsu_threshold']
ROSTRUM_IS_POINTED_TO_THE_FRONT = pre_examination['rostrum_is_pointed_to_front']
AXIS_CENTER_X = pre_examination['axis_center_x']
AXIS_CENTER_Y = pre_examination['axis_center_y']


def create_base_surface_mesh_from_ordered_points(vertices: np.ndarray) -> vtk.vtkPolyData:
    """
    This function takes an array of ordered 3D points representing a surface in a space.
    It creates a polydata mesh in VTK by connecting the points. The points are assumed to be ordered
    such that they form a continuous surface. It first creates triangles between adjacent points
    and then converts these triangles into a polydata mesh.

    Args:
        vertices (np.ndarray): A 2D array of 3D points representing the surface.

    Returns:
        vtk.vtkPolyData: A VTK polydata mesh representing the surface.
    """
    z_length = vertices.shape[0]
    n_circumferences = vertices.shape[1]

    # Create the polygons
    # Rectangles would be easier to create, but triangles are needed for operations on the mesh like smoothing
    # Here, we loop over each point in the array, creating two triangles for each set of four adjacent points
    triangles = vtk.vtkCellArray()
    for z in range(len(vertices) - 1):
        for i_circumference in range(n_circumferences):
            # First triangle
            triangle1 = vtk.vtkTriangle()
            triangle1.GetPointIds().SetId(0, z * n_circumferences + i_circumference)
            triangle1.GetPointIds().SetId(1, z * n_circumferences + (i_circumference + 1) % n_circumferences)
            triangle1.GetPointIds().SetId(2, (z + 1) * n_circumferences + i_circumference)
            triangles.InsertNextCell(triangle1)

            # Second triangle
            triangle2 = vtk.vtkTriangle()
            triangle2.GetPointIds().SetId(0, (z + 1) * n_circumferences + i_circumference)
            triangle2.GetPointIds().SetId(1, z * n_circumferences + (i_circumference + 1) % n_circumferences)
            triangle2.GetPointIds().SetId(2, (z + 1) * n_circumferences + (i_circumference + 1) % n_circumferences)
            triangles.InsertNextCell(triangle2)

    # Create the point cloud
    # Each point in the cloud corresponds to a point in the heightmap
    point_cloud = vtk.vtkPoints()
    for z in range(z_length):
        for point in vertices[z]:
            point_cloud.InsertNextPoint(point)

    # Create the vtk Polydata object
    polydata_mesh = vtk.vtkPolyData()
    polydata_mesh.SetPoints(point_cloud)
    polydata_mesh.SetPolys(triangles)

    return polydata_mesh


def extrapolate_heightmap(input_heightmap: np.ndarray, extrapolate_z_factor: float, dtype: type) -> tuple:
    """
    This function takes a 2D heightmap array and extrapolates it along both dimensions.
    This is done by creating a larger array and placing the input heightmap in the center, then filling
    the surrounding areas by copying and extrapolating the data from the input.

    Args:
        input_heightmap (np.ndarray): The 2D array representing the original heightmap.
        extrapolate_z_factor (float): The factor by which to extrapolate the heightmap.
        dtype (type): The desired data type of the extrapolated heightmap.

    Returns:
        tuple: A tuple containing the extrapolated heightmap and the start and end indices
               along both dimensions where the original heightmap was placed.
    """
    extrapolate_z = round(input_heightmap.shape[0] * extrapolate_z_factor)
    extrapolate_circ = input_heightmap.shape[1]

    # Create an array to hold the extrapolated heightmap
    # This array is larger than the input heightmap, so it can contain the extrapolated data in the middle
    heightmap_extra = np.zeros(
        (input_heightmap.shape[0] + extrapolate_z * 2, input_heightmap.shape[1] + extrapolate_circ * 2), dtype)
    # (* 2 for left, right & top, bottom)

    # Calculate the index range for the small array in the large array
    z_start = (heightmap_extra.shape[0] - input_heightmap.shape[0]) // 2
    z_end = z_start + input_heightmap.shape[0]
    circ_start = (heightmap_extra.shape[1] - input_heightmap.shape[1]) // 2
    circ_end = circ_start + input_heightmap.shape[1]

    # Insert the small array into the large array
    heightmap_extra[z_start:z_end, circ_start:circ_end] = input_heightmap

    # Copy the whole heightmap to the left and right
    for z in range(input_heightmap.shape[0]):
        for circ in range(input_heightmap.shape[1]):
            heightmap_extra[z + extrapolate_z, circ] = input_heightmap[z, circ]
            heightmap_extra[z + extrapolate_z, -circ - 1] = input_heightmap[z, -circ - 1]  # Reverse the values

    # Extrapolate the heightmap at the top and bottom
    for circ in range(heightmap_extra.shape[1]):
        gradient_top = \
            (heightmap_extra[0 + extrapolate_z, circ] -
             heightmap_extra[round(heightmap_extra.shape[0] * GRADIENT_RELATIVE_LENGTH) + extrapolate_z, circ]) \
            * GRADIENT_RELATIVE_LENGTH

        gradient_bottom = \
            (heightmap_extra[-1 - extrapolate_z, circ] -
             heightmap_extra[-round(input_heightmap.shape[0] * GRADIENT_RELATIVE_LENGTH) - extrapolate_z, circ]) \
            * GRADIENT_RELATIVE_LENGTH

        # logging.debug(f"len extrapolation: {round(heightmap_extra.shape[0] * GRADIENT_RELATIVE_LENGTH)}")
        # logging.debug(f"gradient_top: {gradient_top}")
        # logging.debug(f"gradient_bottom: {gradient_bottom}")

        for z_extra in range(extrapolate_z):
            heightmap_extra[extrapolate_z - z_extra - 1, circ] = heightmap_extra[extrapolate_z, circ] + gradient_top * (z_extra + 1)
            heightmap_extra[-extrapolate_z + z_extra, circ] = heightmap_extra[-extrapolate_z - 1, circ] + gradient_bottom * (z_extra + 1)

    return heightmap_extra, z_start, z_end, circ_start, circ_end


def apply_filters_to_heightmap(input_heightmap: np.ndarray) -> np.ndarray:
    """
    This function applies a series of image filters to a given heightmap to smooth it.
    The filters are applied sequentially and include a bilateral filter, a box filter, and a Gaussian filter.

    Args:
        input_heightmap (np.ndarray): The 2D array representing the original heightmap.

    Returns:
        np.ndarray: The filtered heightmap.
    """
    filtered_heightmap = cv2.bilateralFilter(input_heightmap, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
    filtered_heightmap = cv2.boxFilter(filtered_heightmap, -1, (BOX_KSIZE, BOX_KSIZE), True)
    filtered_heightmap = gaussian_filter(filtered_heightmap, GAUSSIAN_SIGMA)

    return filtered_heightmap


def create_median_heightmap_from_heightmap(input_heightmap: np.ndarray) -> np.ndarray:
    """
    This function takes a 2D heightmap array and creates a median heightmap from it.
    The median heightmap is a condensed version of the original where each value is
    the median of a chunk of values from the original heightmap.

    Args:
        input_heightmap (np.ndarray): The 2D array representing the original heightmap.

    Returns:
        np.ndarray: The median heightmap.
    """

    # Calculate the size of the new heightmap in terms of the number of Z slices and angles
    z_size = round(VOLUME_SHAPE[0] / BASE_SURFACE_FACE_EDGE_LENGTH)
    n_angles = int(ELLIPSE_CIRCUMFERENCE / BASE_SURFACE_FACE_EDGE_LENGTH)
    median_heightmap = np.zeros((z_size + (1 if USE_FULL_VOLUME_LENGTH else 0), n_angles))

    heightmap_extra, _, _, _, _ = extrapolate_heightmap(input_heightmap, 0, np.float32)

    if DEBUG_MODE == 'high':
        helper.show_2d_array_as_image(heightmap_extra)

    # The number of pixels to skip in the whole heightmap to have the last chunk end at the last pixel of the heightmap
    pixels_to_skip_in_heightmap = (VOLUME_SHAPE[0] % BASE_SURFACE_FACE_EDGE_LENGTH) / RESOLUTION_FACTOR

    tip_orientation_offset = 1 if ROSTRUM_IS_POINTED_TO_THE_FRONT and USE_FULL_VOLUME_LENGTH else 0
    median_height = np.median(input_heightmap)

    # Loop over each Z slice and angle (of the smaller median_heightmap), extracting a chunk of the heightmap_extra
    # and calculating its median.
    for z in range(z_size):
        z_row_mean_height = np.mean(input_heightmap[round(z * BASE_SURFACE_FACE_EDGE_LENGTH / RESOLUTION_FACTOR)])
        scan_size = round((MAX_CHUNK_SIZE - MIN_CHUNK_SIZE) / 2
                          * min(z_row_mean_height / median_height, 1)
                          + MIN_CHUNK_SIZE / 2)

        center_z = round(z * (BASE_SURFACE_FACE_EDGE_LENGTH / RESOLUTION_FACTOR + pixels_to_skip_in_heightmap / z_size))
        from_z = max(center_z - scan_size, 0)
        to_z = min(center_z + scan_size, heightmap_extra.shape[0])

        for i_angle in range(n_angles):
            center_angle_i = round(i_angle * BASE_SURFACE_FACE_EDGE_LENGTH / RESOLUTION_FACTOR) + input_heightmap.shape[1]
            from_angle_i = max(center_angle_i - scan_size, 0)
            to_angle_i = min(center_angle_i + scan_size, heightmap_extra.shape[1])

            chunk = heightmap_extra[from_z:to_z, from_angle_i:to_angle_i]
            chunk_values = chunk.ravel()
            chunk_values.sort()
            median_heightmap[z + tip_orientation_offset, i_angle] = chunk_values[round(QUANTILE * chunk_values.size)]

    return median_heightmap


def load_volume_parts_in_vtk_helper(vtk_helper: helper.VtkRostrumHelper) -> None:
    """
    This function loads parts of the volume data into the VTK helper for visualization.
    The parts to load are determined by the configuration settings.

    Args:
        vtk_helper (helper.VtkRostrumHelper): The VTK helper object to load the data into.
    """
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
        if not USE_FULL_VOLUME_LENGTH:
            from OpenGL.GL import GL_MAX_3D_TEXTURE_SIZE, glGetIntegerv
            logging.debug(f"OPENGL GL_TEXTURE_3D: {glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE)}")
            if (start - stop) > glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE):
                logging.warning("The volume is too large to be displayed as a 3D texture.")
            else:
                vtk_helper.add_volume_as_data(volume_dset[start:stop], data_type, OTSU_THRESHOLD)
        else:
            for i in range(VTK_VIEW_PARTS):
                relative_part = i / (VTK_VIEW_PARTS - 1)
                start_part = int((relative_part - relative_part * VTK_VIEW_RELATIVE_LENGTH) * stop + start)
                stop_part = int((relative_part + (1 - relative_part) * VTK_VIEW_RELATIVE_LENGTH) * stop + start)
                vtk_helper.add_volume_as_data(volume_dset[start_part:stop_part], data_type, OTSU_THRESHOLD, start_part)


def run_base_surface_creation() -> None:
    """
    This function carries out the full process of creating a base surface from the heightmap.
    It includes creating a median heightmap, extrapolating the median heightmap,
    applying filters to it, and finally creating a polydata mesh from the processed heightmap.
    The created mesh is then saved to a file and visualized.
    """
    logging.debug("HEIGHTMAP.shape: " + str(HEIGHTMAP.shape))

    # Create a median heightmap from the heightmap
    median_heightmap = create_median_heightmap_from_heightmap(HEIGHTMAP)
    if DEBUG_MODE == 'high':
        helper.show_2d_array_as_image(median_heightmap)

    # Extrapolate the median heightmap and apply filters to it
    # (cv2 filters require np.float32 as dtype)
    median_heightmap_extra, z_start, z_end, circ_start, circ_end = extrapolate_heightmap(median_heightmap, EXTRAPOLATE_Z_FACTOR, np.float32)
    if DEBUG_MODE == 'high':
        helper.show_2d_array_as_image(median_heightmap_extra)
    median_heightmap_extra = apply_filters_to_heightmap(median_heightmap_extra)

    # Extract the original small array size from the larger array
    median_heightmap = median_heightmap_extra[z_start:z_end, circ_start:circ_end]

    # set every element in the last row to 0 (for the rostrum tip)
    if USE_FULL_VOLUME_LENGTH:
        if ROSTRUM_IS_POINTED_TO_THE_FRONT:
            median_heightmap[0, :] = 0
        else:
            median_heightmap[-1, :] = 0

    if DEBUG_MODE != 'off':
        helper.show_2d_array_as_image(median_heightmap)

    # Prepare data for the base_surface mesh creation by arranging 3d points (vertices) in the shape of
    # the heightmap in order to connect them as a mesh.
    n_angles = median_heightmap.shape[1]
    angle_unit = 2 * np.pi / n_angles
    z_length = median_heightmap.shape[0]
    z_unit = VOLUME_SHAPE[0] / (z_length - 1)

    # Calculate the center point of the volume by taking half of its dimensions and the rostrum tip
    axis_center_point_start = np.array([VOLUME_SHAPE[2] / 2, VOLUME_SHAPE[1] / 2])
    axis_center_point_end = np.array([AXIS_CENTER_X, AXIS_CENTER_Y])

    vertices = np.zeros((z_length, n_angles), dtype=object)  # vertices per row column combination
    center_points_for_visualization = []
    for z in range(z_length):
        z_relative_progress_in_whole_volume = ((VOLUME_START * z_length / VOLUME_WHOLE_LENGTH) + z * VOLUME_SHAPE[0] / VOLUME_WHOLE_LENGTH) / z_length

        axis_center_for_current_z = z_relative_progress_in_whole_volume ** 2 \
            * (axis_center_point_end - axis_center_point_start) + axis_center_point_start

        # if volume tip is pointed to the front, the z axis is inverted
        if ROSTRUM_IS_POINTED_TO_THE_FRONT:
            z = z_length - z - 1

        center_points_for_visualization.append([axis_center_for_current_z[0], axis_center_for_current_z[1], z * z_unit])

        for i_angle in range(n_angles):
            alpha = i_angle * angle_unit
            z_value = z * z_unit
            height = median_heightmap[z, i_angle]
            vertices[z, i_angle] = [height * np.sin(alpha) + axis_center_for_current_z[0],
                                    height * np.cos(alpha) + axis_center_for_current_z[1], z_value]

    polydata_mesh = create_base_surface_mesh_from_ordered_points(vertices)

    # Write the base surface to a vtk file
    writer = vtk.vtkPLYWriter()
    writer.SetInputData(polydata_mesh)
    writer.SetFileName(helper.create_path_for_new_file_from_config('base_surface', CONFIG))
    writer.Write()

    # Show the base surface in VTK
    vtk_helper = helper.VtkRostrumHelper()
    vtk_helper.add_mesh_polydata(polydata_mesh)
    vtk_helper.add_polylines_actor([center_points_for_visualization])

    # add polylines to the vtk helper around the volume at the outline
    vtk_helper.add_polylines_actor([[[0, 0, 0], [0, 0, VOLUME_SHAPE[0]], [VOLUME_SHAPE[2], 0, VOLUME_SHAPE[0]], [VOLUME_SHAPE[2], 0, 0], [0, 0, 0]],
                                    [[0, VOLUME_SHAPE[1], 0], [0, VOLUME_SHAPE[1], VOLUME_SHAPE[0]], [VOLUME_SHAPE[2], VOLUME_SHAPE[1], VOLUME_SHAPE[0]], [VOLUME_SHAPE[2], VOLUME_SHAPE[1], 0], [0, VOLUME_SHAPE[1], 0]],
                                    [[0, 0, 0], [0, VOLUME_SHAPE[1], 0]],
                                    [[0, 0, VOLUME_SHAPE[0]], [0, VOLUME_SHAPE[1], VOLUME_SHAPE[0]]],
                                    [[VOLUME_SHAPE[2], 0, VOLUME_SHAPE[0]], [VOLUME_SHAPE[2], VOLUME_SHAPE[1], VOLUME_SHAPE[0]]],
                                    [[VOLUME_SHAPE[2], 0, 0], [VOLUME_SHAPE[2], VOLUME_SHAPE[1], 0]]])

    # Show parts of the volume to check if the generated base surface fits the volume
    load_volume_parts_in_vtk_helper(vtk_helper)

    #  # todo keep this animation code for presentation:
    # helper.auto_camera = True
    # for z in range(z_length):
    #     for point in vertices[z]:
    #         helper.add_markers_medium([point])
    #         time.sleep(0.01)

    vtk_helper.start_interactor()


if __name__ == '__main__':
    tic = time.perf_counter()
    script_name = __file__.split('\\')[-1]
    logging.info(f'started {script_name}...\n')
    run_base_surface_creation()
    logging.info(f"finished. {time.perf_counter() - tic:1.1f}s")
    exit(0)
