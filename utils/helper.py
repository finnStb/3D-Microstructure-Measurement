"""
========================
Tools for easier use of the vtk library with the rostrum
========================

Create an instance of a toolbox (including renderer) by creating an ``VtkRostrumHelper()`` object.
Use the functions then to load volumes or meshes and do tweaks.
Run .start_interactor() to start the rendering window.

"""
import logging

import vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
import h5py
import numpy as np
# import threading
import configparser
from datetime import datetime
import os
import math
from PIL import Image, ImageDraw


def print_seperator():
    """
    Print a visual separator in the console.

    :return: None
    """
    print("\n||", "=" * 100, "||\n")


def extract_cells_closest_to_point_from_polydata(polydata, point_to_search_for, point_locator):
    """
    Extracts the cells from a given PolyData object that are closest to a specified point.

    :param polydata: The PolyData object from which to extract cells.
    :type polydata: vtk.vtkPolyData
    :param point_to_search_for: The 3D point for which to find the closest cells.
    :type point_to_search_for: list of float
    :param point_locator: A PointLocator object to assist in finding the closest point.
    :type point_locator: vtk.vtkPointLocator
    :return: A new PolyData object containing only the cells closest to the specified point.
    :rtype: vtk.vtkPolyData
    """
    # Find the ID of the point closest to the search point
    point_id = point_locator.FindClosestPoint(point_to_search_for)

    # Initialize a list to store the IDs of the cells that contain the closest point
    id_list = vtk.vtkIdList()
    polydata.GetPointCells(point_id, id_list)

    # Count the number of cells that contain the closest point
    number_of_cells = id_list.GetNumberOfIds()

    # Initialize an array to store the extracted cells
    extracted_cell_array = vtk.vtkCellArray()

    # Extract each cell that contains the closest point
    for i in range(number_of_cells):
        cell_id = id_list.GetId(i)
        cell = polydata.GetCell(cell_id)
        extracted_cell_array.InsertNextCell(cell)

    # Create a new PolyData object to hold the extracted cells
    extracted_poly_data = vtk.vtkPolyData()
    # Assign the same points as the original PolyData object
    extracted_poly_data.SetPoints(polydata.GetPoints())
    # Assign only the extracted cells
    extracted_poly_data.SetPolys(extracted_cell_array)

    return extracted_poly_data


def create_path_for_new_file_from_config(data_type: str, config: configparser.ConfigParser):
    """
    Create a new file path based on the configuration parameters.

    This function reads the file name for a new file from the config.ini,
    it uses the current date and time as well as the specified data type to construct the file path.

    :param data_type: Type of data, used to select specific configuration parameters.
    :type data_type: str
    :param config: ConfigParser instance with configuration parameters.
    :type config: configparser.ConfigParser
    :return: The newly constructed file path.
    :rtype: str
    """

    # Retrieve the volume file name from the config
    volume_filename = config.get('General', 'volume_filename')

    # Get the current date and time as a string
    date_string = datetime.now().strftime('%Y-%m-%d_%H-%M')

    # Get the directory from the config.ini, depending on the data type
    directory = '../' + config.get('Directories', data_type)

    # Get the file name from the config.ini, depending on the data type
    filename = config.get('Naming', data_type) \
        .replace('{volume_filename}', volume_filename) \
        .replace('{date_string}', date_string)

    # Return the constructed file path
    return directory + filename


def get_path_of_existing_file_from_config(data_type: str, config: configparser.ConfigParser):
    """
    Retrieves the path of an existing file from a provided configuration.

    This function reads the filename of an existing file from the config.ini.
    It gets the directory name from the config.ini depending on the data type.
    It also checks whether to use the latest file or a specific file according to the data type.

    :param data_type: The type of data to be retrieved.
    :param config: The configuration parser object containing the settings.
    :return: The full path of the desired file.

    :type data_type: str
    :type config: configparser.ConfigParser
    :rtype: str
    """

    # Get the directory name from the config.ini, depending on the data type
    directory = '../' + config.get('Directories', data_type)

    if data_type == 'labels':
        return directory + config.get('General', 'labels_filename') + ".csv"

    # Get the volume file name
    volume_filename = config.get('General', 'volume_filename')

    if data_type == 'volume':
        return directory + volume_filename + ".hdf5"

    # Check whether to use the latest file or a specific file, depending on the data type
    stage = config.get('Stages', data_type)
    if stage == 'None':
        # Get the file name from the config.ini, depending on the data type
        filename = config.get('Naming', data_type) \
            .replace('{volume_filename}', volume_filename)
        return directory + filename

    use_latest_file = config.getboolean(stage, 'use_latest_' + data_type)

    if use_latest_file:
        # List all files in the directory
        files = os.listdir(directory)
        files_of_current_volume = []

        # Remove all files that do not contain the name of the volume file
        for file in files:
            if volume_filename in file:
                files_of_current_volume.append(file)

        # Sort the files by date and time in the name
        files_of_current_volume.sort()

        # Select the last file from the list
        filename = files_of_current_volume[-1]
    else:
        # Get the date and time of the specific file from the config.ini
        datetime_of_specific_file = config.get(data_type, 'datetime_of_specific_' + data_type)

        # Get the file name from the config.ini, depending on the data type
        filename = config.get('Naming', data_type) \
            .replace('{volume_filename}', volume_filename) \
            .replace('{date_string}', datetime_of_specific_file)

    # Return the file name
    return directory + filename


def calculate_color(value: float, colors: list | np.ndarray):
    """
    Calculates a gradient color based on a given value and a list or array of colors.

    :param value: A value between 0 and 1, inclusive. Determines the color's position on the gradient.
    :type value: float
    :param colors: A list or an array of colors. Each color is represented as a list of 3 RGB components,
                   where each component is a float between 0 and 1, inclusive.
    :type colors: list | np.ndarray
    :return: The calculated RGB color. Each component is a float between 0 and 1, inclusive.
    :rtype: np.ndarray

    :raises ValueError: If `colors` has fewer than 2 colors.
    """

    # Return black if value is NaN
    if np.isnan(value):
        return [0, 0, 0]

    n_colors = len(colors)

    # If there's only one color, return it
    if n_colors < 2:
        return colors[0]

    # Split the color gradient into equal intervals
    splits = n_colors - 1
    position = value * splits

    # Get the floor and ceiling positions on the gradient
    floor = math.floor(position)
    ceiling = floor + 1

    # If the ceiling is out of bounds, return the last color
    if ceiling >= n_colors:
        return colors[-1]

    # Calculate the ratio between the floor and ceiling
    ratio = position - floor

    # Compute the final color as a weighted average of the floor and ceiling colors
    rgb = ratio * np.array(colors[ceiling]) + (1 - ratio) * np.array(colors[floor])

    return rgb


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray):
    """
    Calculate the angle between two vectors in radians.

    This function calculates the angle between two vectors using the dot product.

    :param v1: The first vector as a NumPy array.
    :param v2: The second vector as a NumPy array.
    :return: The angle between the two vectors in radians.
    :rtype: float
    """

    # Calculate the cosine of the angle between the vectors
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # Clip cos_angle to the range -1 to 1 to handle possible rounding errors
    cos_angle = np.clip(cos_angle, -1, 1)

    return np.arccos(cos_angle)


def angle_between_vectors_circle(v1: np.ndarray, v2: np.ndarray, orientation: str = "x"):
    """
    Compute the angle between two vectors considering the circular nature of angles.

    :param v1: The first vector, represented as a numpy array.
    :param v2: The second vector, represented as a numpy array.
    :param orientation: A string representing the orientation for the angle measurement.
                        It can be "x", "y", or "z". Default is "x".
    :return: The angle between vectors v1 and v2, in radians.
    :rtype: float
    """

    # Define a dictionary to map orientation strings to their corresponding indices
    orientation_codes = {"x": 0, "y": 1, "z": 2}

    # Get the index of the orientation
    orientation = orientation_codes[orientation]

    # Compute the angle between vectors
    angle = angle_between_vectors(v1, v2)

    # If the sum of vectors in the orientation direction is negative, adjust the angle
    if (v1 + v2)[orientation] < 0:
        angle = 2 * math.pi - angle

    return angle


def show_2d_array_as_image(np_2d_array, marker_pos=None):
    """
    Display a 2D numpy array as an image, optionally with a red marker at a specified position.

    The input array is normalized to fit the 0-255 range for image display. If a marker position is
    provided, a red circle is drawn at that position. The displayed image will be in grayscale
    if no marker is specified, and RGB if a marker is drawn.

    :param np_2d_array: a 2D numpy array to be displayed as an image
    :type np_2d_array: numpy.ndarray
    :param marker_pos: a tuple specifying the (x, y) position of the marker; defaults to None
    :type marker_pos: tuple, optional
    """

    # Normalizing the array to fit the 0-255 range for image display.
    data = np_2d_array.ravel()
    data_min, data_max = np.min(data), np.max(data)
    normalize_factor = (data_max - data_min) / 255.0
    np_2d_array_norm = (np_2d_array - data_min) / normalize_factor

    # Creating a PIL image from the normalized array.
    img = Image.fromarray(np_2d_array_norm)

    # If marker_pos is provided, draw a red circle at that position.
    if marker_pos is not None:
        # Converting the image to RGB mode to add color.
        img = img.convert('RGB')

        # Determining the radius of the marker based on image dimensions.
        width, height = img.size
        radius = max(1, (width + height) // 400)

        # Preparing to draw on the image.
        draw = ImageDraw.Draw(img)

        # Calculating the bounding box for the marker.
        x, y = marker_pos
        bbox = (x - radius, y - radius, x + radius, y + radius)

        # Drawing the marker with a red color.
        draw.ellipse(bbox, fill=(255, 50, 50))

    # Displaying the image.
    img.show()


class VtkRostrumHelper:

    def __init__(self, window_size=(1200, 800), background_color="light_grey",
                 polylines_color="Black", auto_camera=True, render_window=None, interactor=None):
        self.renderer = vtk.vtkRenderer()
        self.renderer.UseDepthPeelingOn()  # for transparency of polydata
        self.renderer.UseDepthPeelingForVolumesOn()  # for transparency of polydata
        if render_window is not None:
            self.render_window = render_window

        else:
            self.render_window = vtk.vtkRenderWindow()
            self.render_window.SetSize(window_size)
        self.render_window.AddRenderer(self.renderer)

        if interactor is not None:
            self.interactor = interactor
        else:
            self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.interactor.SetRenderWindow(self.render_window)

        # Renderer
        self.renderer.SetBackground(vtkNamedColors().GetColor3d(background_color))
        self.render_window.Render()
        self.auto_camera = auto_camera

        self.marker_positions = []

        # Marker
        self.poly_data = []

        self.actors_dict = {}
        self.teeth_slice_volume_actor = None

        self.all_vtk_points = vtk.vtkPoints()
        self.cells = vtk.vtkCellArray()  # Create a cell array to store the lines in and add the lines to it

        # Create a polydata to store everything in
        self.poly_data_polylines = vtk.vtkPolyData()
        self.points_counter = 0

        # Add the points to the dataset
        self.poly_data_polylines.SetPoints(self.all_vtk_points)

        # Add the lines to the dataset
        self.poly_data_polylines.SetLines(self.cells)

        # Setup actor and mapper
        mapper_polylines = vtk.vtkPolyDataMapper()
        mapper_polylines.SetInputData(self.poly_data_polylines)

        actor_polylines = vtk.vtkActor()
        actor_polylines.SetMapper(mapper_polylines)
        actor_polylines.GetProperty().SetColor(vtkNamedColors().GetColor3d(polylines_color))
        self.renderer.AddActor(actor_polylines)

        self.camera: vtk.vtkCamera = self.renderer.GetActiveCamera()
        self.camera_position = self.camera.GetPosition()
        self.camera_direction = self.camera.GetDirectionOfProjection()
        self.camera_focal_distance = self.camera.GetFocalDistance()
        self.camera_view_angle = self.camera.GetViewAngle()  # in degrees
        self.camera.SetPosition(0, 1000, 1000)
        self.camera.SetViewAngle(80)
        self.sin_value = 0.0
        self.teeth_polydata_actor = None

    def follow_camera(self, new_position, weight_new_pos=0.9, offset_on=False, vector=None):
        """Moves the camera to a new position and sets the focal point to the new position.
        :param new_position: the new position of the camera
        :param weight_new_pos: the weight of the new position in the exponential moving average
        :param offset_on: if True, the camera will be offset by a sinusoidal function
        :param vector: a vector to add to the camera position
        """
        self.renderer.ResetCamera()

        if offset_on:
            offset = np.array(
                [math.sin(self.sin_value / 3.0 + 0.05), math.cos(self.sin_value + 0.09), math.sin(self.sin_value)])
            self.sin_value += 0.01
        else:
            offset = 0
        ema_position = weight_new_pos * np.array(new_position) + (1.0 - weight_new_pos) * np.array(
            self.camera.GetPosition()) + offset * 50.0

        if vector is not None:
            ema_position = ema_position + np.array(vector)

        self.camera.SetPosition(ema_position)
        self.camera.SetFocalPoint(new_position)

    def refresh(self, ignore_auto_camera=False):
        """Refreshes the render window.
        :param ignore_auto_camera:
        """
        if self.auto_camera and not ignore_auto_camera:
            self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()

        self.renderer.Render()
        self.render_window.Render()

    def start_interactor(self):
        print_seperator()
        logging.info("Starting interactor. Keep in mind that this will block any further code to be executed.")
        self.interactor.Start()

    def reset_view(self):
        self.camera.SetViewUp(0, 1, 0)
        self.camera.SetPosition(4000, 2000, 6000)

        self.refresh()

    def add_volume_as_data(self, volume, data_type, data_otsu, start=0, color_min="Green", color_max="White",
                           special_volume=None):
        # prepare data
        data = volume.ravel()
        data_min, data_max = np.min(data), np.max(data)

        importer = vtk.vtkImageImport()
        # Datentyp prüfen und DataScalarType entsprechend festlegen
        if data_type == 'uint8':
            importer.SetDataScalarTypeToUnsignedChar()
        elif data_type == 'uint16':
            importer.SetDataScalarTypeToShort()
        else:
            exit(0)

        dims = volume.shape
        importer.SetNumberOfScalarComponents(1)
        importer.SetDataExtent(0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1)
        importer.SetWholeExtent(0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1)
        importer.CopyImportVoidPointer(volume, volume.nbytes)

        acf = vtk.vtkPiecewiseFunction()
        acf.AddPoint(data_min, 0.0)
        acf.AddPoint(data_otsu, 0.0)
        acf.AddPoint(data_max, 1.0)

        color_func = vtk.vtkColorTransferFunction()
        color = vtkNamedColors().GetColor3d(color_min)
        color_func.AddRGBPoint(data_min, color[0], color[1], color[2])
        color = vtkNamedColors().GetColor3d(color_max)
        color_func.AddRGBPoint(data_max, color[0] * 2, color[1] * 2, color[2] * 2)

        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_func)
        volume_property.SetScalarOpacity(acf)
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()
        # volume_property.SetInterpolationTypeToNearest()

        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetBlendModeToComposite()
        volume_mapper.SetInputConnection(importer.GetOutputPort())

        vtk_volume = vtk.vtkVolume()
        vtk_volume.SetMapper(volume_mapper)
        vtk_volume.SetProperty(volume_property)

        self.renderer.AddVolume(vtk_volume)

        if start > 0:
            self.transform_volume(z=start)

        if special_volume is not None:
            if special_volume == "teeth_slice":
                self.teeth_slice_volume_actor = vtk_volume

    def add_volume_hdf5(self, path, data_otsu, start=0, stop=200):
        with h5py.File(path, "r") as f:
            volume_dset = f["volume"]
            meta_grp = f["meta"]
            length = int(meta_grp.attrs["length"])
            data_type = meta_grp.attrs["data type"]
            if start < 0:
                start += length + 1
            if stop < 0:
                stop += length + 1
            volume = volume_dset[start:stop]

        self.add_volume_as_data(volume, data_type, data_otsu, start)

    def calculate_rostrum_heatmap(self, value_by_tooth: list[float], tooth_dicts: list[dict],
                                  interpretations: list[dict], colors: list | np.ndarray, slider_val,
                                  heatmap_distance_dicts, hard_threshold: float = None):
        """
        Calculate a heatmap for a rostrum based on the distance from teeth to points on the rostrum.

        :param value_by_tooth: List of numerical values for each tooth.
        :param tooth_dicts: List of dictionaries with details about each tooth.
        :param interpretations: List of dictionaries with interpretation data for each tooth.
        :param colors: Array or list of colors used for the heatmap.
        :param slider_val: A scaling factor for the weight calculation.
        :param heatmap_distance_dicts: List of dictionaries with distances from each tooth to each point on the rostrum.
        :param hard_threshold: Optional. If provided, the color of the heatmap will be binary based on this threshold.
        """

        # Compute the heights for each tooth from the interpretations
        heights_per_tooth = [interpretation['height[mm]'] for interpretation in interpretations]

        # Initialize the array to store the color data for each vertex
        vertex_colors = vtk.vtkUnsignedCharArray()
        vertex_colors.SetNumberOfComponents(3)

        # Compute the weighted value and corresponding color for each point on the rostrum
        for distance_dict in heatmap_distance_dicts:

            # Initialize the summations for this point
            weight_sum = 0.0
            product_sum = 0.0

            # Compute the weighted sum and product for each tooth
            for i_tooth in distance_dict.keys():
                if not np.isnan(value_by_tooth[i_tooth]):
                    weight = 1 / (pow(distance_dict[i_tooth], slider_val * 2)) * heights_per_tooth[i_tooth]
                    weight_sum += weight
                    product_sum += weight * value_by_tooth[i_tooth]

            # Compute the value for this point
            value = np.true_divide(product_sum, weight_sum)

            # Determine the color based on the value and threshold
            if hard_threshold is None:
                color = calculate_color(value, colors=colors)
            else:
                color = colors[0] if value < hard_threshold else colors[-1]

            # Add the color to the vertex color array
            vertex_colors.InsertNextTuple3(color[0], color[1], color[2])

        # Set the vertex color array as the scalars of the rostrum's point data
        self.rostrum_base_polydata.GetPointData().SetScalars(vertex_colors)
        self.refresh(ignore_auto_camera=True)

    def add_mesh_polydata(self, polydata: vtk.vtkPolyData, color="Thistle", special_mesh: str = None):

        # Visualisierung des Polydata-Objekts mit VTK
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        if color is not None:
            actor.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d(color))
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)

        # set shading
        # actor.GetProperty().ShadingOff()
        # actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().SetInterpolationToFlat()
        # set representation to surface or wireframe
        actor.GetProperty().SetRepresentationToSurface()
        # actor.GetProperty().SetRepresentationToWireframe()

        self.renderer.AddActor(actor)

        if special_mesh is not None:

            if special_mesh == "teeth":
                self.remove_actor(unique_actor_name=special_mesh)
                self.teeth_polydata = polydata
                self.teeth_polydata_actor = actor
            elif special_mesh == "rostrum_base":
                self.rostrum_base_polydata = polydata
                self.rostrum_base_polydata_actor = actor

    def remove_actor(self, unique_actor_name: str = None):
        if unique_actor_name is not None:
            if unique_actor_name == "teeth":
                if self.teeth_polydata_actor is not None:
                    self.renderer.RemoveActor(self.teeth_polydata_actor)
                # self.renderer.RemoveActor(self.teeth_polydata_actor)
            elif unique_actor_name == "rostrum_base":
                self.renderer.RemoveActor(self.rostrum_base_polydata_actor)
            elif unique_actor_name == "teeth_slice" and self.teeth_slice_volume_actor is not None:
                self.renderer.RemoveActor(self.teeth_slice_volume_actor)
        self.refresh(ignore_auto_camera=True)

    def transform_volume(self, volume_number=-1, x=0, y=0, z=0):
        # get volume from renderer
        volumes = self.renderer.GetVolumes()
        volume = None

        for i, volume_i in enumerate(volumes):
            if i == volume_number:
                volume = volume_i
            if volume_number == -1:
                volume = volume_i

        transform = vtk.vtkTransform()
        transform.Translate(x, y, z)
        volume.SetUserTransform(transform)

    def transform_actor(self, actor_number=-1, x=0, y=0, z=0):
        # get actor from renderer
        actors = self.renderer.GetActors()
        actor = None
        if actor_number == -1:
            actor = actors.GetLastActor()
        else:
            for i, actor_i in enumerate(actors):
                if i == actor_number:
                    actor = actor_i

        transform = vtk.vtkTransform()
        transform.Translate(x, y, z)
        actor.SetUserTransform(transform)

    def add_markers(self, positions_array):
        """

        :param positions_array: [np.array[1, 2, 3], ...]
        :return:
        """

        # update marker positions
        for point in positions_array:
            self.marker_positions[-1].InsertNextPoint(point[0], point[1], point[2])

        self.poly_data[-1].SetPoints(self.marker_positions[-1])

        self.poly_data[-1].Modified()
        self.refresh()

    def new_markers_area(self, size=2.0, color="Purple"):
        self.marker_positions.append(vtk.vtkPoints())
        self.poly_data.append(vtk.vtkPolyData())

        # Create a new mapper for the marker
        markers_mapper = vtk.vtkGlyph3DMapper()
        markers_mapper.SetInputData(self.poly_data[-1])

        self.current_marker_actor = vtk.vtkActor()
        self.current_marker_actor.SetMapper(markers_mapper)
        self.current_marker_actor.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d(color))

        # Add the new actor to the renderer
        self.renderer.AddActor(self.current_marker_actor)

    def clear_markers(self):
        self.marker_positions = vtk.vtkPoints()
        self.poly_data.Modified()

    def add_polylines_actor(self, point_lists, color="Black", line_width=1, category_key=None):
        vtk_points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()  # Create a cell array to store the lines in and add the lines to it

        # Create a polydata to store everything in
        poly_data_polylines = vtk.vtkPolyData()
        points_counter = 0

        # Add the points to the dataset
        poly_data_polylines.SetPoints(vtk_points)

        # Add the lines to the dataset
        poly_data_polylines.SetLines(cells)

        # Setup actor and mapper
        mapper_polylines = vtk.vtkPolyDataMapper()
        mapper_polylines.SetInputData(poly_data_polylines)

        actor_polylines = vtk.vtkActor()
        actor_polylines.SetMapper(mapper_polylines)
        actor_polylines.GetProperty().SetColor(vtkNamedColors().GetColor3d(color))
        actor_polylines.GetProperty().SetLineWidth(line_width)

        # create a polyline for each point list given
        for point_list in point_lists:
            for point in point_list:
                vtk_points.InsertNextPoint(point)

            # connect the points to a polyline
            polyline = vtk.vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(len(point_list))
            for i_point in range(len(point_list)):
                polyline.GetPointIds().SetId(i_point, i_point + points_counter)
            points_counter += len(point_list)
            cells.InsertNextCell(polyline)

        self.renderer.AddActor(actor_polylines)

        # check if category exists in actors_dict
        if category_key not in self.actors_dict:
            # create an empty list for the category
            self.actors_dict[category_key] = []
        self.actors_dict[category_key].append(actor_polylines)

    def remove_polylines(self, category_key=None):
        if category_key not in self.actors_dict:
            logging.warning(f"category_key: {category_key},  actors_dict: {self.actors_dict}")
            logging.warning("category_key not in actors_dict")
            return

        for actor in self.actors_dict[category_key]:
            self.renderer.RemoveActor(actor)
        self.actors_dict[category_key] = []
