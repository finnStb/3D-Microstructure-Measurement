import logging
import sys
import pickle
import h5py
import numpy as np
from PyQt6 import QtGui, QtCore
from PyQt6.QtWidgets import (QApplication, QWidget, QGridLayout, QLabel, QSpinBox,
                             QDoubleSpinBox, QSlider, QCheckBox, QComboBox,
                             QColorDialog, QPushButton, QFrame)
from PyQt6.QtCore import Qt, QSize
import vtk.qt.QVTKRenderWindowInteractor
import vtk
from PyQt6.QtGui import QColor, QFont, QIcon
from pyqtgraph import PlotWidget
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from helper import calculate_color
from utils import helper
from config_and_logging import CONFIG, DEBUG_MODE, USE_FULL_VOLUME_LENGTH, CUSTOM_VOLUME_LENGTH, CUSTOM_VOLUME_START

# Read constants from the config file
PATH_UI_SETTINGS = CONFIG.get('UI', 'path_ui_settings')
MIN_NUMBER_OF_COLOR_PICKERS = CONFIG.getint('UI', 'min_number_of_color_pickers')
RELATIVE_VOLUME_SLICE_THICKNESS = eval(CONFIG.get("UI", "relative_volume_slice_thickness"))
SCREENSHOT_RESOLUTION = CONFIG.get('UI', 'screenshot_resolution')
TOOTH_CENTERS_LINE_COLOR = CONFIG.get('UI', 'tooth_centers_line_color')
TOOTH_CENTERS_LINE_WIDTH = CONFIG.getfloat('UI', 'tooth_centers_line_width')
TOOTH_FOCUS_ZOOM_HEIGHT = CONFIG.getfloat('UI', 'tooth_focus_zoom_height')
FONT_NAME = CONFIG.get('UI', 'font_name')
FONT_SIZE_DEFAULT = CONFIG.getint('UI', 'font_size_default')
RELATIVE_GRAPH_WIDGET_WIDTH = eval(CONFIG.get('UI', 'relative_graph_widget_width'))
RELATIVE_GRAPH_WIDGET_HEIGHT = eval(CONFIG.get('UI', 'relative_graph_widget_height'))
RELATIVE_SCREENSHOT_ICON_SIZE = eval(CONFIG.get('UI', 'relative_screenshot_icon_size'))


def colorize_teeth(teeth_polydata, n_directions: int, analyses: list[dict], comparisons: list[dict],
                   comparison_combinations: list[dict], is_log_scale: bool, colors: list | np.ndarray,
                   key: str, divisor_key: str = None, opacity=1.0, hard_threshold: float = None):
    """
    Colors each tooth based on analyses and comparisons. The color for each tooth is determined
    by either a specified value (key) or a combination of values (key divided by divisor_key).
    Additionally, an optional hard threshold can be specified, below which the first color is used
    and above which the last color is used.

    :param teeth_polydata: The polydata of the teeth.
    :param n_directions: The number of directions.
    :param analyses: A list of dictionaries containing analysis data for each tooth.
    :param comparisons: A list of dictionaries containing comparison data for each tooth.
    :param comparison_combinations: A list of dictionaries containing combination data for each tooth.
    :param is_log_scale: A boolean indicating whether the value used to determine the color is to be log-scaled.
    :param colors: A list or numpy array of colors to be used for coloring the teeth.
    :param key: The key to the value used to determine the color.
    :param divisor_key: The key to the value used as divisor for determining the color. If None, only the key is used.
    :param opacity: The opacity of the colors. Default is 1.0.
    :param hard_threshold: A hard threshold for the values. Below this value, the first color is used, above it the last color is used. If None, no hard threshold is applied.

    :return: A list of values used for coloring each tooth.
    """

    # Initialize an array to hold the colors for the end points of the teeth
    teeth_endpoint_colors = vtk.vtkUnsignedCharArray()
    teeth_endpoint_colors.SetNumberOfComponents(4)

    value_per_tooth = []

    # Loop through each tooth
    for i_tooth, tooth_comparison in enumerate(comparisons):
        tooth_interpretation = analyses[i_tooth]
        n_layers = len(tooth_interpretation['mean_radius_per_layer[mm]'])

        # Determine the value used for coloring
        if divisor_key == "None" or divisor_key is None:
            value = tooth_comparison[key]
        else:
            value = comparison_combinations[i_tooth][key + "_divided_by_" + divisor_key]
            if is_log_scale:
                value = ln(value)

        # Determine the color based on the value and possible hard threshold
        if hard_threshold is None:
            color = calculate_color(value, colors)
        else:
            color = colors[0] if value < hard_threshold else colors[-1]

        # Assign the color to each direction of each layer of the tooth
        for i_layer in range(n_layers):
            for i_direction in range(n_directions):
                teeth_endpoint_colors.InsertNextTuple4(color[0], color[1], color[2], opacity * 255)

        value_per_tooth.append(value)

    # Set the colors for the teeth
    teeth_polydata.GetPointData().SetScalars(teeth_endpoint_colors)

    return value_per_tooth


def ln(x, invert_on_diagonal: bool = False):
    """
    Computes the natural logarithm of `(x+1)` normalized by `ln(1+1)`. If `invert_on_diagonal` is True,
    it computes the inverse function mirrored at the diagonal, that is `2^(x) - 1`.

    :param x: The input value.
    :type x: float or numpy.ndarray
    :param invert_on_diagonal: A flag to indicate whether to compute the inverse function.
    :type invert_on_diagonal: bool, optional
    :return: The computed logarithm or its inverse function.
    :rtype: float or numpy.ndarray
    """
    if invert_on_diagonal:
        return np.exp2(x) - 1  # f(x) = ln(x+1)/ln(1+1) mirrored at the diagonal
    return np.log(x + 1) / np.log(1 + 1)  # f(x) = ln(x+1)/ln(1+1)


class UI(QWidget):
    """
    The `UI` class is used to manage the user interface of the application.
    This class is responsible for loading and updating visualization data,
    handling user input, and controlling the camera view.
    """

    def __init__(self):
        """
        Initialize the UI, setting up UI components and loading necessary data.
        """
        super().__init__()

        self.block_color_update = True
        self.load_analysis_pickle_and_polydata()

        # Initialize the UI elements
        self.create_layout()

        # Load the pickle file and set the values
        self.load_settings_pickle()

        # Show the UI
        self.helper.refresh()
        self.block_color_update = False
        self.update_tooth_colors()
        self.show()

    def create_layout(self):
        """
        Create the layout and add all the widgets to it.
        """

        # Read the scan data from pickle
        with open(helper.get_path_of_existing_file_from_config('full_teeth_scan', CONFIG), 'rb') as f:
            self.data = pickle.load(f)

        self.tooth_dicts: list[dict] = self.data['tooth_dicts']
        self.n_directions = self.data['n_directions']

        # Create a grid layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Font
        self.setFont(QFont(FONT_NAME))

        # VTK
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.render_window = self.vtkWidget.GetRenderWindow()
        self.helper = helper.VtkRostrumHelper(render_window=self.render_window, interactor=self.vtkWidget)
        self.layout.addWidget(self.vtkWidget, 0, 0, 1, 14)

        self.vtkWidget.Initialize()
        self.vtkWidget.Start()

        self.helper.add_mesh_polydata(self.original_teeth_polydata, special_mesh="teeth")

        # BOXES AND SPACERS
        box_positions = [
            [1, 7, 2, 2],
            [3, 7, 2, 3],
            [5, 7, 2, 2],
            [7, 7, 3, 2],
        ]

        # Add the boxes to the grid layout
        for box_position in box_positions:
            box = QFrame()
            box.setFrameShape(QFrame.Shape.StyledPanel)
            self.layout.addWidget(box, box_position[0], box_position[1], box_position[2], box_position[3])

        # create a vertical line as a separator
        v_line = QFrame()
        v_line.setFrameShape(QFrame.Shape.VLine)  # set the shape to a vertical line
        v_line.setFrameShadow(QFrame.Shadow.Raised)  # set the shadow to sunken
        self.layout.addWidget(v_line, 1, 3, 9, 1)  # add the line to the grid layout at column 1 and span 3 rows

        v_line = QFrame()
        v_line.setFrameShape(QFrame.Shape.VLine)  # set the shape to a vertical line
        v_line.setFrameShadow(QFrame.Shadow.Raised)  # set the shadow to sunken
        self.layout.addWidget(v_line, 1, 6, 9, 1)  # add the line to the grid layout at column 1 and span 3 rows

        # Create a button to plot and connect it to the function
        self.save_button = QPushButton("SAVE SETTINGS")
        self.save_button.clicked.connect(self.save_settings)
        self.layout.addWidget(self.save_button, 9, 0, 1, 3)

        font_bigger = QFont(FONT_NAME,
                            int(FONT_SIZE_DEFAULT * 1.2))  # slightly bigger than default font
        font_bigger.setBold(True)
        self.save_button.setFont(font_bigger)
        self.save_button.setStyleSheet("background-color: lightgrey")

        # Create a button to reset the view
        self.reset_view_button = QPushButton("RESET VIEW")
        self.reset_view_button.clicked.connect(self.helper.reset_view)
        self.layout.addWidget(self.reset_view_button, 0, 0, 0, 14,
                              alignment=QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignRight)
        self.reset_view_button.setStyleSheet("background-color: lightgrey")

        # Create a button to take a screenshot
        self.screenshot_button = QPushButton()
        self.screenshot_button.setStyleSheet("QPushButton {background-color: white}")
        self.screenshot_button.setIcon(
            QIcon("../settings/screenshotIcon.png"))  # set the icon of the button to a camera image
        self.screenshot_button.clicked.connect(self.take_screenshot)
        self.layout.addWidget(self.screenshot_button, 0, 0, 0, 0,
                              alignment=QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)

        # Create a label and a spin box for the tooth ID
        self.toothid_label = QLabel("Focus Tooth by ID")
        self.toothid_spinbox = QSpinBox()
        self.toothid_spinbox.setMinimum(0)
        self.toothid_spinbox.setMaximum(len(self.analyses) - 1)
        self.toothid_spinbox.valueChanged.connect(self.camera_on_tooth_id)
        self.layout.addWidget(self.toothid_label, 1, 0)
        self.layout.addWidget(self.toothid_spinbox, 1, 1)

        # Create a label and a double spin box for the opacity
        self.opacity_label = QLabel(" Teeth Opacity")
        self.opacity_double_spinbox = QDoubleSpinBox()
        self.opacity_double_spinbox.setMinimum(0.0)
        self.opacity_double_spinbox.setMaximum(1.0)
        self.opacity_double_spinbox.setSingleStep(0.01)
        self.opacity_double_spinbox.valueChanged.connect(self.update_opacity_slider_value)
        self.opacity_double_spinbox.valueChanged.connect(self.update_tooth_colors)
        self.layout.addWidget(self.opacity_label, 1, 7)
        self.layout.addWidget(self.opacity_double_spinbox, 1, 8)

        # Create a horizontal float slider and connect it to the double spin box (opacity)
        self.opacity_float_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_float_slider.setMinimum(0)
        self.opacity_float_slider.setMaximum(100)
        self.opacity_float_slider.setTracking(False)
        self.opacity_float_slider.valueChanged.connect(self.update_opacity_spinbox_value)
        self.layout.addWidget(self.opacity_float_slider, 2, 7, 1, 2)

        # HARD COLOR THRESHOLD
        # Create a label and a double spin box for the float slider value
        self.threshold_label = QLabel(" Hard Color Threshold")
        self.threshold_double_spinbox = QDoubleSpinBox()
        self.threshold_double_spinbox.setMinimum(0.0)
        self.threshold_double_spinbox.setMaximum(1.0)
        self.threshold_double_spinbox.setSingleStep(0.01)
        self.threshold_double_spinbox.valueChanged.connect(self.update_threshold_slider_value)
        self.threshold_double_spinbox.valueChanged.connect(self.update_tooth_colors)
        self.layout.addWidget(self.threshold_label, 3, 7)
        self.layout.addWidget(self.threshold_double_spinbox, 3, 8)

        # Create a horizontal float slider and connect it to the double spin box
        self.threshold_float_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_float_slider.setMinimum(0)
        self.threshold_float_slider.setMaximum(100)
        self.threshold_float_slider.valueChanged.connect(self.update_threshold_spinbox_value)
        self.layout.addWidget(self.threshold_float_slider, 4, 7, 1, 2)

        # Create a check box for the on/off option
        self.threshold_checkbox = QCheckBox()
        self.threshold_checkbox.setText(" ")  # for spacing

        self.threshold_checkbox.stateChanged.connect(self.update_tooth_colors)
        self.layout.addWidget(self.threshold_checkbox, 3, 9)

        # Create a check box for the on/off option
        self.smooth_label = QLabel("Smooth Teeth")
        self.smooth_checkbox = QCheckBox()
        self.smooth_checkbox.stateChanged.connect(self.update_smooth_teeth)
        self.layout.addWidget(self.smooth_checkbox, 8, 1)
        self.layout.addWidget(self.smooth_label, 8, 0)

        # VOLUME SLIDER
        # Create a label and a double spin box for the float slider value
        self.volume_label = QLabel(" Volume Slice")  # (intense computation!)
        self.layout.addWidget(self.volume_label, 5, 7)

        # Create a horizontal float slider and connect it to the double spin box
        self.volume_float_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_float_slider.setMinimum(0)
        self.volume_float_slider.setMaximum(100)
        self.volume_float_slider.setTracking(False)
        self.volume_float_slider.valueChanged.connect(self.update_volume_slice)
        self.layout.addWidget(self.volume_float_slider, 6, 7, 1, 2)

        # Create a check box for the on/off option
        self.volume_checkbox = QCheckBox()
        self.volume_checkbox.setText("(high load!)")
        self.volume_checkbox.setStyleSheet("color: grey")
        self.volume_checkbox.stateChanged.connect(self.update_volume_slice)
        self.layout.addWidget(self.volume_checkbox, 5, 8)

        # Create a label and a check box for the on/off option
        self.base_label = QLabel(" Show Rostrum Base")
        self.base_checkbox = QCheckBox()
        self.base_checkbox.stateChanged.connect(self.toggle_rostrum_base_visibility)
        self.layout.addWidget(self.base_label, 7, 7)
        self.layout.addWidget(self.base_checkbox, 7, 8)

        # SLIDER FOR HEATMAP
        # Create a label and a double spin box for the float slider value
        self.heatmap_label = QLabel(" Rostrum Heatmap")
        self.layout.addWidget(self.heatmap_label, 8, 7)

        # Create a horizontal float slider and connect it to the double spin box
        self.heatmap_float_slider = QSlider(Qt.Orientation.Horizontal)
        self.heatmap_float_slider.setMinimum(0)
        self.heatmap_float_slider.setMaximum(100)
        self.heatmap_float_slider.setTracking(False)
        self.heatmap_float_slider.valueChanged.connect(self.update_tooth_colors)
        self.layout.addWidget(self.heatmap_float_slider, 9, 7, 1, 2)

        # Create a check box for the on/off option
        self.heatmap_checkbox = QCheckBox()
        self.heatmap_checkbox.setText("(high load!)")
        self.heatmap_checkbox.setStyleSheet("color: grey")
        self.heatmap_checkbox.stateChanged.connect(self.toggle_heatmap_visibility)
        self.layout.addWidget(self.heatmap_checkbox, 8, 8)

        # TOOTH CENTERS
        # Create a label and a check box for the on/off option
        self.centers_label = QLabel("Show Tooth Centers")
        self.centers_checkbox = QCheckBox()
        self.centers_checkbox.stateChanged.connect(self.toggle_tooth_centers_visibility)
        self.layout.addWidget(self.centers_label, 7, 0)
        self.layout.addWidget(self.centers_checkbox, 7, 1)

        # Create two labels and two combo boxes for the drop down lists
        self.attribute_label = QLabel("Attribute")
        self.attribute_combobox = QComboBox()
        attribute_keys = list(self.comparisons[0].keys())
        attribute_keys.remove("id")
        self.attribute_combobox.addItems(attribute_keys)
        self.layout.addWidget(self.attribute_label, 2, 0)
        self.layout.addWidget(self.attribute_combobox, 2, 1)

        self.divisor_label = QLabel("Divisor Attribute")
        self.divisor_combobox = QComboBox()
        self.divisor_combobox.addItems(attribute_keys)
        self.layout.addWidget(self.divisor_label, 3, 0)
        self.layout.addWidget(self.divisor_combobox, 3, 1)

        # Create a check box for the on/off option for the divisor
        self.divisor_checkbox = QCheckBox()
        self.divisor_checkbox.stateChanged.connect(self.update_tooth_colors)
        self.layout.addWidget(self.divisor_checkbox, 3, 2)

        # Create a label and a check box for the on/off option
        self.log_label = QLabel("Logarithmic Scale")
        self.log_checkbox = QCheckBox()
        self.log_checkbox.setText("(only with div. attr.)")
        self.log_checkbox.setStyleSheet("color: grey")
        self.log_checkbox.stateChanged.connect(self.update_tooth_colors)
        self.layout.addWidget(self.log_label, 4, 0)
        self.layout.addWidget(self.log_checkbox, 4, 1)

        self.vector_label = QLabel("Vector")
        self.vector_combobox = QComboBox()
        self.vector_combobox.addItems(self.vector_keys)
        self.vector_combobox.currentIndexChanged.connect(self.update_vectors)
        self.layout.addWidget(self.vector_label, 5, 0)
        self.layout.addWidget(self.vector_combobox, 5, 1)
        self.previous_vector_key = "initial"

        # VECTOR
        self.vector_position_label = QLabel("Position Vector")
        self.vector_position_combobox = QComboBox()
        self.vector_position_combobox.addItems(["Tooth Tip", "Rostrum Base"])
        self.vector_position_combobox.currentIndexChanged.connect(self.update_vectors)
        self.layout.addWidget(self.vector_position_label, 6, 0)
        self.layout.addWidget(self.vector_position_combobox, 6, 1)

        # Create a list to store the color pickers and their labels
        self.color_pickers: [QPushButton] = []
        self.color_picker_labels: [QLabel] = []

        # Create min number of color pickers and their labels by default
        for i in range(MIN_NUMBER_OF_COLOR_PICKERS):
            self.add_color_picker()

        # Create a button to add more color pickers and connect it to the function
        self.add_color_button = QPushButton("Add color")
        self.add_color_button.clicked.connect(self.add_color_picker)
        self.layout.addWidget(self.add_color_button, 1, 4)

        # Create a button to remove color pickers and connect it to the function
        self.remove_color_button = QPushButton("Remove color")
        self.remove_color_button.clicked.connect(self.remove_color_picker)
        self.remove_color_button.clicked.connect(self.update_tooth_colors)
        self.layout.addWidget(self.remove_color_button, 1, 5)

        # Connect the first color picker to the function that updates the sphere color
        for color_picker in self.color_pickers:
            color_picker.clicked.connect(self.update_tooth_colors)

        self.attribute_combobox.currentIndexChanged.connect(self.update_tooth_colors)
        self.divisor_combobox.currentIndexChanged.connect(self.update_tooth_colors)

        self.graph_widget = PlotWidget()
        self.layout.addWidget(self.graph_widget, 0, 0, 1, 14,
                              alignment=QtCore.Qt.AlignmentFlag.AlignBottom | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.graph_widget.setMouseEnabled(x=False, y=False)  # disable mouse interaction
        self.graph_widget.getAxis("left").hide()  # Blenden Sie die y-Achse aus
        self.graph_widget.setBackground("lightgray")
        self.graph_widget.getAxis("bottom").setTextPen("black")
        self.graph_widget.showGrid(x=True, y=False, alpha=0.8)

    def update_tooth_colors(self):
        """
        Update the colors of the teeth based on the currently selected options.
        """
        if self.block_color_update:
            return

        colors = self.read_colors_from_color_pickers()

        key = self.attribute_combobox.currentText()
        divisor_key = self.divisor_combobox.currentText() if self.divisor_checkbox.isChecked() else "None"
        log_scale = self.log_checkbox.isChecked()
        hard_threshold_on = self.threshold_checkbox.isChecked()

        # colorize the polydata of the teeth:
        self.value_by_tooth = colorize_teeth(self.helper.teeth_polydata,
                                             n_directions=self.n_directions,
                                             analyses=self.analyses,
                                             comparisons=self.comparisons,
                                             comparison_combinations=self.comparison_combinations,
                                             is_log_scale=log_scale,
                                             colors=colors,
                                             key=key,
                                             divisor_key=divisor_key,
                                             opacity=self.opacity_double_spinbox.value(),
                                             hard_threshold=self.threshold_double_spinbox.value() if hard_threshold_on else None
                                             )

        self.helper.refresh(ignore_auto_camera=True)

        # Update the graph
        if divisor_key == "None":
            min_value = self.min_max_values["min_values"][key]
            max_value = self.min_max_values["max_values"][key]
        else:
            min_value = self.min_max_values["min_values"][key + "_divided_by_" + divisor_key]
            max_value = self.min_max_values["max_values"][key + "_divided_by_" + divisor_key]

        if hard_threshold_on:
            x_values = [min_value,
                        self.threshold_double_spinbox.value() * (max_value - min_value) + min_value,
                        max_value]
            color_points = [0, self.threshold_double_spinbox.value() - 0.00001,
                            self.threshold_double_spinbox.value() + 0.00001, 1]
        else:
            x_values = np.linspace(min_value, max_value, 5)
            color_points = np.linspace(0, 1, len(colors))

        if log_scale and divisor_key != "None" and not hard_threshold_on:
            color_points = ln(color_points, invert_on_diagonal=True)

        grad = QtGui.QLinearGradient(min_value, 0, max_value, 0)

        if hard_threshold_on:
            first_color = colors[0]
            last_color = colors[-1]

            for i_color_point in range(2):
                grad.setColorAt(color_points[i_color_point],
                                QtGui.QColor.fromRgb(first_color[0], first_color[1], first_color[2]))
                grad.setColorAt(color_points[i_color_point + 2],
                                QtGui.QColor.fromRgb(last_color[0], last_color[1], last_color[2]))

        else:
            for i_color, color in enumerate(colors):
                grad.setColorAt(color_points[i_color],
                                QtGui.QColor.fromRgb(color[0], color[1], color[2]))
        brush = QtGui.QBrush(grad)
        self.graph_widget.clear()
        self.graph_widget.plot(x=x_values, y=[1] * len(x_values), width=1.0, fillLevel=0, brush=brush)
        axis = self.graph_widget.getAxis('bottom')
        ticks = []
        tick_range = x_values[-1] - x_values[0]
        tick_max = x_values[-1]
        for x in x_values:
            # If the value is very small or very large, display as a power of ten
            if x < 0.01 and tick_range < 0.03:
                text = format(x, '.1e')
                if x == x_values[0] or x == x_values[-1]:
                    text = format(x, '.0e')
            elif x >= 100 and tick_range > 10:
                text = str(round(x))
            else:
                text = str(round(x, 2))

            ticks.append((x, text))
        axis.setTicks([ticks])

        font_default = QFont(FONT_NAME, FONT_SIZE_DEFAULT)

        if hard_threshold_on or divisor_key == "None":
            self.log_label.setStyleSheet("color: grey")
            font = font_default
            font.setStrikeOut(True)
            self.log_label.setFont(font)
        else:
            self.log_label.setStyleSheet("color: black")
            self.log_label.setFont(font_default)

        if self.heatmap_checkbox.isChecked():
            self.toggle_heatmap_visibility(self.heatmap_checkbox.isChecked())

        logging.debug("Updated tooth colors")

    # Define a custom slot that converts the slider value to a double and sets it to the double spin box
    def update_opacity_spinbox_value(self, slider_value):
        """
        Update the opacity spinbox value based on the provided slider value.

        Args:
            slider_value (int): The value of the opacity slider.
        """
        self.opacity_double_spinbox.setValue(slider_value / 100)

    # Define a custom slot that converts the double spin box value to an int and sets it to the slider
    def update_opacity_slider_value(self, spinbox_value):
        """
        Update the opacity slider value based on the provided spinbox value.

        Args:
            spinbox_value (float): The value of the opacity spinbox.
        """
        self.opacity_float_slider.setValue(int(spinbox_value * 100))

    # Define a custom slot that converts the slider value to a double and sets it to the double spin box
    def update_threshold_spinbox_value(self, slider_value):
        """
        Update the threshold spinbox value based on the provided slider value.

        Args:
            slider_value (int): The value of the threshold slider.
        """
        self.threshold_double_spinbox.setValue(slider_value / 100)

    # Define a custom slot that converts the double spin box value to an int and sets it to the slider
    def update_threshold_slider_value(self, spinbox_value):
        """
        Update the threshold slider value based on the provided spinbox value.

        Args:
            spinbox_value (float): The value of the threshold spinbox.
        """
        self.threshold_float_slider.setValue(int(spinbox_value * 100))

    def add_color_picker(self):
        """
        Add a new color picker and its label to the UI and the list.
        The color picker is a button that opens a color dialog.
        The label shows the selected color as a hex code.
        """
        colorPicker = QPushButton()
        colorPicker.clicked.connect(lambda: self.select_color(colorPicker))
        colorPickerLabel = QLabel("#FFFFFF")
        self.color_picker_labels.append(colorPickerLabel)

        self.color_pickers.append(colorPicker)
        self.select_color(colorPicker, is_new_button=True)

        # Get the next free row in the layout
        row = len(self.color_pickers) + 1  # + 1 for the QVTK widget cell height
        self.layout.addWidget(colorPicker, row, 4)
        self.layout.addWidget(colorPickerLabel, row, 5)

        self.color_pickers[-1].clicked.connect(self.update_tooth_colors)

        self.update_tooth_colors()

    def remove_color_picker(self):
        """
        Remove the last color picker and its label from the UI and the list.
        """
        # If there are less than two color pickers, do nothing
        if len(self.color_pickers) > MIN_NUMBER_OF_COLOR_PICKERS:
            colorPicker = self.color_pickers.pop()
            colorPickerLabel = self.color_picker_labels.pop()
            self.layout.removeWidget(colorPicker)
            self.layout.removeWidget(colorPickerLabel)
            colorPicker.deleteLater()
            colorPickerLabel.deleteLater()

    def select_color(self, colorPicker, is_new_button=False):
        """
        Open a color dialog and get the selected color.
        Set the color picker button background to the selected color.
        Set the color picker label text to the selected color hex code.

        Args:
            colorPicker (QPushButton): The color picker button.
            is_new_button (bool, optional): Whether the colorPicker is a new button. Defaults to False.
        """

        if is_new_button:
            color = QColor(255, 255, 255)
        else:
            color = QColorDialog.getColor()

        if color.isValid():
            colorPicker.setStyleSheet(f"background-color: {color.name()}")
            index = self.color_pickers.index(colorPicker)
            self.color_picker_labels[index].setText(color.name())

    def save_settings(self):
        """
        Create a dictionary with all the input values from the UI and save the dictionary as a pickle file.
        """
        settings_dict = {
            "toothid_spinbox": self.toothid_spinbox.value(),
            "attribute_combobox": self.attribute_combobox.currentText(),
            "divisor_combobox": self.divisor_combobox.currentText(),
            "divisor_checkbox": self.divisor_checkbox.isChecked(),
            "log_checkbox": self.log_checkbox.isChecked(),
            "vector_combobox": self.vector_combobox.currentText(),
            "vector_position_combobox": self.vector_position_combobox.currentText(),
            "centers_checkbox": self.centers_checkbox.isChecked(),
            "colors": [label.text() for label in self.color_picker_labels],
            "opacity_double_spinbox": self.opacity_double_spinbox.value(),
            "threshold_double_spinbox": self.threshold_double_spinbox.value(),
            "threshold_checkbox": self.threshold_checkbox.isChecked(),
            "volume_float_slider": self.volume_float_slider.value(),
            "volume_checkbox": self.volume_checkbox.isChecked(),
            "base_checkbox": self.base_checkbox.isChecked(),
            "heatmap_float_slider": self.heatmap_float_slider.value(),
            "heatmap_checkbox": self.heatmap_checkbox.isChecked(),
        }

        with open("../" + PATH_UI_SETTINGS, "wb") as f:
            pickle.dump(settings_dict, f)

        self.helper.remove_polylines()

    def load_settings_pickle(self):
        """
        Load the pickle file if it exists and set the input values accordingly.
        """
        with open("../" + PATH_UI_SETTINGS, "rb") as f:
            settings_dict = pickle.load(f)
        self.toothid_spinbox.setValue(settings_dict["toothid_spinbox"])
        self.attribute_combobox.setCurrentText(settings_dict["attribute_combobox"])
        self.divisor_combobox.setCurrentText(settings_dict["divisor_combobox"])
        self.divisor_checkbox.setChecked(settings_dict["divisor_checkbox"])
        self.log_checkbox.setChecked(settings_dict["log_checkbox"])
        self.vector_combobox.setCurrentText(settings_dict["vector_combobox"])
        self.vector_position_combobox.setCurrentText(settings_dict["vector_position_combobox"])
        self.centers_checkbox.setChecked(settings_dict["centers_checkbox"])
        self.opacity_double_spinbox.setValue(settings_dict["opacity_double_spinbox"])
        self.threshold_double_spinbox.setValue(settings_dict["threshold_double_spinbox"])
        self.threshold_checkbox.setChecked(settings_dict["threshold_checkbox"])
        self.volume_float_slider.setValue(settings_dict["volume_float_slider"])
        self.volume_checkbox.setChecked(settings_dict["volume_checkbox"])
        self.base_checkbox.setChecked(settings_dict["base_checkbox"])
        self.heatmap_float_slider.setValue(settings_dict["heatmap_float_slider"])
        self.heatmap_checkbox.setChecked(settings_dict["heatmap_checkbox"])

        for i in range(len(settings_dict["colors"]) - len(self.color_pickers)):
            self.add_color_picker()
        for i in range(len(self.color_pickers)):
            color = settings_dict["colors"][i]
            self.color_pickers[i].setStyleSheet(f"background-color: {color}")
            self.color_picker_labels[i].setText(color)

    def load_analysis_pickle_and_polydata(self):
        """
        Load the pickle file containing analysis data and polydata.
        """
        # Use the pickle module to read the analysis and polydata information from the file
        with open(helper.get_path_of_existing_file_from_config('analysis_pickle', CONFIG), "rb") as f:
            # Use pickle.load to read the dict from the file
            complete_analysis = pickle.load(f)

        # Access the variables from the dict
        self.analyses = complete_analysis["analyses"]
        self.comparisons = complete_analysis["comparisons"]
        self.comparison_combinations = complete_analysis["comparison_combinations"]
        self.min_max_values = complete_analysis["min_max_values"]
        self.heatmap_distance_dicts = complete_analysis["heatmap_distance_dicts"]

        # Prepare list of keys which contain the word "vector"
        self.vector_keys = [s for s in self.analyses[0].keys() if "vector" in s]

        # Add "None" at the start of the list for user selection
        self.vector_keys.insert(0, "None")

        # Load original and smoothed teeth polydata using VTK's PolyDataReader
        vtk_reader = vtk.vtkPolyDataReader()
        vtk_reader.SetFileName(helper.get_path_of_existing_file_from_config('teeth_polydata', CONFIG))
        vtk_reader.Update()
        self.original_teeth_polydata = vtk_reader.GetOutput()

        vtk_reader = vtk.vtkPolyDataReader()
        vtk_reader.SetFileName(helper.get_path_of_existing_file_from_config('smoothed_teeth_polydata', CONFIG))
        vtk_reader.Update()
        self.smoothed_teeth_polydata = vtk_reader.GetOutput()

        # Load the rostrum base polydata using VTK's PLYReader
        ply_reader = vtk.vtkPLYReader()
        ply_reader.SetFileName(helper.get_path_of_existing_file_from_config('base_surface', CONFIG))
        ply_reader.Update()
        self.rostrum_base_polydata = ply_reader.GetOutput()

    def camera_on_tooth_id(self, tooth_id):
        """
        Move the camera to focus on a specific tooth.

        Args:
            tooth_id (int): The ID of the tooth to focus on.
        """
        # Set the camera to follow the target tooth's tip with specific zoom height
        self.helper.follow_camera(
            self.tooth_dicts[tooth_id]["tooth_tip"], 1.0,
            vector=self.analyses[tooth_id]["vector_under_tooth_tip"] * TOOTH_FOCUS_ZOOM_HEIGHT)
        self.helper.refresh(ignore_auto_camera=True)

    # toggles the visibility of the rostrum base
    def toggle_rostrum_base_visibility(self, checked):
        """
        Toggle the visibility of the rostrum base.

        Args:
            checked (bool): Whether the rostrum base should be visible.
        """
        if checked:
            self.helper.add_mesh_polydata(self.rostrum_base_polydata, special_mesh="rostrum_base")
        else:
            self.helper.remove_actor(unique_actor_name="rostrum_base")

    def toggle_tooth_centers_visibility(self, checked):
        """
        Toggle the visibility of the tooth centers.

        Args:
            checked (bool): Whether the tooth centers should be visible.
        """
        if checked:
            for tooth_dict in self.tooth_dicts:
                self.helper.add_polylines_actor(
                    [tooth_dict["tooth_centers"]],
                    category_key="tooth_centers",
                    color=TOOTH_CENTERS_LINE_COLOR,
                    line_width=TOOTH_CENTERS_LINE_WIDTH
                )
        else:
            self.helper.remove_polylines(category_key="tooth_centers")
        self.helper.refresh(ignore_auto_camera=True)

    def toggle_volume_visibility(self, checked):
        """
        Toggle the visibility of the volume.

        Args:
            checked (bool): Whether the volume should be visible.
        """
        if checked:
            self.update_volume_slice()
        else:
            self.helper.remove_actor(unique_actor_name="teeth_slice")
        self.helper.refresh(ignore_auto_camera=True)

    def toggle_heatmap_visibility(self, checked):
        """
        Toggle the visibility of the heatmap.

        Args:
            checked (bool): Whether the heatmap should be visible.
        """
        if self.block_color_update:
            return

        if checked and self.base_checkbox.isChecked():

            self.helper.calculate_rostrum_heatmap(self.value_by_tooth, self.tooth_dicts, self.analyses,
                                                  self.read_colors_from_color_pickers(),
                                                  self.heatmap_float_slider.value() / 100,
                                                  self.heatmap_distance_dicts,
                                                  hard_threshold=self.threshold_double_spinbox.value() if self.threshold_checkbox.isChecked() else None)  # , self.kd_tree)

        else:
            self.rostrum_base_polydata.GetPointData().SetScalars(None)  # reset color to actors color
            self.helper.refresh(ignore_auto_camera=True)

    def update_vectors(self):
        """
        Update the vectors based on the currently selected options.
        """
        if self.previous_vector_key != "initial":
            self.helper.remove_polylines(category_key=self.previous_vector_key)

        category_key = self.vector_combobox.currentText()
        if category_key != "None":
            for tooth_id in range(len(self.analyses)):
                vector = self.analyses[tooth_id][category_key]

                if self.vector_position_combobox.currentText() == "Tooth Tip":
                    position_vector = self.tooth_dicts[tooth_id]["tooth_tip"]
                elif self.vector_position_combobox.currentText() == "Rostrum Base":
                    position_vector = self.tooth_dicts[tooth_id]["deepest_center_point_base_surface"]
                else:
                    position_vector = [0, 0, 0]  # default

                self.helper.add_polylines_actor([[position_vector, position_vector + vector]], color="white",
                                                line_width=2, category_key=category_key)
        self.previous_vector_key = category_key
        self.helper.refresh(ignore_auto_camera=True)

    def resizeEvent(self, event):
        """
        Handle the resize event, updating the graph widget and screenshot button size.

        Args:
            event (QResizeEvent): The resize event.
        """
        self.graph_widget.setFixedSize(int(self.size().width() * RELATIVE_GRAPH_WIDGET_WIDTH),
                                       int(self.size().height() * RELATIVE_GRAPH_WIDGET_HEIGHT))
        screenshot_icon_size = round(self.size().width() * RELATIVE_SCREENSHOT_ICON_SIZE)
        self.screenshot_button.setIconSize(QSize(screenshot_icon_size, screenshot_icon_size))

    def update_volume_slice(self):
        """
        Update the volume slice based on the current slider value.
        """
        self.helper.remove_actor(unique_actor_name="teeth_slice")

        if self.volume_checkbox.isChecked():
            slider_val = self.volume_float_slider.value() / 100

            with h5py.File(helper.get_path_of_existing_file_from_config('volume', CONFIG), "r") as f:
                volume_dset = f["volume"]
                meta_grp = f["meta"]
                volume_length = int(meta_grp.attrs["length"])
                volume_data_type = meta_grp.attrs["data type"]
                slice_width = round(RELATIVE_VOLUME_SLICE_THICKNESS * volume_length)
                slice_start = round(slider_val * volume_length - slice_width * slider_val)
                slice_stop = round(slider_val * volume_length + slice_width * (1 - slider_val))
                volume_slice = volume_dset[slice_start:slice_stop]
            # read pre-examination pickle
            import pickle
            pre_examination = pickle.load(
                open(helper.get_path_of_existing_file_from_config('pre-examination', CONFIG), 'rb'))
            otsu_threshold = pre_examination['otsu_threshold']
            self.helper.add_volume_as_data(volume_slice, volume_data_type, otsu_threshold, slice_start,
                                           special_volume="teeth_slice")
            self.helper.refresh(ignore_auto_camera=True)

    def read_colors_from_color_pickers(self):
        """
        Read the colors from the color pickers.

        Returns:
            list: A list of the selected colors.
        """
        colors = []
        for color in self.color_picker_labels:
            color = QColor(color.text())
            # append rgb values to colors as tuples
            colors.append([color.red(), color.green(), color.blue()])
        return colors

    def take_screenshot(self):
        """
        Take a screenshot of the current view and save it as a PNG file.
        """

        logging.info("Taking screenshot")
        screenshot_resolution = tuple(map(int, SCREENSHOT_RESOLUTION.split(",")))

        # Create a resizing window to image filter
        rw2if = vtk.vtkResizingWindowToImageFilter()
        rw2if.SetInput(self.render_window)
        rw2if.SetSize(screenshot_resolution[0], screenshot_resolution[1])

        # Create a PNG writer
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(helper.create_path_for_new_file_from_config('screenshot', CONFIG))
        writer.SetInputConnection(rw2if.GetOutputPort())
        writer.Write()

        # Also save a screenshot of the graph widget
        graph_widget_screenshot = self.graph_widget.grab(self.graph_widget.rect())
        graph_widget_screenshot.save(
            f"{helper.create_path_for_new_file_from_config('screenshot_graph_widget', CONFIG)}")

    def update_smooth_teeth(self, checked):
        """
        Update the display of the teeth, switching between the original and smoothed teeth.

        Args:
            checked (bool): Whether to display the smoothed teeth.
        """
        self.helper.add_mesh_polydata(
            self.smoothed_teeth_polydata if checked else self.original_teeth_polydata,
            special_mesh="teeth"
        )
        self.update_tooth_colors()


def run_ui():
    # Create an application and an instance of the UI class
    app = QApplication(sys.argv)
    app.setAttribute(QtCore.Qt.ApplicationAttribute.AA_Use96Dpi)
    ui = UI()
    # Start the application loop
    app.exec()


if __name__ == '__main__':
    run_ui()
