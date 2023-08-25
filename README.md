# 3D-Microstructure-Measurement

## Instructions
Put the volume hdf5 and the labels csv in the data/input/ folder.
Set the names of the input files to use in the config.ini in [General].
Execute the python scripts in order:
- run "0_pre_examination.py" __main__
- run "1_create_heightmap.py" __main__
- run "2_base_surface_from_heightmap.py" __main__
- run "3_scan_teeth.py" __main__
- run "4_analyze_scan.py" __main__
- run "5_results_ui.py" __main__

There is no controller to run the whole process automatically. 
You have to run each step manually, because you have to check the results of each step.


## config.ini parameters

| Parameter                          | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| **[General]**                        |                                                                             |
| debug_mode                         | Options for debug mode: off/low/high                                        |
| volume_filename                    | Filename for the volume data                                                |
| labels_filename                    | Filename for the 3D labels data of the microteeth                           |
| use_full_volume_length             | Use the full volume length or custom length (True/False)                    |
| custom_volume_length               | Custom length for the volume if not using the full length                   |
| custom_volume_start                | Custom starting point for the volume if not using the full length           |
|                                    |                                                                             |
| **[Pre-Examination]**                |                                                                             |
| volume_tip_side_relative_length    | Relative length of the volume's tip side used for pre-examination           |
| volume_back_side_relative_length   | Relative length of the volume's back side used for pre-examination          |
| use_fixed_otsu_threshold           | Use a fixed Otsu threshold for image segmentation (True/False)                                |
| fixed_otsu_threshold               | Fixed Otsu threshold value if `use_fixed_otsu_threshold` is True                              |
|                                    |                                                                             |
| **[Height Map]**                |                                                                             |
| relative_diameter_of_ellipse_in_volume| Relative diameter of the ellipse in the volume, used to calculate the circumference of the ellipse that represents the rostrum outline at the cross section of the rostrum's end. |
| resolution_factor                    | Factor to determine the resolution of the height map, affecting the size of the height map array. |
| step_length                          | Step length used in the calculation of the height map, affecting the precision of the distance measurements. |
| scan_radius_margin_factor            | Margin factor for the scan radius, used to define the starting point for the scan from the center to the boundary. |
| threshold_factor                     | Factor applied to the threshold value (Otsu's threshold) to determine the boundary of the rostrum in the volume. | | |
| **[Base Surface]**                |                                                                             |
| use_latest_heightmap             | Use the latest created heightmap of the current dataset or a specific one (True/False)  |
| datetime_of_specific_heightmap   | datetime of specific heightmap if `use_latest_heightmap` is True |
| extrapolate_z_factor             | relative length of the extrapolation in z direction |
| base_surface_face_edge_length    | the disired length of the rostrum surface polydata in z direction |
| max_chunk_size                   | maximum size of the median filter scan chunks |
| min_chunk_size                   | minimum size of the median filter scan chunks |
| gradient_relative_length         | relative length of the considered area at the edges for extrapolation |
| bilateral_d                      | d param for the bilateral filter |
| bilateral_sigma_color            | sigma_color param for the bilateral filter |
| bilateral_sigma_space            | sigma_space param for the bilateral filter |
| box_ksize                        | ksize param for the box filter |
| gaussian_sigma                   | sigma param for the gaussian filter |
| quantile                         | quantile to use for each chunk of the "median" heightmap. 0.5 would be for the true median |
| vtk_view_parts                   | number of parts to show of the rostrum |
| vtk_view_relative_length         | relative length of the above parts |
| | |
| **[Scan]**                        |                                                                             |
| use_latest_base_surface            |                                                                              |
| datetime_of_specific_base_surface  |                                                                              |
| n_directions                       |                                                                              |
| horizontal_resolution              |                                                                              |
| vertical_resolution                |                                                                              |
| max_expected_radius                |                                                                              |
| perfect_fit_minimum_score          |                                                                              |
| gaussian_filter_sigma              |                                                                              |
| inwards_shift_multiplier           |                                                                              |
| radius_multiplier                  |                                                                              |
| endpoints_init_factor              |                                                                              |
| s_initial_value_sigmoid_coeff_1    |                                                                              |
| s_initial_value_sigmoid_coeff_2    |                                                                              |
| s_initial_contrast_sigmoid_coeff   |                                                                              |
| s_initial_contrast_sigmoid_offset  |                                                                              |
| s_initial_proximity_sigmoid_coeff  |                                                                              |
| s_initial_proximity_sigmoid_offset |                                                                              |
| s_initial_value_weight             |                                                                              |
| s_initial_contrast_weight          |                                                                              |
| s_initial_proximity_weight         |                                                                              |
| p_new_val_factor_start             |                                                                              |
| p_new_val_factor_end               |                                                                              |
| p_normal_factor_multiplier         |                                                                              |
| p_ema_diff_adjust_factor           |                                                                              |
| p_current_tooth_radius_weight      |                                                                              |
| p_previous_ema_radius_weight       |                                                                              |
| p_median_tooth_radius_weight       |                                                                              |
| p_expected_radius_weight           |                                                                              |
| c_scan_progress_threshold          |                                                                              |
| c_min_layer_index                  |                                                                              |
| c_offset_threshold                 |                                                                              |
| c_endpoint_prediction_weight       |                                                                              |
| c_expected_radius_weight           |                                                                              |
| c_smoothing_factor_base            |                                                                              |
| c_layer_radius_median_multiplier   |                                                                              |
| c_expected_radius_multiplier       |                                                                              |
| c_correction_factor_at_start       |                                                                              |
| c_correction_factor_at_end         |                                                                              |
| | |
| **[Analysis]**                        |                                                                             |
| use_latest_full_teeth_scan         |                                                                              |
| datetime_of_specific_full_teeth_scan |                                                                              |
| kd_tree_radius_for_heatmap         |                                                                              |
| | |
| **[UI]**                        |                                                                             |
|  use_latest_analysis_pickle      |         |
|  datetime_of_specific_analysis_pickle      |         |
|  use_latest_teeth_polydata      |         |
|  datetime_of_specific_teeth_polydata      |         |
|  use_latest_smoothed_teeth_polydata      |         |
|  datetime_of_specific_smoothed_teeth_polydata      |         |
|  path_ui_settings      |         |
|  min_number_of_color_pickers      |         |
|  relative_volume_slice_thickness      |         |
|  screenshot_resolution      |         |
|   tooth_centers_line_color     |         |
|  tooth_centers_line_width      |         |
|   font_name     |         |
|  font_size_default      |         |
|  relative_graph_widget_width      |         |
|  relative_graph_widget_height      |         |
|   relative_screenshot_icon_size     |         |

