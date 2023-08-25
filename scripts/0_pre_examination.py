import time
import logging
import numpy as np
import h5py

import helper
from config_and_logging import CONFIG, DEBUG_MODE, USE_FULL_VOLUME_LENGTH, CUSTOM_VOLUME_LENGTH, CUSTOM_VOLUME_START

# Read constants from the config.ini
OTSU_THRESHOLD = CONFIG.getint('Pre-Examination', 'fixed_otsu_threshold')
USE_FIXED_OTSU_THRESHOLD = CONFIG.getboolean('Pre-Examination', 'use_fixed_otsu_threshold')
FIXED_OTSU_THRESHOLD = CONFIG.getint('Pre-Examination', 'fixed_otsu_threshold')
VOLUME_TIP_SIDE_RELATIVE_LENGTH = CONFIG.getfloat('Pre-Examination', 'volume_tip_side_relative_length')
VOLUME_BACK_SIDE_RELATIVE_LENGTH = CONFIG.getfloat('Pre-Examination', 'volume_back_side_relative_length')
VOLUME_ENDS_RELATIVE_LENGTH = max(VOLUME_TIP_SIDE_RELATIVE_LENGTH, VOLUME_BACK_SIDE_RELATIVE_LENGTH)


def run_pre_examination() -> None:
    """Perform the pre-examination of the rostrum.

    This function loads a volume from an hdf5 file, determines the rostrum direction,
    calculates the Otsu threshold, finds the tip position, and saves the results to a pickle file.
    """

    # Load a volume from hdf5 file and read metadata
    with h5py.File(helper.get_path_of_existing_file_from_config('volume', CONFIG), "r") as f:
        volume_dset = f["volume"]
        meta_grp = f["meta"]
        length = int(meta_grp.attrs["length"])
        data_type = meta_grp.attrs["data type"]
        start = 0
        # start = 0 if USE_FULL_VOLUME_LENGTH else CUSTOM_VOLUME_START
        # you should always use the full length to the tip in order to avoid bugs (only in pre-examination)
        stop = length
        # stop = length if USE_FULL_VOLUME_LENGTH else (CUSTOM_VOLUME_LENGTH + start)

        # Extract the front and back sections of the volume
        volume_front_stop = int(length * VOLUME_ENDS_RELATIVE_LENGTH)
        volume_front: np.ndarray = np.array(volume_dset[start:volume_front_stop])
        volume_back_start = int(length * (1 - VOLUME_ENDS_RELATIVE_LENGTH))
        volume_back: np.ndarray = np.array(volume_dset[volume_back_start:stop])

        logging.info(f"volume {f.filename} is loaded")
        logging.debug(f"stop: {stop}")
        logging.debug(f"volume_front.shape: {volume_front.shape}")
        logging.debug(f"volume_back.shape: {volume_back.shape}")

    # Determine which side the rostrum is pointed to based on the mean values of the front and back sections
    rostrum_is_pointed_to_front = volume_back.mean() > volume_front.mean()
    logging.info(f"The rostrum is pointed to the {'front' if rostrum_is_pointed_to_front else 'back'}")

    # Assign the volume sections based on the rostrum direction
    volume_tip_side = volume_front if rostrum_is_pointed_to_front else volume_back
    volume_back_side = volume_back if rostrum_is_pointed_to_front else volume_front

    # Take a relative length of the volume sections, which is enough for the pre-examination
    volume_tip_side = volume_tip_side[0:int(length * VOLUME_TIP_SIDE_RELATIVE_LENGTH)]
    volume_back_side = volume_back_side[0:int(length * VOLUME_BACK_SIDE_RELATIVE_LENGTH)]

    # Plot the histogram  # todo maybe use for BA
    # Histogram
    # Compute the histogram of the volume values
    # hist, bin_edges = np.histogram(volume_back_side, bins=100)
    # import matplotlib.pyplot as plt
    # plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges))
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    # plt.show()

    # Calculate the maximum value for the given data type
    max_val = np.iinfo(data_type).max
    logging.debug(f"max_val for data type {data_type} is {max_val}")

    # Determine the Otsu threshold for image segmentation and further calculations in the following scripts
    if USE_FIXED_OTSU_THRESHOLD:
        otsu_threshold = FIXED_OTSU_THRESHOLD
        logging.info(f"fixed otsu_threshold: {otsu_threshold}")
    else:
        import cv2
        otsu_threshold, _ = cv2.threshold(volume_back_side, 0, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logging.info(f"calculated otsu_threshold: {otsu_threshold}")

    # Calculate percentiles of the volume back side values for further calculations in the following scripts
    all_values = volume_back_side.ravel()
    all_values.sort()
    percentile_25 = all_values[int(len(all_values) * 0.25)]
    percentile_50 = all_values[int(len(all_values) * 0.5)]
    percentile_75 = all_values[int(len(all_values) * 0.75)]
    logging.info(f"percentile_25: {percentile_25}")
    logging.info(f"percentile_50: {percentile_50}")
    logging.info(f"percentile_75: {percentile_75}")

    # Find out the tip position (x, y) of the rostrum
    # This is done by finding the center of mass of the volume at each slice of the volume tip side and taking the mean
    centers_of_mass = []

    # Set all values below the Otsu threshold to 0, so they don't influence the center of mass
    volume_tip_side[volume_tip_side < otsu_threshold] = 0

    for i_slice in range(len(volume_tip_side)):
        volume_slice: np.ndarray = volume_tip_side[i_slice, :, :]

        # Calculate the centers of mass
        weight_x = volume_slice.sum(axis=0)
        weight_y = volume_slice.sum(axis=1)
        if weight_x.sum() != 0 and weight_y.sum() != 0:
            x_center = np.average(np.arange(volume_slice.shape[1]), weights=weight_x)
            y_center = np.average(np.arange(volume_slice.shape[0]), weights=weight_y)
            centers_of_mass.append([x_center, y_center])

    # Calculate the mean of the x-values and the y-values
    x_mean, y_mean = np.mean(centers_of_mass, axis=0)
    x_mean = int(x_mean)
    y_mean = int(y_mean)
    logging.info(f"calculated tip center position: ({x_mean}, {y_mean})")

    # Display the volume tip side as an image with the calculated tip center position as a red dot
    helper.show_2d_array_as_image(volume_tip_side.mean(axis=0), (x_mean, y_mean))

    # Save the examination to a pickle file
    examination = {
        'otsu_threshold': otsu_threshold,
        'rostrum_is_pointed_to_front': rostrum_is_pointed_to_front,
        'axis_center_x': x_mean,
        'axis_center_y': y_mean,
        'percentile_25': percentile_25,
        'percentile_50': percentile_50,
        'percentile_75': percentile_75
    }
    logging.debug(f"examination: {examination}")

    import pickle
    with open(helper.create_path_for_new_file_from_config('pre-examination', CONFIG), 'wb') as f:
        pickle.dump(examination, f)
        logging.info(f"saved pre-examination to {f.name}")


if __name__ == '__main__':
    tic = time.perf_counter()
    script_name = __file__.split('\\')[-1]
    logging.info(f'started {script_name}...\n')
    run_pre_examination()
    logging.info(f"finished. {time.perf_counter() - tic:1.1f}s")
    exit(0)
