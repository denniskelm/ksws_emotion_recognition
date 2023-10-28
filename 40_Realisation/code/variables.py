"""
For defining global variables (such as DataFrames that are transferred between modules)
"""
import os

import pandas as pd


def init():
    global df
    df = pd.DataFrame()

    global merged_frame
    merged_frame = pd.DataFrame()

    global time_frame
    time_frame = pd.DataFrame()

    global numbers_of_clusters
    numbers_of_clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    global personIDs
    personIDs = [8, 10, 12, 13, 15, 20, 21, 25, 27, 33, 35, 40, 46, 48, 49, 52, 54, 55]

    global sensor_motion_data
    sensor_motion_data = [
        'raw_acc:magnitude_stats:mean',
        'raw_acc:magnitude_stats:std',
        'raw_acc:magnitude_stats:moment3',
        'raw_acc:magnitude_stats:moment4',
        'raw_acc:magnitude_stats:percentile25',
        'raw_acc:magnitude_stats:percentile50',
        'raw_acc:magnitude_stats:percentile75',
        'raw_acc:magnitude_stats:value_entropy',
        'raw_acc:magnitude_stats:time_entropy',
        'raw_acc:magnitude_spectrum:log_energy_band0',
        'raw_acc:magnitude_spectrum:log_energy_band1',
        'raw_acc:magnitude_spectrum:log_energy_band2',
        'raw_acc:magnitude_spectrum:log_energy_band3',
        'raw_acc:magnitude_spectrum:log_energy_band4',
        'raw_acc:magnitude_spectrum:spectral_entropy',
        'raw_acc:magnitude_autocorrelation:period',
        'raw_acc:magnitude_autocorrelation:normalized_ac',
        'raw_acc:3d:mean_x',
        'raw_acc:3d:mean_y',
        'raw_acc:3d:mean_z',
        'raw_acc:3d:std_x',
        'raw_acc:3d:std_y',
        'raw_acc:3d:std_z',
        'raw_acc:3d:ro_xy',
        'raw_acc:3d:ro_xz',
        'raw_acc:3d:ro_yz',
        'proc_gyro:magnitude_stats:mean',
        'proc_gyro:magnitude_stats:std',
        'proc_gyro:magnitude_stats:moment3',
        'proc_gyro:magnitude_stats:moment4',
        'proc_gyro:magnitude_stats:percentile25',
        'proc_gyro:magnitude_stats:percentile50',
        'proc_gyro:magnitude_stats:percentile75',
        'proc_gyro:magnitude_stats:value_entropy',
        'proc_gyro:magnitude_stats:time_entropy',
        'proc_gyro:magnitude_spectrum:log_energy_band0',
        'proc_gyro:magnitude_spectrum:log_energy_band1',
        'proc_gyro:magnitude_spectrum:log_energy_band2',
        'proc_gyro:magnitude_spectrum:log_energy_band3',
        'proc_gyro:magnitude_spectrum:log_energy_band4',
        'proc_gyro:magnitude_spectrum:spectral_entropy',
        'proc_gyro:magnitude_autocorrelation:period',
        'proc_gyro:magnitude_autocorrelation:normalized_ac',
        'proc_gyro:3d:mean_x',
        'proc_gyro:3d:mean_y',
        'proc_gyro:3d:mean_z',
        'proc_gyro:3d:std_x',
        'proc_gyro:3d:std_y',
        'proc_gyro:3d:std_z',
        'proc_gyro:3d:ro_xy',
        'proc_gyro:3d:ro_xz',
        'proc_gyro:3d:ro_yz',
        'raw_magnet:magnitude_stats:mean',
        'raw_magnet:magnitude_stats:std',
        'raw_magnet:magnitude_stats:moment3',
        'raw_magnet:magnitude_stats:moment4',
        'raw_magnet:magnitude_stats:percentile25',
        'raw_magnet:magnitude_stats:percentile50',
        'raw_magnet:magnitude_stats:percentile75',
        'raw_magnet:magnitude_stats:value_entropy',
        'raw_magnet:magnitude_stats:time_entropy',
        'raw_magnet:magnitude_spectrum:log_energy_band0',
        'raw_magnet:magnitude_spectrum:log_energy_band1',
        'raw_magnet:magnitude_spectrum:log_energy_band2',
        'raw_magnet:magnitude_spectrum:log_energy_band3',
        'raw_magnet:magnitude_spectrum:log_energy_band4',
        'raw_magnet:magnitude_spectrum:spectral_entropy',
        'raw_magnet:magnitude_autocorrelation:period',
        'raw_magnet:magnitude_autocorrelation:normalized_ac',
        'raw_magnet:3d:mean_x',
        'raw_magnet:3d:mean_y',
        'raw_magnet:3d:mean_z',
        'raw_magnet:3d:std_x',
        'raw_magnet:3d:std_y',
        'raw_magnet:3d:std_z',
        'raw_magnet:3d:ro_xy',
        'raw_magnet:3d:ro_xz',
        'raw_magnet:3d:ro_yz',
        'raw_magnet:avr_cosine_similarity_lag_range0',
        'raw_magnet:avr_cosine_similarity_lag_range1',
        'raw_magnet:avr_cosine_similarity_lag_range2',
        'raw_magnet:avr_cosine_similarity_lag_range3',
        'raw_magnet:avr_cosine_similarity_lag_range4',
        'watch_acceleration:magnitude_stats:mean',
        'watch_acceleration:magnitude_stats:std',
        'watch_acceleration:magnitude_stats:moment3',
        'watch_acceleration:magnitude_stats:moment4',
        'watch_acceleration:magnitude_stats:percentile25',
        'watch_acceleration:magnitude_stats:percentile50',
        'watch_acceleration:magnitude_stats:percentile75',
        'watch_acceleration:magnitude_stats:value_entropy',
        'watch_acceleration:magnitude_stats:time_entropy',
        'watch_acceleration:magnitude_spectrum:log_energy_band0',
        'watch_acceleration:magnitude_spectrum:log_energy_band1',
        'watch_acceleration:magnitude_spectrum:log_energy_band2',
        'watch_acceleration:magnitude_spectrum:log_energy_band3',
        'watch_acceleration:magnitude_spectrum:log_energy_band4',
        'watch_acceleration:magnitude_spectrum:spectral_entropy',
        'watch_acceleration:magnitude_autocorrelation:period',
        'watch_acceleration:magnitude_autocorrelation:normalized_ac',
        'watch_acceleration:3d:mean_x',
        'watch_acceleration:3d:mean_y',
        'watch_acceleration:3d:mean_z',
        'watch_acceleration:3d:std_x',
        'watch_acceleration:3d:std_y',
        'watch_acceleration:3d:std_z',
        'watch_acceleration:3d:ro_xy',
        'watch_acceleration:3d:ro_xz',
        'watch_acceleration:3d:ro_yz',
        'watch_acceleration:spectrum:x_log_energy_band0',
        'watch_acceleration:spectrum:x_log_energy_band1',
        'watch_acceleration:spectrum:x_log_energy_band2',
        'watch_acceleration:spectrum:x_log_energy_band3',
        'watch_acceleration:spectrum:x_log_energy_band4',
        'watch_acceleration:spectrum:y_log_energy_band0',
        'watch_acceleration:spectrum:y_log_energy_band1',
        'watch_acceleration:spectrum:y_log_energy_band2',
        'watch_acceleration:spectrum:y_log_energy_band3',
        'watch_acceleration:spectrum:y_log_energy_band4',
        'watch_acceleration:spectrum:z_log_energy_band0',
        'watch_acceleration:spectrum:z_log_energy_band1',
        'watch_acceleration:spectrum:z_log_energy_band2',
        'watch_acceleration:spectrum:z_log_energy_band3',
        'watch_acceleration:spectrum:z_log_energy_band4',
        'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range0',
        'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range1',
        'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range2',
        'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range3',
        'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range4',
        'watch_heading:mean_cos',
        'watch_heading:std_cos',
        'watch_heading:mom3_cos',
        'watch_heading:mom4_cos',
        'watch_heading:mean_sin',
        'watch_heading:std_sin',
        'watch_heading:mom3_sin',
        'watch_heading:mom4_sin',
        'watch_heading:entropy_8bins']


def getSavePath(directory, path):
    """
    directory: data or viz
    path: path for the file to save
    """
    if directory == "viz":
        directory = "visualizations"

    dirname = os.path.dirname
    save_path = os.path.join(dirname(dirname(__file__)),
                             os.path.join(directory, path))
    return save_path
