from .expand_jam import append, prepend

import numpy as np
import pandas as pd

import logging
import cv2


def jam_resample_travel_times(travel_times, segments_size=60, minutes_size=90):
    return cv2.resize(travel_times, dsize=(minutes_size, segments_size))

def jam_extract_actual_travel_times(segments, minutes: int, first_minute: int, segmentdumps: pd.DataFrame, total_optimal_traveltimes: pd.Series, optimal_jam_travel_times, lengths):
    """
        args:
            - basemap (Basemap) - The Basemap to use
            - segments (iterable) - Segments of the jam
            - minutes (int) - Amount of minutes to capture, starting from first_minute
            - first_minute (int) - The starting point of our X amount of minutes to capture
            - 
        returns:
            Tuple (travel_times: np.ndarray, travel_speeds: np.ndarray):
                    Shape: ((X, Y), (X, Y))
                        X: Segments
                        Y: Minutes in time
    """

    all_travel_times = []
    all_travel_speeds = []

    for x in range(minutes):
        df = segmentdumps[first_minute + x]
        travel_times = df[~df.index.duplicated()].reindex(segments)['travelTime'].fillna(value=total_optimal_traveltimes).values.astype(int)
        travel_speeds = list(np.divide(lengths, travel_times) * 3.6)
        all_travel_speeds.append(travel_speeds)
        all_travel_times.append(travel_times)

    return (np.asarray(all_travel_times).T.tolist(), np.asarray(all_travel_speeds).T.tolist())
