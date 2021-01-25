from collections import deque
from collections import defaultdict
from itertools import count
from datetime import timedelta

from ..basemap import Basemap

import numpy as np
import pandas as pd

def segments_get_predecessor(basemap: Basemap, segment: int) -> np.int64:
    """Get the predecessor segment when arriving at an intersection with multiple options.

    * Requires the Successors entry in the Basemap.
    Args:
        basemap (Basemap): The Basemap with a Successors entry.
        segment (int): The Segment ID of the segment at the intersection.
    Returns:
        np.int64: Returns the SegmentID of the predecessor segment if found, otherwise return 0.
    """    
    basemap.check('successors')
    successor = basemap.map['successors'].loc[basemap.map['successors']['arrivingSegmentID'] == segment, ['departingSegmentID']]
    if successor.shape[0] == 0:
        return 0
    else:
        return successor.squeeze()

def segments_get_successor(basemap: Basemap, segment: int) -> np.int64:
    """Get the successor segment when arriving at an intersection with multiple options.

    * Requires the Successors entry in the Basemap.
    Args:
        basemap (Basemap): The Basemap with a Successors entry.
        segment (int): The Segment ID of the segment at the intersection.
    Returns:
        np.int64: Returns the SegmentID of the successor segment if found, otherwise return 0.
    """
    basemap.check('successors')
    predecessor =  basemap.map['successors'].loc[basemap.map['successors']['departingSegmentID'] == segment, ['arrivingSegmentID']]
    if predecessor.shape[0] == 0:
        return 0
    else:
        return predecessor.squeeze()

def segments_get_meta(basemap: Basemap, segments: list) -> tuple:
    """Get metadata for a given list of segments

    Args:
        basemap (Basemap): The Basemap with a Segments entry
        segments (list): A list of segments that you want the metadata for.

    Returns:
        tuple: A tuple containing 3 items: 
            - list: The Optimal Travel Time in Milliseconds for each of the segments
            - list: The Length in Millimeters for each of the segments
            - list: The Optimal Travel Speed in km/h for each of the segments
    """    
    basemap.check('segments')
    optimal_traveltime_ms, length_mm = basemap.map['segments'].loc[segments][['OptimalTTMs', 'LengthMm']].values.T.tolist() # Transpose so we can access all OptimalTTMs as the first-element
    return (optimal_traveltime_ms, length_mm, list(np.divide(length_mm, optimal_traveltime_ms) * 3.6)) # Calculation of TravelSpeeds


def links_find(basemap: Basemap, segments: list) -> list:
    """Return all the links corresponding to a list of segments.

    Args:
        basemap (Basemap): The Basemap with a Segments entry
        segments (list): A list of segments that you want the corresponding links for.

    Returns:
        list: A list of the links corresponding to the segments.
    """    
    basemap.check('segments')
    return basemap.map['segments'].loc[segments]['LinkID'].values.tolist()

def links_unique(link_ids: list = None, basemap: Basemap = None, segments: list = None) -> list:
    """Convert link_ids in a list of links OR in a list of segments to a digit corresponding to their occurence.
    E.g.: [5136, 5136, 94685, 94685, 3467, 94685] -> [0, 0, 1, 1, 2, 1]

    Args:
        link_ids (list, optional): [description]. Required if segments = None.
        basemap (Basemap, optional): The Basemap containing a Segments entry. Required if link_ids = None
        segments (list, optional): [description]. Required if link_ids = None

    Raises:
        TypeError: [description]
        TypeError: [description]

    Returns:
        [type]: [description]
    """    
    if link_ids is not None:
        link_ids = link_ids
    elif link_ids is None and basemap is not None and segments is not None:
        basemap.check('segments')
        link_ids = links_find(basemap, segments)
    elif link_ids is None and (basemap is None or segments is None):
        raise TypeError("Basemap and segments must be given when link_ids are not given.")
    else:
        raise TypeError("Either link_ids or a combination of a Basemap and Segments should be given.")
    
    
    dd = defaultdict(lambda c=count(): next(c))
    return [dd[el] for el in link_ids]

def junctions_find(basemap, segments):
    
    basemap.check('segments', 'links')

    link_ids = links_find(basemap, segments)
    junctions_in_jam = len(set(link_ids)) - 1

    distance_before = basemap.map['segments'].loc[segments[0]]['CumulativeDistanceMm'] - basemap.map['segments'].loc[segments[0]]['LengthMm']
    distance_after = basemap.map['links'].loc[basemap.map['segments'].loc[segments[-1]]['LinkID']]['LengthMm'] - basemap.map['segments'].loc[segments[-1]]['CumulativeDistanceMm']

    if distance_before < 0:
        distance_before = 0
    if distance_after < 0:
        distance_after = 0

    return [junctions_in_jam, distance_after, distance_before]

def segments_per_kilometer(basemap = None, segments = None, lengths = None, only_indices = False):

    if lengths is None and basemap is None and segments is None:
        raise ValueError("Either both basemap and segments or lengths should be given.")
    elif lengths is None and (basemap is None or segments is None):
        raise ValueError("Either both or neither of basemap and segments should be given.")
    elif lengths is not None:
        lengths = lengths
    else:
        basemap.check('segments')
        lengths = basemap.map['segments'].loc[segments]['LengthMm'].values.tolist()
    
    cumlength = (np.cumsum(lengths) / 1000000).astype(int) # Lengths are in mm, so convert to km: / 1000000
    indices = [np.where(cumlength == x)[0][0] for x in range(0, np.max(cumlength) + 1)] # Get unique element positions

    if only_indices:
        return indices
    else:
        # TODO: Segments should be set, otherwise this gives an error!
        return np.asarray(segments)[[indices]] 


def sincostime(timestamp):
    return [np.cos((timestamp.hour * 60 + timestamp.minute) / 60 / 24 * 2 * 3.141592), np.sin((timestamp.hour * 60 + timestamp.minute) / 60 / 24 * 2 * 3.141592)]

def dataframe_timestamp_to_datetime(dataframe: pd.DataFrame, timestamp_column, offset = 0, **kwargs):
    if (dataframe[timestamp_column].dtype == np.dtype('O')):
        dataframe[timestamp_column] = pd.to_datetime(dataframe[timestamp_column], **kwargs)
        dataframe[timestamp_column] = dataframe[timestamp_column] + timedelta(seconds=offset)
    else:
        dataframe[timestamp_column] = pd.to_datetime(dataframe[timestamp_column] + offset, **kwargs)
    return dataframe

def split_in_batches(files_and_minutes, n):
    i = 0
    _, minutes = list(zip(*files_and_minutes))
    while i < len(minutes):
        last_index = right_index(minutes, minutes[i] + timedelta(minutes=n))
        batch = files_and_minutes[i:last_index]
        i = last_index
        yield batch
    else:
        raise StopIteration

def right_index(haystack, needle):
     while True:
        try:
            return_index = len(haystack) - haystack[::-1].index(needle) - 1
            return return_index
        except ValueError as e: # If the minute is not found, we will try the next minute
            needle += timedelta(minutes=1)
            if needle > haystack[-1]: # We should not try to find minutes after the last one, because that would be stupid.
                break
            continue
        except Exception as e: # If there is some other error, break!
            logging.error(e)
            break