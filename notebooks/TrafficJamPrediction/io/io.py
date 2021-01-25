import itertools
import json
import logging
import os
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from tempfile import mkdtemp
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class IO(object):

    def __init__(self, directory):
        self.directory = {}


## Segmentsdump
def segmentsdump_count(directory: str, days_range: Iterable, patterns: Iterable) -> list:
    """
    Print how many SegmentsDump files there are in a directory, split by days.
    In one day, there should be 1440 files, each for one minute.

    Args:
        directory (str): The directory where the SegmentsDump files are located.
        days_range (Iterable): The date range to search for. E.g.: range(1, 5) to get days from 01 to 05. Days are left-padded to 2-digits.
        patterns (Iterable): A list of 2 patterns, one before the day and one behind it. E.g.: 201808{day}*.gz --> ['201808', '*.gz']

    Returns:
        list: List containing the count of files for each day in the days_range.
    """
    test = [len([f.split('/')[-1][8:12] for f in sorted(glob(f"{directory}/{patterns[0]}{day:02}{patterns[1]}"))]) for day in days_range]
    return test

def segmentsdump_get_multiple_with_timestamps(directory: str, dump_datetimes: Iterable, timestamps = None):
    #TODO: Use multiprocessing / joblib!
    
    dumps = []
    for dump_datetime in dump_datetimes:
        append = True
        if timestamps:
            if dump_datetime.timestamp() in timestamps:
                logging.debug(f"{dump_datetime.timestamp()} already in deque.")
                append = False
        if append: dumps.append([segmentsdump_get_datetime(directory, dump_datetime), dump_datetime.timestamp()])
    return dumps
    

def segmentsdump_get_datetime(directory: str, dump_datetime: datetime) -> pd.DataFrame:
    """Open one SegmentsDump file based on the datetime.

    Args:
        directory (str): The directory where the SegmentsDump files are located.
        dump_datetime (datetime): The datetime of the SegmentsDump file to open.

    Returns:
        pd.DataFrame: The opened SegmentsDump file in a Pandas Dataframe.
    """    
    filename = datetime.strftime(dump_datetime, '%Y%m%d%H%M_segmentsdump.csv.gz')
    return segmentsdump_get(f"{directory}/{filename}")
    
def segmentsdump_get(filename: str) -> pd.DataFrame:
    """Open one SegmentsDump file

    Args:
        filename (str): Absolute location to the SegmentsDump file to open

    Returns:
        pd.DataFrame: The opened SegmentsDump file in a Pandas Dataframe.
    """    
    return pd.read_csv(filename, compression='gzip', delimiter=';', usecols=[1, 2], skiprows=1, header=0, index_col=0, names=['segmentID', 'travelTime'])

## TrafficEvents

def trafficevent_get_all(directory, days_range, patterns):
    return list(itertools.chain(*[
        sorted(glob(f"{directory}/{patterns[0]}{day:02}{patterns[1]}", recursive=True)) for day in days_range
    ]))

def trafficevent_get_multiple(directory, patterns):
    return  sorted(glob(f"{directory}/{patterns[0]}{patterns[1]}", recursive=True))

def trafficevent_get_unfinished(directory, finished_directory, days_range, patterns):
    """
        Function to read all TrafficEvent jsons in a directory and compare to a finished directory
    """
    all_files = trafficevent_get_all(directory, days_range, patterns)
    finished_files = sorted(glob(finished_directory + '/*.json'))
    
    unfinished_files = sorted([
        f"{directory}/{f}" # Append filename to directory
        for f in set( # Get only unique filenames
                    [f.split('/')[-1] for f in all_files]
                ).difference( # Compare to finished files
                set( # Get only unique finished filenames
                    [f.split('/')[-1] for f in finished_files]
                ))
    ])

    return unfinished_files

def trafficevent_get(filename, road_class = 3):
    """
        Function to read a TrafficEvent json 
    """
    jams = []
    with open(filename) as f:
        for line in f.readlines():
            message = json.loads(line)
            jams.extend([msg
                for msg in message
                if (msg['what']['category'] == 'trafficjam') and (msg['where']['basemap'][0]['roadClass'] < road_class)
             ])
    return jams

def trafficevent_from_datetime_minute(directory, event_datetime):
    filename = datetime.strftime(event_datetime, 'traffic-events-%Y%m%d%H%M')
    return trafficevent_get_multiple(f"{directory}", [filename, '*.json'])


def trafficevent_from_datetime(directory, event_datetime):
    filename = datetime.strftime(event_datetime, 'traffic-events-%Y%m%d%H%M%S.json')
    return trafficevent_get(f"{directory}/{filename}")

def trafficevent_to_datetime(filename):
    event_datetime = datetime.strptime(filename.split('/')[-1], 'traffic-events-%Y%m%d%H%M%S.json')
    return event_datetime - timedelta(seconds=event_datetime.second)

def trafficevent_finish(finished_directory, filename):
    """
        Function to log a TrafficEvent json as finished.
    """
    from pathlib import Path
    file = Path(f"{finished_directory}/{filename.split('/')[-1]}")
    file.touch(exist_ok=True)
    st = os.stat(file)
    return st



## Dataframes / Pickles

def dataframes_read_pickles(directory, limit = None):
    dataframe = []
    files = glob(f"{directory}/*.pkl")[:limit]
    logging.info(files)
    for count, pkl_file in enumerate(files):
        pkl = pd.read_pickle(pkl_file)
        logging.info(f"Finished reading pickle {count}")
        dataframe.append(pkl)
    return pd.concat(dataframe)

def dataframes_read_parquets(directory, bounds = None):
    filenames = sorted(glob(f"{directory}/**/*"))
    if bounds is not None and len(bounds) > 0:
        filenames = sorted(list(filter(lambda filename: bounds[0] < datetime.fromtimestamp(float(filename.split('=')[-1])) < bounds[1], filenames)))
        
    dataframe = []
    for filename in filenames:
        df = pq.read_table(filename).to_pandas()
        df['timestamp'] = datetime.fromtimestamp(float(filename.split('=')[-1]))
        dataframe.append(df)

    return pd.concat(dataframe, ignore_index=True)

## Mmapping

def mmap_save(dataframe, columns, filename):
    dataframe = dataframe[columns]
    df_columns = dataframe.reset_index().columns
    df_values = dataframe.reset_index().to_numpy()

    mmap_file = Path(mkdtemp(), filename)

    mmap = np.memmap(mmap_file, dtype='int64', mode='w+', shape=df_values.shape)
    mmap[:] = df_values[:]

    return mmap_file

def mmap_load(mmap_file, columns):
    mmap_data = np.memmap(mmap_file, dtype='int64', mode='r')
    mmap_data = mmap_data.reshape(int(mmap_data.shape[0] / len(columns)), len(columns))
    dataframe = pd.DataFrame(mmap_data, columns=columns)
    return dataframe.set_index(columns[0])
