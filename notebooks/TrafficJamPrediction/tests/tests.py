import pandas as pd
def check_duplicate_segments(jam: pd.Series):
    return len(jam['segmentIDs']) != len(set(jam['segmentIDs']))


# def check_duplicate_indices():
# starting_date = datetime(2018, 8, 1, 0, 1, 0)
# for x in range(300):
#     new_date = starting_date + timedelta(minutes=x)
#     print(new_date)
#     df = tjp.io.segmentsdump_get_datetime('Bemobile.Data/SegmentsDump/', new_date)
#     if (df[~df.index.duplicated()].shape != df.shape):
#         print(x)
#         print(df[~df.index.duplicated()].shape)
#         print(df.shape)


# test_segments = test_df.iloc[13]['segmentIDs']
# starting_date_for_segments = datetime.fromtimestamp(float(test_df.iloc[13]['timestamp'])) - timedelta(minutes=29)
# segmentdumps = [tjp.io.segmentsdump_get_datetime('Bemobile.Data/SegmentsDump/', starting_date_for_segments + timedelta(minutes=x)) for x in range(90)]
# meta = tjp.utils.segments_get_meta(basemap, test_segments)