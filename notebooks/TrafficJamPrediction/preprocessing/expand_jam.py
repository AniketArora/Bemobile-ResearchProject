from ..utils import segments_get_predecessor, segments_get_successor
import logging
format = "%(asctime)s: [ %(thread)d ] %(message)s"
logging.basicConfig(format=format, level=logging.DEBUG, datefmt="%H:%M:%S")
def prepend(basemap, segments, meters_to_append):
    """
        Prepend X amount of meters to the front of the TrafficJam.
    """
    basemap.check('segments', 'successors')
    segments_to_add = []
    mms_added = 0
    stop = False
    current_segment = segments[0]
    
    ## We can try to find other segments in the same link
    linkID, sequence_number = basemap.map['segments'].loc[current_segment][['LinkID','SegmentSequenceNumber']]
    ## Maybe there are precedingSegments
    preceding_segments = basemap.map['segments'].loc[(basemap.map['segments']['LinkID'] == linkID) & (basemap.map['segments']['SegmentSequenceNumber'] < sequence_number)].copy()


    while (mms_added <= (meters_to_append * 1000)) and stop == False:
        if len(preceding_segments) == 0: #If there are none in this link, we need to search in another link
            predecessor = segments_get_predecessor(basemap, current_segment)
            

            ## What if predecessor is 0?
            if predecessor == 0:
                # logging.warning(f"We could not find a predecessor for segment {current_segment}, we have to find it in another way.")
                # logging.warning("We cannot determine the previous link or segment. We are sorry. \n There are either too many possible options or even none.")
                stop = True
                break
            else:
                pass
                # logging.info("Predecessor found!")
                # logging.info(predecessor)

            # We can set the new preceding segments to the Link corresponding to the predecessor segment we found.
            preceding_segments = basemap.map['segments'].loc[basemap.map['segments']['LinkID'] == basemap.map['segments'].loc[predecessor]['LinkID']].copy()
            # logging.info("Preceding segments are")
            # logging.info(preceding_segments[['LinkID', 'SegmentSequenceNumber', 'LengthMm']])

        else:
            preceding_segments['CumulativeLengthMm'] = preceding_segments['LengthMm'].cumsum()
            # Only get the values that have a CumulativeLength less than what we still have to append
            preceding_segments_to_add = preceding_segments[preceding_segments['CumulativeLengthMm'] + mms_added <= (meters_to_append * 1000)][::-1]

            # logging.info(f"Found {preceding_segments_to_add.index.values} but not all will be added, some are already in the list: Officially adding:")
            segments_not_yet_added = [segment for segment in preceding_segments_to_add.index.values if (segment not in segments) and (segment not in segments_to_add)]
            segments_to_add.extend(segments_not_yet_added)
            # logging.info(segments_not_yet_added)
            ## Reset the preceding segments for the next loop.
            preceding_segments = []

            if len(segments_not_yet_added) == 0:
                # logging.info("As we did not add at least one segment, we might be at a roundabout.")
                stop = True
                break


            ## Make sure the current segment is now the last one we just added
            current_segment = segments_to_add[-1]


        if len(segments_to_add) > 40:
            # logging.info("Stopping because we added 40 segments, this should be more than enough.")
            stop = True
        else:
            pass
            # logging.info(f"Currently at {len(segments_to_add)} segments")
    return segments_to_add

def append(basemap, segments, meters_to_append):
    """
        Append X amount of meters to the end of a TrafficJam.
    """
    basemap.check('segments', 'successors')
    segments_to_add = []
    mms_added = 0
    stop = False
    current_segment = segments[-1]
    
    ## We can try to find other segments in the same link
    linkID, sequence_number = basemap.map['segments'].loc[current_segment][['LinkID','SegmentSequenceNumber']]
    ## Maybe there are following_segments
    following_segments = basemap.map['segments'].loc[(basemap.map['segments']['LinkID'] == linkID) & (basemap.map['segments']['SegmentSequenceNumber'] > sequence_number)].copy()


    while (mms_added <= (meters_to_append * 1000)) and stop == False:
        
        if len(following_segments) == 0: #If there are none in this link, we need to search in another link
            successor = segments_get_successor(basemap, current_segment)
            

            ## What if successor is 0?
            if successor == 0:
                # logging.warning(f"We could not find a successor for segment {current_segment}, we have to find it in another way.")
                # logging.warning("We cannot determine the next link or segment. We are sorry. \n There are either too many possible options or even none.")
                stop = True
                break
            else:
                pass
                # logging.info("Successor found!")
                # logging.info(successor)

            # We can set the new following segments to the Link corresponding to the following segment we found.
            following_segments = basemap.map['segments'].loc[basemap.map['segments']['LinkID'] == basemap.map['segments'].loc[successor]['LinkID']].copy()
            # logging.info("Following segments are")
            # logging.info(following_segments[['LinkID', 'SegmentSequenceNumber', 'LengthMm']])

        else:
            following_segments['CumulativeLengthMm'] = following_segments['LengthMm'].cumsum()
            # Only get the values that have a CumulativeLength less than what we still have to append
            following_segments_to_add = following_segments[following_segments['CumulativeLengthMm'] + mms_added <= (meters_to_append * 1000)]

            # logging.info(f"Found {following_segments_to_add.index.values} but not all will be added, some are already in the list: Officially adding:")
            segments_not_yet_added = [segment for segment in following_segments_to_add.index.values if (segment not in segments) and (segment not in segments_to_add)]
            segments_to_add.extend(segments_not_yet_added)
            # logging.info(segments_not_yet_added)
            ## Reset the preceding segments for the next loop.
            following_segments = []

            if len(segments_not_yet_added) == 0:
                # logging.info("As we did not add at least one segment, we might be at a roundabout.")
                stop = True
                break


            ## Make sure the current segment is now the last one we just added
            current_segment = segments_to_add[-1]


        if len(segments_to_add) > 40:
            # logging.info("Stopping")
            stop = True
        else:
            pass
            # logging.info(f"Currently at {len(segments_to_add)} segments")
    return segments_to_add