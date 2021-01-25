import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json

import TrafficJamPrediction as tjp


df = pd.read_pickle('../data/test-dataframe.pkl')
df_plot = df[:1000]

with open('../api/app/test/train_item.npy', 'rb') as f:
    dataset = np.load(BytesIO(f.read()))

dataset = dataset[:1000]

# API call voorspelling
response = requests.post('http://api.172.23.82.60.xip.io:8527/predict/', data=json.dumps({"Array": json.dumps(dataset.tolist()) }))
if not response.ok:
    raise Exception('Failed to get predictions')

parsed_response = response.json()
predictions_cnn = np.array(parsed_response['predictions'])

def reshape_to_original(predictions, segments):
    predictions_unflattened = np.reshape(predictions, (60, 60) )
    predictions_original_shape = tjp.preprocessing.jam_resample_travel_times(np.asarray(predictions_unflattened), segments_size=segments, minutes_size=60)
    return predictions_original_shape

df_plot['predictedTravelSpeeds'] = pd.Series([reshape_to_original(predictions_cnn[i], len(df_plot.iloc[i]['segmentIDs'])) for i in range(len(predictions_cnn)) ])

df_plot['actualTravelSpeeds.Colors'] = tjp.plotting.dataframe_normalize_speed_to_colors(df_plot, 'optimalTravelSpeeds', 'actualTravelSpeeds')

df_plot['predictedTravelSpeeds.Colors'] = tjp.plotting.dataframe_normalize_speed_to_colors(df_plot, 'optimalTravelSpeeds', 'predictedTravelSpeeds')

basemap = tjp.Basemap({'_directory': '../data/basemap'})
basemap.load({
        'segments': {'index': 'SegmentID'},
        'nodes': {'index': 'NodeID'},
})
basemap.check('segments', 'nodes')
df_plot['coordinates'] = df_plot.apply(lambda row: tjp.plotting.jam_get_coordinates(basemap, row['segmentIDs']) , axis=1)

# Create plots

def create_plot(plotting, dataframe, idx):
    event = dataframe.iloc[idx]
    (lons, lats) = list(zip(*event['coordinates']))
    
    plotting.plotly_subplots(
        rows=3, cols=2,
        subplot_titles=("Comparison of speeds", "Meta data of event " + str(event['eventID']) +" at timestamp " + str(event['timestamp']), "Error of prediction versus actual speed"),
        row_heights=[100, 100, 100],
        column_widths=[0.7, 0.3],
        specs=[[{"type": "scatter"}, {"type": "table", "rowspan": 2}],
               [{"type": "scatter"}, None],
               [{"type": "mapbox", "colspan": 2}, None]
              ],
    )
    
    # Scatter plots
    
    scatter_layers = [
        dict(trace_type='Scatter',x=event['segmentIDs'], y=event['actualTravelSpeeds'][:,30:].T, name='Actual', row=1, col=1),
        dict(trace_type='Scatter',x=event['segmentIDs'], y=event['predictedTravelSpeeds'].T, name='Predicted', row=1, col=1, fillcolor="red"),
        dict(trace_type='Scatter',x=event['segmentIDs'], y=np.asarray(event['optimalTravelSpeeds']).astype(int), name='Optimal', row=1, col=1, fillcolor="green", line=dict(color='green', width=2, dash='dash')),
        dict(trace_type='Scatter',x=event['segmentIDs'], y=event['predictedTravelSpeeds'].T - event['actualTravelSpeeds'][:,30:].T, name='Error', row=2, col=1, fillcolor="purple"),
    ]
    

    for layer in scatter_layers:
        plotting.plotly(**layer)
    
    
    plotting.figure.update_xaxes( 
        nticks=20,
        title_text='Segments',
        type='category',
    )
    plotting.figure.update_yaxes(
        title_text='Speed in km/h',
    )
    
    
    
    for km in tjp.utils.segments_per_kilometer(lengths = event['lengths'], only_indices=True)[1:]:
        plotting.figure.update_layout(
                # Line Vertical
                shapes=[dict(
                    type="line",
                    xref='x', yref='y',
                    x0=event['segmentIDs'][km], x1=event['segmentIDs'][km],
                    y0=0, y1=125,
                    line=dict(
                        color="black",
                        width=1,
                    )
                )]
        )
    
    # Tables
    plotting.plotly_table(
        header = {
            'values': ['<b>Field</b>', '<b>Values</b>'],
        },
        cells = {
            'values': [
                ['eventID', 'timestamp', 'roadClass', 'dayOfWeek', 'weekend', 'Segments In Jam', 'Length of jam in mms', 'Junctions in Jam', 'Distance to previous junction', 'Distance to next junction'],
                [*event[['eventID', 'timestamp', 'roadClass', 'dayOfWeek', 'weekend']], len(event['segmentIDs']), np.sum(event['lengths']), *event['junctions']]
            ],
        },
        row = 1,
        col = 2
    )

    
    ## Mapbox preperations
    travel_speeds = event['actualTravelSpeeds'].T[30:]
    colors = event['actualTravelSpeeds.Colors'].T[30:] - 0.01 # We remove 0.01 because we mapped 1 for White. Black is 0.99. Plotly does not allow for color scales beyond [0,1]
    prediction_travel_speeds = event['predictedTravelSpeeds'].T
    prediction_colors = event['predictedTravelSpeeds.Colors'].T - 0.01
    layers = {
        'speed': [travel_speeds, np.ones(travel_speeds.shape), prediction_travel_speeds],
        'colors': [colors, np.ones(colors.shape), prediction_colors],
        'hover_texts': ["Actual Speed: %{text:.3f} km/h", "Border", "Prediction Speed: %{text:.3f} km/h"],
        'show_legend': [True, False, False]
    }
    
    ## PLotly Mapbox
    mapbox_frames = plotting.plotly_mapbox_frames(layers, f"Traffic jam {event['eventID']}", idx, event['coordinates'])
    plotting.plotly_mapbox(mapbox_frames, row=3, col=1)
    plotting.plotly_mapbox_init(mapbox={'domain': {'x': [0, 1], 'y': [0, 0.25]}, 'center': go.layout.mapbox.Center(lat=np.median(lats), lon=np.median(lons))})
    
    ## Animations
    layout_kwargs = [
        { 'title_text': f"Speed predictions of event {event['eventID']} at timestamp {str(event['timestamp'] + timedelta(minutes=i + 1))} (minute {str(i + 1)} )"}
    for i in range(0, 60)]
    
    scatter_frames = plotting.plotly_scatter_frames([scatter_layers[i] for i in [0, 1, 3]], 60)
    frames = np.concatenate((scatter_frames, np.asarray(mapbox_frames)[:, [0, 2]]), axis=1)
    
    plotting.plotly_add_frames(frames, [0, 1, 3, 5, 7], layout_kwargs)
    plotting.plotly_add_buttons()
    
    plotting.figure.update_layout(height=1000, hovermode='x unified')
    
    return plotting.figure

mapbox_access_token = open(".mapbox-token").read()
plotting = tjp.plotting.Plotting(mapbox_access_token)

create_plot(plotting, df_plot, 0)
