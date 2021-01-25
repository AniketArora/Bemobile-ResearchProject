import logging
from datetime import timedelta

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash
from plotly.subplots import make_subplots

from ..utils import links_unique


class Plotting(object):

    SIZES = {
        'Extra_Small': 15,
        'Small': 20,
        'Medium': 35,
        'Large': 40,
        'Extra_Large': 55,
    }

    def __resettable(f):
        def helper(self, *args, **kwargs):
            if 'reset' in kwargs and kwargs['reset']:
                self.reset()
            return f(self, *args, **kwargs)
        return helper

    def __subplottable(f):
        def helper_subplot(self, *args, **kwargs):
            if self.subplots:                    
                kwargs.update({ 'plot': {k: v for k, v in kwargs.items() if k in ['row', 'col']}})
            else:
                kwargs = {k: v for k, v in kwargs.items() if k not in ['row', 'col']} # Removing Row, Col if not subplotting
                kwargs.update({ 'plot': {}})
            return f(self, *args, **kwargs)
        return helper_subplot


    def __init__(self, token = None, *args, **kwargs):
        self.token = token
        self.reset()

    @property
    def figure(self):
        return self.__figure

    @figure.setter
    def figure(self, value):
        self.__figure = value

    def reset(self):
        self.figure = go.Figure()
        self.subplots = False

    def show(self, layout = None, xaxes = None, yaxes = None, *args, **kwargs):
        if layout is not None:
            self.figure.update_layout(**layout)
        if xaxes is not None:
            self.figure.update_xaxes(**xaxes)
        if yaxes is not None:
            self.figure.update_yaxes(**yaxes)
        self.figure.show(*args, **kwargs)

    def plotly_subplots(self, **kwargs):
        self.subplots = True
        self.figure = make_subplots(**kwargs)

    def plotly_add_buttons(self, *args):
        """
            *args contains **kwargs that represent the different items in the buttons list
        """
        buttons = []

        # Default play and pause button. You can easily change these things.
        default_button = {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 150, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 200,
                                                                            "easing": "cubic-in-out"}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }

        buttons.append(default_button)
        
        for idx, button in enumerate(args):
            buttons[idx].update(**button)
        self.figure.update_layout(updatemenus=buttons)

    def plotly_mapbox_frames(self, layers, name, row_idx, coordinates=None, basemap=None, segments=None, *args, **kwargs):
        """
            Function to plot multiple layers of a jam, with each layer containing different colors and text.
            The jam segments will stay the same.
        """

        if coordinates is None and basemap is None and segments is None:
            raise ValueError(
                "Either both basemap and segments or coordinates should be given.")
        elif coordinates is None and (basemap is None or segments is None):
            raise ValueError(
                "Either both or neither of basemap and segments should be given.")
        elif coordinates is not None:
            coordinates = coordinates
        else:
            coordinates = jam_get_coordinates(basemap, segments)

        (lons, lats) = list(zip(*coordinates))


        frames = [[go.Scattermapbox(
                        lat=lats,
                        lon=lons,
                        text=layers['speed'][x][i],
                        mode='markers',
                        marker=go.scattermapbox.Marker(
                            size=np.ones(len(coordinates)) * sorted(self.SIZES.values())[x] ,
                            color=layers['colors'][x][i],
                            colorscale = [[0, 'green'], [0.5, 'red'], [0.99, 'black'], [1, 'white']],
                            cmax=1,
                            cmin=0,
                        ),
                        hovertemplate=layers['hover_texts'][x],
                        name=f"{name}",
                        subplot='mapbox',
                        legendgroup=row_idx,
                        showlegend=layers['show_legend'][x],
                    ) for x in (2, 1, 0)] for i in range(60)]

        return frames

    def plotly_scatter_frames(self, traces, minutes):
        return [
            [
                go.Scatter(x=trace['x'], y=trace['y'][i])
                for trace in traces
            ] for i in range(0, minutes)
        ]

    def plotly_add_frames(self, traces, trace_indices, layout_kwargs_per_frame = {}, *args, **kwargs):
        """
            Function to add frames to the figure.
            
            args:
                - traces: dict containing at least X and Y values to plot.
                    The Y values are the ones being animated. This is a numpy array where each iteration contains one frame to plot.
                - trace_indices: list containing the indices of the Plotly frames corresponding to the traces

            # TODO: Add a check to make sure that the animated traces all have the same shape
            # TODO: Add a dynamic minutes parameter
        """

        layouts = [
            go.Layout(
                **layout_kwargs
            ) for layout_kwargs in layout_kwargs_per_frame
        ]

        if traces.shape[-1] != len(trace_indices):
            raise ValueError("The amount of trace indices does not correspond to the amount of traces to animate.")
        
        
        self.figure.frames = [
            go.Frame(
                data=[
                    *traces[i],
                ],
                layout=layouts[i],
                traces=[*trace_indices]
            )
            for i in range(traces.shape[0])
        ]
    
    @__resettable
    @__subplottable
    def plotly_table(self, header, cells, *args, **kwargs):

        plot_kwargs = kwargs['plot'] if 'plot' in kwargs else {}
        if 'values' not in header:
            raise ValueError("We need to set values for the Table Header.")
        if 'values' not in cells:
            raise ValueError("We need to set values for the Table Body.")
    
        
        header_kwargs = {
            'align': 'left',
            'fill_color': 'paleturquoise',
        }

        header_kwargs.update(header)

        cells_kwargs = {
            'align': 'left',
            'fill_color': 'lavender',
            'height': 30,
        }
        cells_kwargs.update(cells)

        self.figure.add_trace(
            go.Table(
                header = header_kwargs,
                cells = cells_kwargs,
            ),
            **plot_kwargs
        )

    @__subplottable
    @__resettable
    def plotly(self, x, y = None, name = "", trace_type = 'Histogram', *args, **kwargs):
        
        plot_kwargs = kwargs['plot'] if 'plot' in kwargs else {}
        kwargs = {k: v for k, v in kwargs.items() if k not in ['row', 'col', 'plot']}
        if trace_type == 'Histogram':
            trace_kwargs = kwargs
            if y is not None:
                trace = go.Histogram(x=x, y=y, name=name, **trace_kwargs)
            else:
                trace = go.Histogram(x=x, name=name, **trace_kwargs)

    
        elif trace_type == 'Scatter':
            trace_kwargs = {'mode': 'lines'}
            trace_kwargs.update(kwargs)

            if (len(np.array(y).shape) > 1):
                y = y[0]
            
            trace = go.Scatter(x=x, y=y, name=name, **trace_kwargs)   
                     

        self.figure.add_trace(
            trace,
            **plot_kwargs
        )

    @__resettable
    @__subplottable
    def plotly_mapbox(self, frames, **kwargs):
       
        plot_kwargs = kwargs['plot'] if 'plot' in kwargs else {}
        
        for idx in range(np.asarray(frames).shape[1]):
            self.figure.add_trace(
               frames[0][idx],
               **plot_kwargs
            )

    def plotly_mapbox_init(self, layout={}, mapbox={}):
        """
            Wrapper around Plotly figures, can be used when working with Mapboxes.

            - **layout: All go.Figure() layout options as kwargs
            - **mapbox: All go.layout.Mapbox() options as kwargs
        """

        layout_kwargs = {
            
        }
        mapbox_kwargs = {
            'bearing': 0,
            'zoom': 9,
        }

        layout_kwargs.update(layout)
        mapbox_kwargs.update(mapbox)

        self.figure.update_layout(
                mapbox=go.layout.Mapbox(
                    accesstoken=self.token,
                    **mapbox_kwargs
                ),
                **layout_kwargs
        )

    def jam_with_links(self, basemap, segments, *args, **kwargs):
        """
            Function to plot one jam, with colors indicating the LinkIDs.
            TODO: Add coordinates to this function
        """
        basemap.check('segments', 'nodes')

        link_ids = basemap.map['segments'].loc[segments]['LinkID']
        colors = links_unique(link_ids)
        (lons, lats) = list(zip(*jam_get_coordinates(basemap, segments)))

        self.figure.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers',
            text=link_ids,
            marker=go.scattermapbox.Marker(
                size=10,
                color=colors
            ),
            name="Trace 0"
        ))

        layout = {}
        if 'layout' in kwargs:
            layout.update(kwargs['layout'])

        mapbox = {
            'center': go.layout.mapbox.Center(
                lat=np.median(lats),
                lon=np.median(lons)
            )
        }
        if 'mapbox' in kwargs:
            mapbox.update(kwargs['mapbox'])
        
        self.plotly_mapbox_init(layout, mapbox)

        return self.figure

class Dashboard(object):

    EXTERNAL_STYLESHEETS = [
        'https://codepen.io/chriddyp/pen/bWLwgP.css'
    ]
    
    def __init__(self, host = "0.0.0.0", port = 8050, stylesheets = []):
        self.host = host
        self.port = port
        self.stylesheets = stylesheets

        self.stylesheets.extend(self.EXTERNAL_STYLESHEETS)

        self.app = JupyterDash(__name__, external_stylesheets=self.stylesheets)

    def init(self):
        return self.reset()

    def reset(self):
        self.app = JupyterDash(__name__, external_stylesheets=self.stylesheets)
        return self.app

    def run(self, host = None):
        if host is not None:
            self.host = host
        self.app.run_server(host=self.host, port=self.port)
        return 

   
    def update_table(self, page_current, page_size):
        jams_to_show = self.dataframe.iloc[page_current * page_size:(page_current + 1) * page_size][['timestamp', 'roadClass']]
        return [{"idx": idx, "timestamp": str(value[0]), "roadClass": str(value[1]),  "id": idx} for idx, value in list(zip(jams_to_show.index.values, jams_to_show.values)) ]

    

def jam_get_coordinates(basemap, segments, *args, **kwargs) -> pd.Series:
    basemap.check('segments', 'nodes')
    return basemap.map['nodes'].loc[basemap.map['segments'].loc[segments]['BeginNodeID'].values.tolist()][['Longitude', 'Latitude']].values.tolist()

def jam_normalize_speed_to_colors(optimal_travel_speeds: np.ndarray, actual_travel_speeds: np.ndarray) -> np.ndarray:
    """
        Function to normalize speeds to colors in a range of [0, 1].
        0 equals 0 km/h
        1 equals the optimalTravelSpeed for that segment (mostly 30, 50, 70, 90 or 120 km/h).
        NOTE: The normalization happens per segment, so not for the whole jam.
        This means that 30km/h can be 1 when that segment occurs together in a jam with segments going to 120km/h.
        args:
            optimal_travel_speeds and actual_travel_speeds of the TrafficJam you want to convert the speeds to color-values
            - optimal_travel_speeds: np.ndarray()
                Shape: (X,)
                    X: Segments
            - actual_travel_speeds: np.ndarray()
                Shape: (X,Y)
                    X: Segments
                    Y: Minutes in time
        returns:
            Normalized speeds in a range of [0, 1]
            - np.ndarray()
                Shape: (X, Y)
                    X: Segments
                    Y: Minutes in time
    """
    colors = [(optimal_travel_speeds - actualTravelSpeed) / optimal_travel_speeds
                # We need to transpose the speeds so that one item from the list equals the shape of the optimalTravelSpeeds. (i.e.: iterate over all minutes, getting all segments each iteration)
                for actualTravelSpeed in actual_travel_speeds.T
                ]
    colors = np.asarray(colors).T  # We have to transpose it again
    return colors

def dataframe_normalize_speed_to_colors(dataframe, optimal_travel_speeds_column, actual_travel_speeds_column):
    return dataframe.apply(lambda row: jam_normalize_speed_to_colors(row[optimal_travel_speeds_column], row[actual_travel_speeds_column]), axis=1)
