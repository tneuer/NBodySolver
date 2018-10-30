#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : DashComparison.py
    # Creation Date : Mon 29 Okt 2018 15:20:38 CET
    # Last Modified : Die 30 Okt 2018 17:15:46 CET
    # Description : Compares the results of the NBodySolvers for comparison.
"""
#==============================================================================

import time
import dash
import json
import random

import numpy as np
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html


from colour import Color
from collections import deque
from dash.dependencies import Input, Output, Event, State

from LeapFrog import N_Body_Gravitation_LF

DT = 60*60*24
M_SUN = 1.98892e30
CLICKS_START = 0
CLICKS_RESET = 0
SOLVER = {}

app = dash.Dash("vehicle_data")

app.layout = html.Div([
    html.H1(
            children='N-Body Solvers',
            style={
                'float': 'center',
                'textAlign': 'center',
                }
            ),

    html.Div([
        html.Div([
            html.Div(children="Available planets:"),

            html.Div(id="div_dd_planets", children=dcc.Dropdown(
                id='dd_planets',
                options=[
                    {'label': "Mercury", 'value': "mercury"},
                    {'label': "Venus", 'value': "venus"},
                    {'label': "Earth", 'value': "earth"},
                    {'label': "Mars", 'value': "mars"},
                    {'label': "Jupiter", 'value': "jupiter"},
                    {'label': "Saturn", 'value': "saturn"},
                    {'label': "Uranus", 'value': "uranus"},
                    {'label': "Neptun", 'value': "neptun"},
                    ],
                value=['earth', 'venus'],
                multi=True,
                style={"marginTop": "50px"}
                )),
            ],
            className="eight columns"),

        html.Div([
            html.Div([
                html.Div([
                    html.Div("Bodyname", style={"textAlign": "center"}),
                    html.Div("x-init [AU]", style={"textAlign": "center", "marginTop": "30px"}),
                    html.Div("y-init [AU]", style={"textAlign": "center", "marginTop": "30px"}),
                    html.Div("z-init [AU]", style={"textAlign": "center", "marginTop": "30px"}),
                    html.Div("Colour [HTML]", style={"textAlign": "center", "marginTop": "30px"})
                    ],
                    className="three columns"),
                html.Div([
                    dcc.Input(id="bodyname", style={"padding": "0px"}, value="Custom1"),
                    dcc.Input(id="xpos", style={"padding": "0px"}),
                    dcc.Input(id="ypos", style={"padding": "0px"}, value=0),
                    dcc.Input(id="zpos", style={"padding": "0px"}, value=0),
                    dcc.Input(id="pcolor", style={"padding": "0px"}),
                    ],
                    className="three columns"),
                html.Div([
                    html.Div("Bodymass [kg]", style={"textAlign": "center"}),
                    html.Div("vx-init [AU]", style={"textAlign": "center", "marginTop": "30px"}),
                    html.Div("vy-init [m/s]", style={"textAlign": "center", "marginTop": "30px"}),
                    html.Div("vz-init [m/s]", style={"textAlign": "center", "marginTop": "30px"}),
                    html.Button(id="AddPlanet", children="AddPlanet", style={"marginTop": "30px"})
                    ],
                    className="three columns"),
                html.Div([
                    dcc.Input(id="bodymass", style={"padding": "0px"}),
                    dcc.Input(id="xvel", style={"padding": "0px"}, value=0),
                    dcc.Input(id="yvel", style={"padding": "0px"}),
                    dcc.Input(id="zvel", style={"padding": "0px"}, value=0),
                    ],
                    className="three columns"),
                ],
                className="row")
            ],
            className="four columns"),
        ],
        className='row'),

    html.Div([
        html.Div("Solver", style={"marginTop": "10px"}),
        dcc.Checklist(
            options=[
                {"label": "Leapfrog", "value": "lf"},
                {"label": "Euler", "value": "em"},
                {"label": "Runge-Kutta 2", "value": "rk2"},
                {"label": "Runge-Kutta 4", "value": "rk4"},
                ],
            values = ["lf", "rk2"],
            labelStyle={
                "display": "inline-block",
                "width": "200px"
                }
            ),
        ]),

    html.Div(
        children=html.Div(id='graphs'),
        ),
    dcc.Interval(id="graph-update", interval=1e8),

    html.Div([
        html.Div(id="StartStopButton", children=html.Button(
            id="start_stop",
            children="Start",
            style={"background-color": "#00F240"},
            className="two columns")),
        html.Button(id="reset", children="Reset", className="two columns"),
        html.Div(dcc.Slider(
            id="sun_mass",
            marks={i: "{}".format(10**i) for i in range(-3,4)},
            min=-3,
            max=3,
            value=0,
            step=0.01,
            updatemode="drag"
            ), className="four columns"),
        html.Div(children="Timestep", className="one column"),
        html.Div(dcc.Input(
                    id="timestep",
                    value=DT,
                    style={"width": "95%", "padding": "0px"}
                    ),
                className="one column",
                style={"marginLeft": "-10px"}
                ),
        html.Div(children="Drag", className="one column"),
        html.Div(dcc.Input(
                    id="drag",
                    value=1,
                    style={"width": "95%", "padding": "0px"}
                    ),
                className="one column",
                style={"marginLeft": "-20px"}
                ),
        ],
        className="row"),
    ],
    style={
        'width':'98%',
        'margin-left':10,
        'margin-right':10,
        'max-width':50000},
    )

external_css = [
        "https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css",
        'https://codepen.io/chriddyp/pen/bWLwgP.css',
        ]
for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/js/materialize.min.js']
for js in external_css:
    app.scripts.append_script({'external_url': js})


@app.callback(
        Output("StartStopButton", "children"),
        [Input("start_stop", "n_clicks"), Input("reset", "n_clicks")],
        [State("start_stop", "children")])
def change_start_to_stop_Button(clicks_start, clicks_reset, status):
    if status == "Stop" or clicks_start is None:
        button = html.Button(
            id="start_stop",
            children="Start",
            style={"background-color": "#00F240"},
            className="two columns")
        return button

    else:
        button = html.Button(
            id="start_stop",
            children="Stop",
            style={"background-color": "#FF0000"},
            className="two columns")
        return button


@app.callback(
        Output("div_dd_planets", "children"),
        [Input("AddPlanet", "n_clicks")],
        [State("dd_planets", "value"), State("dd_planets", "options"),
        State("bodyname", "value"), State("bodymass", "value"),
        State("xpos", "value"), State("ypos", "value"), State("zpos", "value"),
        State("xvel", "value"), State("yvel", "value"), State("zvel", "value"),
        State("pcolor", "value")]
        )
def update_dropdown_planets(clicks, values, options, *args):
    correct_planet_format = transform_user_planet_input(args)
    if correct_planet_format:
        name = correct_planet_format["name"]
        options += [{"label": name, "value": name.lower()}]
        values += [name.lower()]

    dropdown_menu = dcc.Dropdown(
        id='dd_planets',
        options=options,
        value=values,
        multi=True,
        style={"marginTop": "50px"}
        )

    return dropdown_menu

def transform_user_planet_input(user_input):
    planet_info = {}
    try:
        planet_info["name"] = user_input[0]
        planet_info["mass"] = np.float(user_input[1])
        planet_info["r_init"] = [
                np.float(user_input[2]),
                np.float(user_input[3]),
                np.float(user_input[4])]
        planet_info["v_init"] = [
                np.float(user_input[5]),
                np.float(user_input[6]),
                np.float(user_input[7])]
        color = user_input[8].replace(" ", "")
        planet_info["color"] = Color(color)

        return planet_info

    except (TypeError, ValueError, AttributeError):
        return False


@app.callback(
        Output("graph-update", "interval"),
        [Input("start_stop", "n_clicks"), Input("reset", "n_clicks")],
        [State("start_stop", "children"), State("graph-update", "interval")]
        )
def control_Animation(start_clicks, reset_clicks, start_button_status, interval):
    global CLICKS_START, CLICKS_RESET, SOLVER, DT

    if start_clicks is not None:
        CLICKS_START = start_clicks
        interval = 100 if interval==1e8 else 1e8
    elif reset_clicks is not None and CLICKS_RESET-reset_clicks:
        CLICKS_RESET = reset_clicks
        interval= 1e8
        SOLVER = {"LF": N_Body_Gravitation_LF(DT)}


    return interval


@app.callback(
        Output("graphs", "children"),
        [Input("dd_planets", "value"), Input("sun_mass", "value"),
        Input("timestep", "value"), Input("drag", "value")],
        events=[Event("graph-update", "interval")]
        )
def update_graphs(planets, sun_mass, timestep, drag):
    global CLICKS_START
    sun_mass = transform_sun_mass(sun_mass)

    print("Started")

    # planet_trajectories = graphs.append(html.Div(dcc.Graph(
    #         id=data_name,
    #         animate=True,
    #         figure={'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(times),max(times)]),
    #                                                     yaxis=dict(range=[min(data_dict[data_name]),max(data_dict[data_name])]),
    #                                                     margin={'l':50,'r':1,'t':45,'b':1},
    #                                                     title='{}'.format(data_name))}
    #         ), className=class_choice))

    return None


def transform_sun_mass(sun_mass_scale):
    return M_SUN  * 10**sun_mass_scale


def read_json_to_correct_input(jsonpath, planets):

    with open(jsonpath, "r") as f:
        initials = json.load(f)

    masses = []
    r_init = []
    v_init = []
    colors = []

    for key, value in initials.items():
        if key in planets:
            masses.append(value["mass"])
            r_init.append(value["r_init"][:2])
            v_init.append(value["v_init"][:2])
            colors.append(value["color"])

    masses = np.array(masses); r_init = np.array(r_init); v_init = np.array(v_init)

    initials = {
            "r": r_init,
            "v": v_init,
            "m": masses
            }

    print(initials)

    return masses, r_init, v_init, colors



if __name__ == "__main__":
    app.server.run(debug=True)



