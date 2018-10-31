#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : DashComparison.py
    # Creation Date : Mon 29 Okt 2018 15:20:38 CET
    # Last Modified : Mit 31 Okt 2018 13:52:23 CET
    # Description : Compares the results of the NBodySolvers for comparison.
"""
#==============================================================================

import os
import time
import dash
import json
import random

import numpy as np
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html

from shutil import copyfile
from colour import Color
from collections import deque
from dash.dependencies import Input, Output, Event, State

from LeapFrog import N_Body_Gravitation_LF

DT = 60*60*24*4
M_SUN = 1.98892e30
CLICKS_START = 0
CLICKS_RESET = 0
SOLVER = {}
DEFAULTPLANETS = ["earth", "venus", "mercury"]
INITIALS = 0

if not os.path.exists("./user_initials.json"):
    copyfile("./default_initial.json", "./user_initials.json")


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
                    {'label': "Neptune", 'value': "neptune"},
                    ],
                value=DEFAULTPLANETS,
                multi=True,
                style={"marginTop": "50px"}
                )),
            ],
            className="eight columns"),

        html.Div([
            html.Div([
                html.Div([
                    html.Div("Bodyname", style={"textAlign": "center"}),
                    html.Div("x-init [AU]", style={"textAlign": "center", "marginTop": "3px"}),
                    html.Div("y-init [AU]", style={"textAlign": "center", "marginTop": "3px"}),
                    html.Div("z-init [AU]", style={"textAlign": "center", "marginTop": "3px"}),
                    html.Div("Colour [HTML]", style={"textAlign": "center", "marginTop": "3px"})
                    ],
                    className="three columns"),
                html.Div([
                    dcc.Input(id="bodyname", style={"width": "90%", "padding": "0px"}, value="Custom1"),
                    dcc.Input(id="xpos", style={"width": "90%", "padding": "0px"}),
                    dcc.Input(id="ypos", style={"width": "90%", "padding": "0px"}, value=0),
                    dcc.Input(id="zpos", style={"width": "90%", "padding": "0px"}, value=0),
                    dcc.Input(id="pcolor", style={"width": "90%", "padding": "0px"}),
                    ],
                    className="three columns"),
                html.Div([
                    html.Div("Bodymass [kg]", style={"textAlign": "center", "marginTop": "3px"}),
                    html.Div("vx-init [AU]", style={"textAlign": "center", "marginTop": "3px"}),
                    html.Div("vy-init [m/s]", style={"textAlign": "center", "marginTop": "3px"}),
                    html.Div("vz-init [m/s]", style={"textAlign": "center", "marginTop": "3px"}),
                    html.Button(id="AddPlanet", children="AddPlanet", style={"marginTop": "30px"})
                    ],
                    className="three columns"),
                html.Div([
                    dcc.Input(id="bodymass", style={"width": "90%", "padding": "0px"}),
                    dcc.Input(id="xvel", style={"width": "90%", "padding": "0px"}, value=0),
                    dcc.Input(id="yvel", style={"width": "90%", "padding": "0px"}),
                    dcc.Input(id="zvel", style={"width": "90%", "padding": "0px"}, value=0),
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
            id = "Solver",
            options=[
                {"label": "Leapfrog", "value": "LF"},
                {"label": "Euler", "value": "EM"},
                {"label": "Runge-Kutta 2", "value": "RK2"},
                {"label": "Runge-Kutta 4", "value": "RK4"},
                ],
            values = ["LF"],
            labelStyle= {
                "display": "inline-block",
                "width": "200px",
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
            marks={i: "{}".format(np.round(10**i, 2)) for i in np.arange(-0.9, 0.9, 0.3)},
            min=-1,
            max=1,
            value=0,
            step=0.2,
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
    global RUNNING
    if status == "Stop" or clicks_start is None:
        RUNNING = False
        button = html.Button(
            id="start_stop",
            children="Start",
            style={"background-color": "#00F240"},
            className="two columns")
        return button

    else:
        RUNNING = True
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
    global INITIALS

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
        [State("start_stop", "children"), State("graph-update", "interval"),
        State("dd_planets", "value")]
        )
def control_Animation(start_clicks, reset_clicks, start_button_status, interval, planets):
    global CLICKS_START, CLICKS_RESET, SOLVER, DT, R_MAX_INIT, R_MAX, INITIALS
    if start_clicks is not None:
        CLICKS_START = start_clicks
        interval = 200 if interval==1e8 else 1e8
    elif reset_clicks is not None and CLICKS_RESET-reset_clicks:
        CLICKS_RESET = reset_clicks
        interval= 1e8
        INITIALS = read_initials_from_json("./user_initials.json", planets+["sun"])
        R_MAX = np.max(INITIALS["r_init"])
        R_MAX = 1.05 * R_MAX
        R_MAX_INIT = R_MAX
        SOLVER = {"LF": N_Body_Gravitation_LF(DT, INITIALS, verbose=False)}

    return interval


@app.callback(
        Output("graphs", "children"),
        [Input("dd_planets", "value"), Input("sun_mass", "value"),
        Input("timestep", "value"), Input("drag", "value"), Input("reset", "n_clicks")],
        [State("start_stop", "n_clicks"), State("Solver", "values")],
        events=[Event("graph-update", "interval")]
        )
def update_graphs(planets, sun_mass, timestep, drag, n_clicks_reset, n_clicks_start, ode_solvers):
    global CLICKS_START, SOLVER, INITIALS, R_MAX, ENERGY_MAX, VALID_DRAG, RUNNING, GRAPHS, R_MAX_INIT
    sun_mass = transform_sun_mass(sun_mass)
    graphs = []

    try:
        VALID_DRAG = int(drag)
    except (ValueError, TypeError):
        pass

    if CLICKS_START != 0:
        if RUNNING:
            for solver in ode_solvers:
                results  = SOLVER[solver].evolve(steps=1, saveOnly=VALID_DRAG, mass_sun=sun_mass)
                positions = np.array(results[0])
                velocities = results[1]
                timesteps = results[2]
                try:
                    energies = np.abs(np.array(results[3]).sum(axis=1))
                except:
                    print(results)
                forces = results[4]
                max_pos = np.max(positions)
                if max_pos > R_MAX:
                    R_MAX = max_pos
                    R_MAX = 1.05 * R_MAX
                scale_for_escaping = R_MAX_INIT / R_MAX
                INITIALS["sizes"][0] = (np.log10(sun_mass/5.9724e24)+10)
                INITIALS["scaled_sizes"] = INITIALS["sizes"] * scale_for_escaping
                days = np.round(timesteps[-1]/(3600 * 24), 2)
                years = np.round(days/365,2)
    else:
        INITIALS = read_initials_from_json("./user_initials.json", planets+["sun"])
        R_MAX_INIT = np.max(INITIALS["r_init"])
        R_MAX = R_MAX_INIT
        R_MAX = 1.05 * R_MAX_INIT
        SOLVER = {"LF": N_Body_Gravitation_LF(DT, INITIALS, verbose=False)}
        positions = np.array([INITIALS["r_init"]])
        days = 0
        years = 0
        timesteps= [0]
        energies = [0]
        INITIALS["scaled_sizes"] = INITIALS["sizes"]
        ENERGY_MAX = 0

    if n_clicks_reset is not None:
        reset_clicked = bool(n_clicks_reset - CLICKS_RESET)
    else:
        reset_clicked = False

    if reset_clicked:
        print("RESET")
        R_MAX_INIT = np.max(INITIALS["r_init"])
        R_MAX = R_MAX_INIT
        R_MAX = 1.05 * R_MAX_INIT
        SOLVER = {"LF": N_Body_Gravitation_LF(DT, INITIALS, verbose=False)}
        positions = np.array([INITIALS["r_init"]])
        days = 0
        years = 0
        timesteps= [0]
        energies = [0]
        INITIALS["scaled_sizes"] = INITIALS["sizes"]
        ENERGY_MAX = 0

    if RUNNING or CLICKS_START==0 or reset_clicked:
        traj_data = []
        energ_data = []
        for i, name in enumerate(INITIALS["names"]):
            traj_data.append(go.Scatter(
                x = positions[:, i, 0],
                y = positions[:, i, 1],
                name = name,
                marker = {
                    "size": INITIALS["scaled_sizes"][i],
                    "color": INITIALS["colors"][i]
                    }
                ))

        if np.max(energies) > ENERGY_MAX:
            ENERGY_MAX = np.max(energies)

        energ_data.append(go.Bar(
            x = [0],
            y = [energies[-1]],
            name = "LF",
            ))


        planet_trajectories = html.Div(dcc.Graph(
                id="trajectories",
                figure={
                    'data': traj_data,
                    'layout' : go.Layout(
                        xaxis=dict(
                            range=(-R_MAX, R_MAX),
                            showgrid=False,
                            zeroline=False,
                            showline=False,
                            ticks="",
                            showticklabels=False
                            ),
                        yaxis=dict(
                            range=(-R_MAX, R_MAX),
                            showgrid=False,
                            zeroline=False,
                            showline=False,
                            ticks="",
                            showticklabels=False
                            ),
                        margin={'l':50,'r':1,'t':45,'b':1},
                        paper_bgcolor="#000000",
                        plot_bgcolor="#000000",
                        title='{} days / {} years'.format(days, years))
                }
                ), className="eight columns")

        planet_energies = html.Div(dcc.Graph(
                id="energies",
                figure={
                    'data': energ_data,
                    'layout' : go.Layout(
                        xaxis=dict(
                            range=(0, 3),
                            ),
                        yaxis=dict(
                            range=(0, ENERGY_MAX),
                            ),
                        margin={'l':50,'r':1,'t':45,'b':1},
                        title='Energies')
                }
                ), className="four columns")

        GRAPHS = html.Div([
                    planet_trajectories,
                    planet_energies,
                    ],
                    className="row"
                )

    return GRAPHS

def transform_sun_mass(sun_mass_scale):
    return M_SUN  * 10**sun_mass_scale


def read_initials_from_json(jsonpath, planets=None):

    with open(jsonpath, "r") as f:
        initials = json.load(f)

    names = []; masses = []; r_init = []; v_init = []; colors = []; sizes = []

    if planets is None:
        planets = list(initials.keys())

    for key, value in initials.items():
        if key in planets:
            names.append(key)
            masses.append(value["mass"])
            r_init.append(value["r_init"])
            v_init.append(value["v_init"])
            colors.append(value["color"])
            powers_comp_to_earth = np.log10(value["mass"]/5.9742e24)
            markersize = powers_comp_to_earth if powers_comp_to_earth>0 else -1/(powers_comp_to_earth-1)
            sizes.append((markersize+10))

    masses = np.array(masses); r_init = np.array(r_init); v_init = np.array(v_init);
    sizes = np.array(sizes)

    initials = {
            "names": names,
            "r_init": r_init,
            "v_init": v_init,
            "masses": masses,
            "colors": colors,
            "sizes": sizes
            }

    return initials




if __name__ == "__main__":
    app.server.run(debug=True)



