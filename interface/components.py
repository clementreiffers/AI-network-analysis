from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_daq as daq
from interface.assets.style import TEXT_STYLE
"""

        html.Div([daq.NumericInput(id=id, min=min, max=max, value=max//3),],
                 style={'verticalAlign': 'top', 'textAlign': 'center', 'width': '30%', 'display': 'inline-block'}),

"""

def dropdown_data_scatter_plot(text, id, min=0, max=200, detail=""):
    return html.Div([

        html.Div([text],
                 style={'verticalAlign': 'top', 'width': '26%', 'display': 'inline-block', 'color': 'white'}),

        html.Div([dcc.Input(id=id, type="number", min=min, max=max, value=max // 2)],
                 style={'verticalAlign': 'top', 'textAlign': 'center', 'width': '22%', 'display': 'inline-block'}),

        html.Div([html.Img(src='assets/point_interrogation.png',
                           id='tooltip-target-' + id,
                           style={'height': '100%', 'width': '100%'})],
                 style={'verticalAlign': 'top', 'textAlign': 'center', 'width': '3%', 'display': 'inline-block'}),

        dbc.Tooltip(detail,
                    placement='right',
                    target='tooltip-target-' + id,
                    style=TEXT_STYLE),

    ])


footer = (html.Div([
    html.Div([
        html.Br(),
        html.Div(['Projet | Rendu de mi parcours'],
                 style={'color': 'white'}),
        html.Div(['Maëlle MARCELIN, Camille BAYON DE NOYER, Clément REIFFERS, Sonia MOGHRAOUI'],
                 style={'color': 'white', 'fontSize': 10}),
        html.Br(),
    ], style={'textAlign': 'center', 'backgroundColor': '#212529'})
]))
