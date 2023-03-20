from dash import html, dcc, Dash, ctx
from dash.dependencies import Input, Output, State
import pickle
import numpy as np
import warnings

from assets.style import TEXT_STYLE, SUB_CONTENT_STYLE, CONTENT_STYLE
from interface.components import dropdown_data_scatter_plot, footer

warnings.filterwarnings('ignore')


def zone(name, id):
    return html.Div([
        name + " : ",
        dcc.Input(id=id, type="number", min=0, max=3000, value=30)
    ], style=TEXT_STYLE),


def is_survied(Init_Win_bytes_forward=62,
               Total_Length_of_Fwd_Packets=90,
               Bwd_Header_Length=40,
               Destination_Port=53,
               Subflow_Fwd_Bytes=68,
               Packet_Length_Std=24.29,
               Packet_Length_Variance=233.2,
               Bwd_Packets_s=2321.3,
               Average_Packet_Size=78,
               Bwd_Packet_Length_Std=121):
    model = pickle.load(open('model_detection_attack.sav', 'rb'))

    x = np.array([Init_Win_bytes_forward,
                  Total_Length_of_Fwd_Packets,
                  Bwd_Header_Length,
                  Destination_Port,
                  Subflow_Fwd_Bytes,
                  Packet_Length_Std,
                  Packet_Length_Variance,
                  Bwd_Packets_s,
                  Average_Packet_Size,
                  Bwd_Packet_Length_Std
                  ]).reshape(1, 10)

    return model.predict(x)[0], model.predict_proba(x)[0][1]


app = Dash(__name__, title='Détection Intrusion')
server = app.server

app.layout = html.Div([
    html.Div([
        html.H1(["Problème de détection d'intrusion réseau"], style=TEXT_STYLE),

        dropdown_data_scatter_plot('Init Win bytes forward :', 'Init_Win_bytes_forward', min=0, max=100000,
                                   detail="Le nombre total d'octets envoyés dans la fenêtre initiale dans le sens direct"),
        dropdown_data_scatter_plot('Total Length of Fwd Packets :', 'Total_Length_of_Fwd_Packets', min=0, max=100000,
                                   detail="La quantité totale d'octets dans le sens direct obtenue à partir de tout le flux (tous les paquets transmis)"),
        dropdown_data_scatter_plot('Bwd Header Length :', 'Bwd_Header_Length', min=0, max=500000,
                                   detail="Longueur de l'entête au retour"),
        dropdown_data_scatter_plot('Destination Port :', 'Destination_Port', min=0, max=66000,
                                   detail="Port de contact de destination du flux"),
        dropdown_data_scatter_plot('Subflow Fwd Bytes :', 'Subflow_Fwd_Bytes', min=0, max=100000,
                                   detail="Moyenne de byte dans un sous flux à l'aller"),
        dropdown_data_scatter_plot('Packet Length Std :', 'Packet_Length_Std', min=0, max=3000,
                                   detail="Longueur standard de paquet enregistré dans le flux"),
        dropdown_data_scatter_plot('Packet Length Variance :', 'Packet_Length_Variance', min=0, max=100000,
                                   detail="Variance de la longueur des paquets dans le flux"),
        dropdown_data_scatter_plot('Bwd Packets s :', 'Bwd_Packets_s', min=0, max=3000,
                                   detail="Nombre packet/s retour"),
        dropdown_data_scatter_plot('Average Packet Size :', 'Average_Packet_Size', min=0, max=3000,
                                   detail="Taille moyenne de chaque paquet"),
        dropdown_data_scatter_plot('Bwd Packet Length Std :', 'Bwd_Packet_Length_Std', min=0, max=6600,
                                   detail="Déviation standard d'un paquet retour"),

        html.Button('Valider', id='button_submit', n_clicks=0),
        html.Br(),
        html.Br(),
        html.Div(id='result'),
        html.Br(),
        html.Br(),

    ],
        style=SUB_CONTENT_STYLE),
    footer,
],
    style=CONTENT_STYLE)


@app.callback(
    Output('result', 'children'),
    Input('button_submit', 'n_clicks'),

    Input('Init_Win_bytes_forward', 'value'),
    Input('Total_Length_of_Fwd_Packets', 'value'),
    Input('Bwd_Header_Length', 'value'),
    Input('Destination_Port', 'value'),
    Input('Subflow_Fwd_Bytes', 'value'),
    Input('Packet_Length_Std', 'value'),
    Input('Packet_Length_Variance', 'value'),
    Input('Bwd_Packets_s', 'value'),
    Input('Average_Packet_Size', 'value'),
    Input('Bwd_Packet_Length_Std', 'value'),

    State('Init_Win_bytes_forward', 'value'),
    State('Total_Length_of_Fwd_Packets', 'value'),
    State('Bwd_Header_Length', 'value'),
    State('Destination_Port', 'value'),
    State('Subflow_Fwd_Bytes', 'value'),
    State('Packet_Length_Std', 'value'),
    State('Packet_Length_Variance', 'value'),
    State('Bwd_Packets_s', 'value'),
    State('Average_Packet_Size', 'value'),
    State('Bwd_Packet_Length_Std', 'value'),
    prevent_initial_call=True
)
def update_output_div(button,

                      input_Init_Win_bytes_forward,
                      input_Total_Length_of_Fwd_Packets,
                      input_Bwd_Header_Length,
                      input_Destination_Port,
                      input_Subflow_Fwd_Bytes,
                      input_Packet_Length_Std,
                      input_Packet_Length_Variance,
                      input_Bwd_Packets_s,
                      input_Average_Packet_Size,
                      input_Bwd_Packet_Length_Std,

                      Init_Win_bytes_forward,
                      Total_Length_of_Fwd_Packets,
                      Bwd_Header_Length,
                      Destination_Port,
                      Subflow_Fwd_Bytes,
                      Packet_Length_Std,
                      Packet_Length_Variance,
                      Bwd_Packets_s,
                      Average_Packet_Size,
                      Bwd_Packet_Length_Std):
    triggered_id = ctx.triggered_id

    # Launch IA --------------------------------------------------------------------
    if triggered_id == 'button_submit':

        survived, percentage_survived = is_survied(Init_Win_bytes_forward,
                                                   Total_Length_of_Fwd_Packets,
                                                   Bwd_Header_Length,
                                                   Destination_Port,
                                                   Subflow_Fwd_Bytes,
                                                   Packet_Length_Std,
                                                   Packet_Length_Variance,
                                                   Bwd_Packets_s,
                                                   Average_Packet_Size,
                                                   Bwd_Packet_Length_Std)
        color = 'red' if survived == 0 else "rgba(0, 255, 0, 1)"
        attack_str = 'ATTACK' if percentage_survived <= 0.5 else "BENIGN"

        return html.Div([
            html.Div([attack_str],
                     style={'color': color, 'fontSize': 25}),
            html.Div([f"Chance qu'il n'y ai pas d'intrusion : {str(round(percentage_survived * 100, 2))}%"],
                     style={'color': color, 'fontSize': 15})
        ])
    else:
        return html.Div([""], style={'fontSize': 25})


if __name__ == '__main__':
    app.run_server(debug=True)
