# coding=utf-8 

import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import datetime as dt
import dash_table
import plotly.express as px
from dash.dependencies import Input, Output


###########
# ci deve sempre essere almeno una funct.
def func():
    a =667

#########################à

#Dataframes Loads

df = pd.read_excel('supplier.xlsx')

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


############################################################
                   # Grafici
############################################################

######################### plot graf1 #######################
x = np.arange(10)

fig = go.Figure(data=go.Scatter(x=x, y=x**2))


######################### plot graf2 #######################

np.random.seed(1)

N = 100
random_x = np.linspace(0, 1, N)
random_y0 = np.random.randn(N) + 5
random_y1 = np.random.randn(N)
random_y2 = np.random.randn(N) - 5

# Create traces
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=random_x, y=random_y0,
                    mode='lines',
                    name='lines'))
fig2.add_trace(go.Scatter(x=random_x, y=random_y1,
                    mode='lines+markers',
                    name='lines+markers'))
fig2.add_trace(go.Scatter(x=random_x, y=random_y2,
                    mode='markers', name='markers'))

######################### plot graf3 #######################
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 3, 2, 3, 1])

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=x, y=y, name="linear",
                    line_shape='linear'))
fig3.add_trace(go.Scatter(x=x, y=y + 5, name="spline",
                    text=["tweak line smoothness<br>with 'smoothing' in line object"],
                    hoverinfo='text+name',
                    line_shape='spline'))
fig3.add_trace(go.Scatter(x=x, y=y + 10, name="vhv",
                    line_shape='vhv'))
fig3.add_trace(go.Scatter(x=x, y=y + 15, name="hvh",
                    line_shape='hvh'))
fig3.add_trace(go.Scatter(x=x, y=y + 20, name="vh",
                    line_shape='vh'))
fig3.add_trace(go.Scatter(x=x, y=y + 25, name="hv",
                    line_shape='hv'))

fig3.update_traces(hoverinfo='text+name', mode='lines+markers')
fig3.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16))

######################### plot graf4 #######################

df4 = px.data.iris()
fig4 = px.scatter(df4, x=df4.sepal_length, y=df4.sepal_width, color=df4.species, size=df4.petal_length)

######################### plot graf5 #######################

df5 = px.data.gapminder()
gdp = np.log(df5['pop'] * df5['gdpPercap'])  # NumPy array
fig5 = px.bar(df5, x='year', y=gdp, color='continent', labels={'y':'log gdp'},
             hover_data=['country'],
             title='Evolution of world GDP')


########################  INPUT   ########################

ALLOWED_TYPES = (
    "text", "number", "password", "email", "search",
    "tel", "url", "range", "hidden",
)


###################################################################
                   # LAYOUT DASH
###################################################################

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__) #, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([   
                html.P("Table 1")
                ], id='table-one-layout', className="four columns info_container ")
            ], className="row"),

#########################################################
#                  PLOTS

        html.Div(children=[
            html.H4(children='titolo grafico'),
            generate_table(df)
        ]),

## Graf1
        html.Div([
            dcc.Graph(figure=fig)
        ]),

## Graf2
        html.Div([
            dcc.Graph(figure=fig2)
        ]),

## Graf3
        html.Div([
            dcc.Graph(figure=fig3)
        ]),
## Graf4
        html.Div([
            dcc.Graph(figure=fig4)
        ]),

## Graf5
        html.Div([
            dcc.Graph(figure=fig5)
        ]),

# INPUT
        html.Div(
            [
                dcc.Input(
                    id="input_{}".format(_),
                    type=_,
                    placeholder="input type {}".format(_),
                )
                for _ in ALLOWED_TYPES
            ]
            + [html.Div(id="out-all-types")]
        ),

#######################

#### RIGA


        html.Div([
                html.Div([
                    html.H3("zzzzzzzzzzzzzzzzzzzzz"),
                ],className="six columns pretty_container"),
                html.Div([
                    html.H3("ddddddddddddddddddddddddddddd"),
                    ], className="six columns pretty_container"),
            ], className="row"), 
        
#####################

        
        ])
    ],id="main")


## for INPUT
@app.callback(
    Output("out-all-types", "children"),
    [Input("input_{}".format(_), "value") for _ in ALLOWED_TYPES],
)
def cb_render(*vals):
    return " | ".join((str(val) for val in vals if val))


if __name__ == '__main__':
    app.run_server(debug=True,use_reloader=True) #, host="0.0.0.0", port=8800)