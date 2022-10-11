import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from functions import *
import pickle
import os
import tensorflow as tf

knn = pickle.load(open("models/knn_model_music", 'rb'))
svm = pickle.load(open("models/svm_model_music", 'rb'))
nn = tf.keras.models.load_model("models/nn_model_music.h5")

app = dash.Dash(__name__)
server = app.server

app.title="Music Genre Classification"

app.layout = html.Div(
    children=[
        dcc.Loading(id="loading", className="loading", type="default"),
        dcc.Upload(
            id="file_upload",
            children=html.Div(
                ["Drag and drop or click to select .mp3 or .wav music file to predict genre."]
            ),
            style={
                "width": "calc(100% - 22px)",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            }
        ),
        html.Audio(
            id='audio_player',
            controls=True,
            autoPlay=False
        ),
        html.Div(
            id="graphs_div"
        ),
    ]
)

@app.callback(
    [
        Output('graphs_div', 'children'),
        Output("audio_player", "src"),
        Output("loading", "children"),
    ],
    [
        Input('file_upload', 'filename'),
        Input('file_upload', 'contents'),
    ]
)
def update_graphs(filename, filecontents):
    div_children = []
    if filename is not None and filecontents is not None:
        save_file(filename, filecontents)
        features = extract_features_audio_file("audio_files/" + filename, offset=30, duration=60)
        test_frame = normilize_newdata(features)

        genres = le.inverse_transform([i for i in range(10)])

        svm_graph = px.bar(
            x=genres,
            y=svm.predict_proba(test_frame)[0],
            labels={"x": "Support Vector Machine"},
        )
        svm_graph.update_layout(margin=dict(t=10, b=10, l=10, r=10), yaxis_title=None)

        knn_graph = px.bar(
            x=genres,
            y=knn.predict_proba(test_frame)[0],
            labels={"x": "k-Nearest Neighbors"},
        )
        knn_graph.update_layout(margin=dict(t=10, b=10, l=10, r=10), yaxis_title=None)

        nn_graph = px.bar(
            x=genres,
            y=nn.predict(test_frame)[0],
            labels={"x": "Deep Neural Network"},
        )
        nn_graph.update_layout(margin=dict(t=10, b=10, l=10, r=10), yaxis_title=None)

        div_children = [
            dcc.Graph(figure=svm_graph),
            dcc.Graph(figure=knn_graph),
            dcc.Graph(figure=nn_graph),
        ]

        os.remove("audio_files/" + filename)
        
    return [div_children, filecontents, ""]


if __name__ == '__main__':
    app.run_server(debug=True)