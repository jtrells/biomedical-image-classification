# pip install -U flask-cors
# export FLASK_APP=app.py
# python -m flask run --host=0.0.0.0 -p 6006
#

import pandas as pd
import numpy as np
from flask import Flask
from flask_cors import CORS
# from flask_socketio import SocketIO, send, emit
from markupsafe import escape
from os import path
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

app = Flask(__name__)
CORS(app)
# socketio = SocketIO(app, cors_allowed_origins="*")

ROOT = '/jtrell2-2'
PROJECTIONS = '/mnt/artifacts/projections'
CLASSIFIERS = '/mnt/artifacts/classifiers'

# def calc_neighborhood_hit(df, projection, label_col, n_neighbors=6):    
#     projections = [[i, j] for (i, j) in zip(df.x.values, df.y.values)]
#     neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(projections)
#     le = preprocessing.LabelEncoder().fit(df[label_col].unique())
    
#     n_hits = []
#     for neighborhood in neigh.kneighbors(projections, n_neighbors + 1, return_distance=False):
#         labels  = le.transform(df.iloc[neighborhood][label_col].values)
#         targets = [labels[0]] * (len(labels) - 1) 
#         n_hit = np.mean(targets == labels[1:])
#         n_hits.append(n_hit)
#     return n_hits

def file2json(file_path, projection, dataset):
    df = pd.read_parquet(file_path)
    
    if dataset.upper() != 'UNLABELED':
        df = df[df['split_set'] == dataset.upper()] # if there are no other, returns all :S    
    x = f"{projection}_x"
    y = f"{projection}_y"
    hits = f"{projection}_hits"
    
    print(df.head())
    
    df = df[["img", "img_path", x, y, hits, "target", "target_predicted"]].rename(columns={
        "target": "label",
        x: "x",
        y: "y",
        hits: "hits"
    })
    
    return df.to_json(orient="records")


@app.route('/hello')
def hello():
    return {"message": "hello"}

@app.route(ROOT + "/projection/<string:classifier>/<int:version>/<string:dataset>/<string:projection>", methods=['GET'])
def fetch_projections(classifier, version, dataset, projection):
    unlabeled_suffix = "" if escape(dataset).upper() != 'UNLABELED' else "_unlabeled"    
    file_name = f"{escape(classifier)}{unlabeled_suffix}_v{escape(version)}.parquet"
    file_path = path.join(PROJECTIONS, file_name)
    if path.exists(file_path):
        return file2json(file_path, escape(projection), escape(dataset))
    else:
        return { "error": "file not found" }
    
# this may be an overkill for the prototype. Let's assume that the data is always there for
# every classifer
# @socketio.on("fetch_projection")
# def fetch_projection_socket(classifier, version, projection):
#     file_name = f"{classifier}_{version}.csv"
#     file_path = path.join(PROJECTIONS, file_name)
#     if path.exists(file_path):
#         # also check if the projection has been calculated before
#         emit("fetch_projection", { "status": "complete", "data": file2json(file_path, projection) })
#     else:
#         # calculate projections, can i emit and do something else on another thread?
#         emit("fetch_projection", { "status": "calculating" })
#         calc_projections(classifier, version, projection)
    
# if __name__ == '__main__':
#     socketio.run(app)
