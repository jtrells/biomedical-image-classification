# pip install -U flask-cors
# export FLASK_APP=app.py
# python -m flask run --host=0.0.0.0 -p 5000
# flask run --host=0.0.0.0 -p 5000
#
import sys
module_path = './'
if module_path not in sys.path:
    sys.path.append(module_path)

import pandas as pd
from os import getenv, path, listdir
from flask import Flask
from flask_cors import CORS
from markupsafe import escape
from dotenv import load_dotenv
from flask_pymongo import PyMongo

from wrapper import get_data

load_dotenv()
app = Flask(__name__)
app.config["MONGO_URI"] = getenv("DB_CONN")
mongo = PyMongo(app)
CORS(app)

ROOT = getenv('FLASK_ROOT')

vil_path = getenv('VIL_PATH')


@app.route(ROOT + '/hello')
def hello():
    return {"message": "hello"}


@app.route(ROOT + "/images/<string:taxonomy>/<string:classifier>/<string:projection>/<string:version>/<string:subset>", methods=['GET'])
def fetch_reduced_image_features(taxonomy, classifier, projection, version, subset):
    taxonomy = escape(taxonomy)
    classifier = escape(classifier)
    projection = escape(projection)
    version = escape(version)
    subset = escape(subset)

    vil_path = getenv('VIL')
    df = get_data(mongo.db, vil_path, taxonomy, classifier,
                  projection, version=version, subset=subset, num_dimensions=2)
    df = df[["img", "img_path", "x", "y", "hits", "label", "prediction"]]

    return df.to_json(orient="records")


def merge_dfs(df1, df2):
    " Given two parquet dataframes, get the label for the deepes element in the taxonomy"
    df1 = df1[["img", "label"]]
    df2 = df2[["img", "label"]]

    df3 = df1.merge(df2, on='img', how='outer')
    df3.label_x = df3.label_x.fillna('')
    df3.label_y = df3.label_y.fillna('')

    # keep label with more details (larger ones)
    df3['label'] = df3.apply(lambda x: x.label_y if len(
        x.label_y) > len(x.label_x) else x.label_x, axis=1)
    df3 = df3.drop(['label_x', 'label_y'], axis=1)

    return df3


@app.route(ROOT + '/tree/<string:taxonomy>', methods=['GET'])
def load_tree(taxonomy):
    taxonomy = escape(taxonomy)
    taxonomy_info = mongo.db.taxonomies.find_one({"name": taxonomy})

    if taxonomy_info:
        taxonomy_tree = [x['label'] for x in taxonomy_info['modalities']]
        taxonomy_tree.append(taxonomy)

        parquets_path = path.join(vil_path, taxonomy)
        dfs = [pd.read_parquet(path.join(parquets_path, x))
               for x in listdir(parquets_path) if x.endswith('.parquet')]
        df = dfs[0]
        for i in range(1, len(dfs)):
            df = merge_dfs(df, dfs[i])
        df = df.drop(['img'], axis=1)

        return {
            "taxonomy": taxonomy_tree,
            "data": df.label.values.tolist()
        }
    else:
        return {}
