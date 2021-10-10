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
import json
from os import getenv, path, listdir
from flask import Flask, jsonify, request
from flask_cors import CORS
from markupsafe import escape
from dotenv import load_dotenv
from flask_pymongo import PyMongo

from wrapper import get_data, get_figure_neighbors, get_active_classifiers, 
                    upsert_label_updates, get_updated_images

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
    df = df[["img", "img_path", "x", "y", "hits",
             "label", "prediction", "width", "height", "full_label", "caption", "source"]]

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

        parquet_path = path.join(vil_path, taxonomy, 'merged_labels.parquet')
        # dfs = [pd.read_parquet(path.join(parquets_path, x))
        #        for x in listdir(parquets_path) if x.endswith('.parquet')]
        # df = dfs[0]
        # for i in range(1, len(dfs)):
        #     df = merge_dfs(df, dfs[i])
        # df = df.drop(['img'], axis=1)
        df = pd.read_parquet(parquet_path)

        return {
            "taxonomy": taxonomy_tree,
            "data": df.label.values.tolist()
        }
    else:
        return {}


@app.route(ROOT + '/taxonomy/<string:taxonomy>', methods=['GET'])
def get_taxonomy(taxonomy):
    taxonomy = escape(taxonomy)
    taxonomy_info = mongo.db.taxonomies.find_one({"name": taxonomy})

    if taxonomy_info:
        return {
            "name": taxonomy_info["name"],
            "modalities": taxonomy_info["modalities"]
        }
    else:
        return {}


@app.route(ROOT + '/image/<string:taxonomy>/<string:img_path>', methods=['GET'])
def get_image_info(taxonomy, img_path):
    taxonomy = escape(taxonomy)
    img_path = escape(img_path)
    img_path = img_path.replace('*', '/')
    parquet_path = path.join(vil_path, taxonomy, 'all.parquet')

    df = pd.read_parquet(parquet_path)
    image = df[df.img_path == img_path].iloc[0]

    if 'tinman' in image.img_path:
        last_slash_idx = image.img_path.rfind('/')
        parent_path = image.img_path[0:last_slash_idx]
        related_df = df[df.img_path.str.contains(parent_path)]
        related = json.loads(related_df.to_json(orient="records"))
    else:
        related = []

    return {
        'img_path': image.img_path,
        'caption': image.caption,
        'full_label': image.label,
        'related': related
    }


@app.route(ROOT + '/neighbors/<string:taxonomy>/<string:classifier>/<string:subset>/<string:img_path>/<int:num_neighbors>', methods=['GET'])
def get_neighbors(taxonomy, classifier, subset, img_path, num_neighbors):
    taxonomy = escape(taxonomy)
    img_path = escape(img_path)
    img_path = img_path.replace('*', '/')
    classifier = escape(classifier)
    subset = escape(subset)
    num_neighbors = int(escape(num_neighbors))

    vil_path = getenv('VIL')
    version = 'latest'
    df = get_figure_neighbors(mongo.db, vil_path, taxonomy,
                              classifier, version, img_path, num_neighbors, subset=subset)
    output = json.loads(df.to_json(orient="records"))
    return {'neighbors': output}


@app.route(ROOT + '/classifiers/<string:taxonomy>/active')
def get_available_active_classifiers(taxonomy):
    taxonomy = str(escape(taxonomy))
    classifiers = get_active_classifiers(mongo.db, taxonomy)
    return {'results': classifiers}

@app.route(ROOT + '/images', methods=['POST'])
def upsert_images():
    images_to_update = request.json
    upsert_label_updates(mongo.db, images_to_update)
    return {"hi": "hi"}


@app.route(ROOT + '/images', methods=['GET'])
def get_updated_images():
    images_dictionary = get_updated_images
    return jsonify(images_dictionary)