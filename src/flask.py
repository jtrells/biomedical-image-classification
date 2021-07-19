# pip install -U flask-cors
# export FLASK_APP=app.py
# python -m flask run --host=0.0.0.0 -p 5000
# flask run --host=0.0.0.0 -p 5000
#
import sys
module_path = './'
if module_path not in sys.path:
    sys.path.append(module_path)

from os import getenv
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
    df = get_data(mongo.db, vil_path, taxonomy, classifier, projection, version=version, subset=subset, num_dimensions=2)
    df = df[["img", "img_path", "x", "y", "hits", "label", "prediction"]]

    return df.to_json(orient="records")