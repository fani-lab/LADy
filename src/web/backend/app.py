
import random
import csv
from flask_cors import CORS
from flask import request, jsonify
from flask import Flask

import sys
import os
import argparse
# needs to be before importing Review and Lda
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..',)))
print(sys.path)

from cmn.review import Review
from aml.lda import Lda
from aml.btm import Btm
from aml.ctm import Ctm
from aml.rnd import Rnd
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests


@app.route('/api', methods=['POST'])
def api():
    nwords = 20
    text = request.json['text'] + " from backend"
    model = request.json['model']
    lang = request.json['lang']
    naspects = request.json['naspects']

    print(text, model, lang, naspects)

    parser = argparse.ArgumentParser(description='Latent Aspect Detection')
    parser.add_argument('-naspects', dest='naspects', type=int, default=5,
                        help='user-defined number of aspects, e.g., -naspect 25')
    args = parser.parse_args()

    if model == "lda": am = Lda(naspects, nwords)
    if model == "btm": am = Btm(naspects, nwords)
    if model == "rnd": am = Rnd(naspects, nwords)
    if model == "ctm": am = Ctm(naspects, nwords, contextual_size = 768, nsamples =10)
    path = f"./models/{naspects}{'.'+lang if lang else ''}/{am.name()}"
    am.load(f'{path}/f0.')
   
    r = Review(id=0, sentences=[text.split()], time=None, author=None, aos=[[([0],[],0)]], lempos=None, parent=None, lang='eng_Latn', category=None)
    # predicting the words as aspects in descending order
    r_pred_aspects = am.infer_batch(reviews_test=[r], h_ratio=0.0, doctype='snt')[0][1][:naspects]
    resultdict = dict((x, str(y)) for x, y in r_pred_aspects)
    print('result',resultdict)
    return jsonify(resultdict)


@app.route("/random", methods=['GET'])
def get_random_row_from_csv():
    file_path = 'reviews.csv'
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        return jsonify(random.choice(rows))


if __name__ == "__main__":
    app.run()
