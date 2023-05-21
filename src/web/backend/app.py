
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

    text = request.json['text'] + " from backend"
    model = request.json['model']
    lang = request.json['lang']
    naspects = request.json['naspects']

    path = f"./models/{naspects}.{lang if lang else ''}/{model}"
    print(text, model, lang, naspects)

    parser = argparse.ArgumentParser(description='Latent Aspect Detection')
    parser.add_argument('-naspects', dest='naspects', type=int, default=5,
                        help='user-defined number of aspects, e.g., -naspect 25')
    args = parser.parse_args()

    if model == "lda": am = Lda(naspects, 20)
    if model == "btm": am = Btm(naspects, 20)
    if model == "ctm": am = Ctm(naspects, 20)
    if model == "rnd": am = Rnd(naspects, 20)

    try:
        am.load(f'{path}/f0.', "")
    except:
        print("No model found")
        return {}

    # creating a single object for the input review after removing stop words and split()
    r = Review(id=0, sentences=[['ree']], time=None,
               author=None, aos=None, lempos=None, parent=None, lang='eng_Latn')

    # predicting the words as aspects in descending order
    r_pred_aspects = am.infer(reviews_test=[r], h_ratio=0.0, doctype='snt')[0][1][:naspects]
    # print(r_pred_aspects)
    #top_words = am.get_aspects_words( 5)

    print('top_words',r_pred_aspects)
    
    # word = top_words[:len(top_words)//2]
    # value = top_words[len(top_words)//2:]
    # print('word ', word[0])
    # print('value ', value[0])
    # data = dict(zip(word[0][0], value[0][0]))
    return jsonify(r_pred_aspects)


@app.route("/random", methods=['GET'])
def get_random_row_from_csv():
    file_path = 'reviews.csv'
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        return jsonify(random.choice(rows))


if __name__ == "__main__":
    app.run(debug=True)
