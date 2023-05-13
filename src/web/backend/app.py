
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
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests


@app.route('/api', methods=['POST'])
def api():

    data = {"bird": 0.3, "cat": 0.2, "dog": 0.5, "fish": 0.1, "horse": 0.4}

    text = request.json['text'] + " from backend"
    model = request.json['model']
    lang = request.json['lang']
    num = request.json['num']

    path = f"./models/{num}.{lang}/{model}"
    print(text, model, lang, num)

    parser = argparse.ArgumentParser(description='Latent Aspect Detection')
    parser.add_argument('-naspects', dest='naspects', type=int, default=5,
                        help='user-defined number of aspects, e.g., -naspect 25')
    args = parser.parse_args()

    am = Lda(args.naspects)
    am.load(f'{path}/f0.', "")

    # creating a single object for the input review after removing stop words and split()
    r = Review(id=0, sentences=[['ree']], time=None,
               author=None, aos=None, lempos=None, parent=None, lang='eng_Latn')

    # predicting the words as aspects in descending order
    r_pred_aspects = am.infer(r, "snt")
    # print(r_pred_aspects)
    top_words = am.get_aspects_words( 5)
    print('ree',top_words)
    return jsonify(data)


@app.route("/random", methods=['GET'])
def get_random_row_from_csv():
    file_path = 'reviews.csv'
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        return jsonify(random.choice(rows))


if __name__ == "__main__":
    app.run(debug=True)
