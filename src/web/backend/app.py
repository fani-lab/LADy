
import random
import csv
from flask_cors import CORS
from flask import request, jsonify, make_response
from flask import Flask

import sys
import os
import argparse
import torch

# needs to be before importing Review and Lda
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..',)))

from cmn.review import Review
from aml.btm import Btm
from aml.ctm import Ctm
from aml.lda import Lda
from aml.rnd import Rnd


__dirname = os.path.dirname(__file__)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

os.environ['CUDA_VISIBLE_DEVICES'] = ''

@app.route('/api', methods=['POST'])
def api():
    nwords = 20
    req = request.json

    if(req is None):
        return jsonify({'error': 'Invalid request params'})

    text = req['text'] + ' from backend'
    model = req['model']
    lang = req['lang']
    naspects = req['naspects']
    print(text, model, lang, naspects)

    parser = argparse.ArgumentParser(description='Latent Aspect Detection')
    parser.add_argument('-naspects', dest='naspects', type=int, default=5,
                        help='user-defined number of aspects, e.g., -naspect 25')
    args = parser.parse_args()

    am = None

    if model == 'lda': am = Lda(naspects, nwords)
    if model == 'btm': am = Btm(naspects, nwords)
    if model == 'rnd': am = Rnd(naspects, nwords)
    if model == 'ctm': am = Ctm(naspects, nwords, contextual_size = 768, nsamples =10)

    if(am is None):
        return jsonify({'error': 'Model not found'})

    path = os.path.join(__dirname, f"models/{naspects}{'.'+lang if lang else ''}/{am.name()}")

    am.load(f'{path}/f0.')
   
    r = Review(id='0', sentences=[text.split()], time=None, author=None, aos=([[([0],[], 0)]]), lempos=None, parent=None, lang='eng_Latn', category=None)
  
    # settings = {'nllb': 'facebook/nllb-200-distilled-600M', 'max_l': 1024, 'device': 'cpu'}
    # res = r.translate('pes_Arab', settings)
    # tranlated_review = res[0].get_txt()
    # backtranslated_review = res[1].get_txt()
    # semantic_similarity = res[2]
    # print('translated',tranlated_review )
    # print("back",backtranslated_review )
    # print("sem", semantic_similarity )

    r_pred_aspects = am.infer_batch(reviews_test=[r], h_ratio=0.0, doctype='snt')[0][1][:naspects]
    resultdict = dict((x, str(y)) for x, y in r_pred_aspects)
    print('result',resultdict)
    return jsonify(resultdict)


@app.route('/random', methods=['GET'])
def get_random_row_from_csv():
    file_path = os.path.join(__dirname, 'reviews.csv')
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            return jsonify(random.choice(rows))
    except FileNotFoundError:
        return jsonify({'error': '[Server]: reviews.csv not found'}), 500


if __name__ == '__main__':
    app.run(debug=True)
