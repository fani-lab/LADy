from flask import Flask
from flask import request, jsonify
from flask_cors import CORS
import csv
import random

app = Flask(__name__)
CORS(app) # Allow cross-origin requests


@app.route('/api', methods=['POST'])
def api():
    data = {"bird":0.3, "cat":0.2, "dog":0.5, "fish":0.1, "horse":0.4}
    # Get the data from the POST request.
    text= request.json['text'] + " from backend"
    model = request.json['model']
    print(data)
    return jsonify(data)

@app.route("/random", methods=['GET'])
def get_random_row_from_csv():
    file_path ='reviews.csv'
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        return jsonify(random.choice(rows))

if __name__ == "__main__":
    app.run(debug=True)