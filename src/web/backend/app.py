from flask import Flask
from flask import request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app) # Allow cross-origin requests

@app.route('/api', methods=['POST'])
def api():
    # Get the data from the POST request.
    data= request.json['text'] + " from backend"
    print(data)
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)