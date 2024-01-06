import os.path

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from train import *
from util import convert_pt_model_to_onnx

app = Flask(__name__)
CORS(app)


@app.route('/train', methods=['POST'])
def train_model_request():
    try:

        train_model(request.get_json())

        if os.path.exists("out/model.pt"):
            convert_pt_model_to_onnx("model")
            if os.path.exists("out/model.onnx"):
                return send_file("out/model.onnx", as_attachment=True)
            else:
                return jsonify({'status': 'error', 'message': "An error occurred while converting the model to onnx"})
        else:
            return jsonify({'status': 'error', 'message': "An error occurred while training the model"})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/storeTestData', methods=['POST'])
def store_test_data():
    import json

    try:
        directory = "data/test"
        files = os.listdir(directory)
        numeric_files = [int(os.path.splitext(f)[0]) for f in files if os.path.splitext(f)[0].isdigit() and f.lower().endswith('.json')]

        # Find the file with the highest number
        if numeric_files:
            max_number = max(map(int, numeric_files))
        else:
            max_number = 0

        # Increment the highest number by one
        new_number = max_number + 1

        # Create a new file with the incremented number
        new_file_path = os.path.join(directory, str(new_number) + '.json')  # Adjust the file extension as needed

        with open(new_file_path, 'w') as json_file:
                json.dump(request.get_json(), json_file, indent=4)
        return jsonify({'status': 'success', 'message': "Stored data in file " + new_file_path})

    except Exception:
        return jsonify({'status': 'error', 'message': "An error occurred while storing the data"})


if __name__ == '__main__':
    app.run(debug=True)
