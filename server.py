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


if __name__ == '__main__':
    app.run(debug=True)
