from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    ret = { 'success': False, }
    data = request.get_json()
    try:
        clr = pickle.load(open('model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        ret['results'] = clr.predict(scaler.transform(data)).tolist()
        ret['success'] = True
    except Exception as e:
        ret['error'] = str(e)
    return jsonify(ret)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
