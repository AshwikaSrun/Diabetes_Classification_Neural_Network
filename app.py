from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('diabetes_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
