# Diabetes Classification Neural Network

This project uses TensorFlow/Keras to develop a neural network for predicting diabetes based on medical data. The trained model is deployed via a Flask-based REST API, allowing easy integration with front-end applications.

## Features
- Neural network for diabetes prediction.
- Flask-based REST API for serving predictions.
- Preprocessing of medical data for training the model.

## Requirements
- Python 3.7+
- TensorFlow
- Flask
- NumPy, Pandas

## Usage
1. Train the model or use a pre-trained model (`diabetes_model.h5`).
2. Run `app.py` to start the Flask server.
3. Use a tool like Postman to send input data to the `/predict` endpoint.

## Example
Send a POST request with input features to `/predict` to get predictions.

