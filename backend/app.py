from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained model and scaler
model = joblib.load("../model/random_forest_model.pkl")
scaler = joblib.load("../model/scaler.pkl")


@app.route("/")
def home():
    return "Wine Quality Classification API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = [
        data["fixed_acidity"],
        data["volatile_acidity"],
        data["citric_acid"],
        data["residual_sugar"],
        data["chlorides"],
        data["free_sulfur_dioxide"],
        data["total_sulfur_dioxide"],
        data["density"],
        data["pH"],
        data["sulphates"],
        data["alcohol"]
    ]

    # Convert input into NumPy array
    input_data = np.array([features])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        result = "Good Wine"
    else:
        result = "Bad Wine"

    return jsonify({
        "prediction": int(prediction),
        "result": result
    })


if __name__ == "__main__":
    app.run(debug=True)



    