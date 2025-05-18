from flask import Blueprint, request, jsonify, render_template, make_response
from flask_jwt_extended import jwt_required, get_jwt_identity
import os
from flask import current_app
from model import db, PredictionHistory
import joblib
from joblib import load
import numpy as np
from io import StringIO
import csv
import json, pickle

predict = Blueprint('predict', __name__)

model = joblib.load("co2_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load('feature_columns.pkl')

# Renders the HTML page for prediction form (NO token required)
@predict.route('/predict-form', methods=['GET'])
def predict_form_page():
    return render_template("predict-form.html")


# Handles the prediction via fetch call (token required)
@predict.route('/predict', methods=['POST'])
@jwt_required()
def predict_api():
    user_id = get_jwt_identity()

    try:
        data = request.get_json()

        # Ensure the incoming data is a dict with the required raw fields
        required_fields = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 
                           'Fuel Consumption Hwy (L/100 km)', 'Is_Hybrid', 
                           'Vehicle Class', 'Fuel Type', 'Transmission']

        if not data or not all(field in data for field in required_fields):
            return jsonify({"error": "Missing one or more required features."}), 400

        # Convert single input into DataFrame
        import pandas as pd
        df_input = pd.DataFrame([data])

        # One-hot encode categorical fields to match training format
        df_input = pd.get_dummies(df_input, columns=['Vehicle Class', 'Fuel Type', 'Transmission'], drop_first=False)

        # Align columns with training set
        df_input = df_input.reindex(columns=feature_columns, fill_value=0)

        # Scale the features
        scaled_features = scaler.transform(df_input)

        # Make prediction
        prediction = round(model.predict(scaled_features)[0], 2)
        # Save to database
        record = PredictionHistory(
            user_id=user_id,
            features=json.dumps(data),
            prediction=prediction
        )
        db.session.add(record)
        db.session.commit()

        return jsonify({"prediction": str(prediction)})

    except Exception as e:
        print("🔥 Error in /predict:", e)
        return jsonify({"error": "Something went wrong during prediction."}), 500


@predict.route('/compare-form', methods=['GET'])
def compare_form():
    return render_template("compare-form.html")


@predict.route('/compare', methods=['POST'])
@jwt_required()
def compare_predictions():
    user_id = get_jwt_identity()

    data = request.get_json()

    required_fields = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 
                       'Fuel Consumption Hwy (L/100 km)', 'Is_Hybrid', 
                       'Vehicle Class', 'Fuel Type', 'Transmission']

    if not data or "features1" not in data or "features2" not in data:
        return jsonify({"error": "Missing 'features1' or 'features2'"}), 400

    features1 = data["features1"]
    features2 = data["features2"]

    # Validate feature dicts
    if not all(field in features1 for field in required_fields) or not all(field in features2 for field in required_fields):
        return jsonify({"error": f"Each feature set must include all of: {', '.join(required_fields)}"}), 400
    try:
        import pandas as pd

        # Convert both feature sets to DataFrames
        df1 = pd.DataFrame([features1])
        df2 = pd.DataFrame([features2])

        # One-hot encode
        df1 = pd.get_dummies(df1, columns=['Vehicle Class', 'Fuel Type', 'Transmission'], drop_first=False)
        df2 = pd.get_dummies(df2, columns=['Vehicle Class', 'Fuel Type', 'Transmission'], drop_first=False)

        # Align to training columns
        df1 = df1.reindex(columns=feature_columns, fill_value=0)
        df2 = df2.reindex(columns=feature_columns, fill_value=0)

        # Scale
        scaled1 = scaler.transform(df1)
        scaled2 = scaler.transform(df2)

        # Predict
        prediction1 = round(model.predict(scaled1)[0], 2)
        prediction2 = round(model.predict(scaled2)[0], 2)
        difference = round(abs(prediction1 - prediction2), 2)

        return jsonify({
            "scenario_1_prediction": prediction1,
            "scenario_2_prediction": prediction2,
            "difference": difference
        }), 200

    except Exception as e:
        print("🔥 Error in /compare:", e)
        return jsonify({"error": "Something went wrong during comparison."}), 500



@predict.route('/history', methods=['GET'])
@jwt_required()
def get_history():
    user_id = get_jwt_identity()
    history = PredictionHistory.query.filter_by(user_id=user_id).all()

    return jsonify([
        {
            "id": h.id,
            "features": h.features,
            "prediction": h.prediction,
            "timestamp": h.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }
        for h in history
    ])

@predict.route('/history-view', methods=['GET'])
def view_history_page():
    return render_template("history.html")


@predict.route('/delete-history', methods=['DELETE'])
@jwt_required()
def delete_history():
    user_id = get_jwt_identity()
    history = PredictionHistory.query.filter_by(user_id=user_id).all()

    if not history:
        return jsonify({"message": "No history found to delete."}), 404

    for item in history:
        db.session.delete(item)
    db.session.commit()
    
    return jsonify({"message": "Prediction history deleted successfully."}), 200

@predict.route('/delete-entry/<int:entry_id>', methods=['DELETE'])
@jwt_required()
def delete_single_entry(entry_id):
    user_id = get_jwt_identity()

    entry = PredictionHistory.query.filter_by(id=entry_id, user_id=user_id).first()

    if not entry:
        return jsonify({"error": "Entry not found"}), 404

    db.session.delete(entry)
    db.session.commit()

    return jsonify({"message": "Prediction entry deleted successfully."}), 200

