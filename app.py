from flask import Flask, request, jsonify, Response
import threading
import os
import pickle
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
from flask_cors import CORS
from dotenv import load_dotenv
from OrloskyPupilDetector_RealTime import process_video_realtime, start_logging, stop_logging, generate_video_feed

app = Flask(__name__)
CORS(app)
load_dotenv()

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI is not set in the environment")
client = MongoClient(MONGO_URI)
db = client["JobRecVR"]
collection = db["biometrics_gaze"]

# Shared variables
buffered_data = []
is_logging = False
current_id = None


# Start camera process in a separate thread
def start_camera_process():
    thread = threading.Thread(target=process_video_realtime, daemon=True)
    thread.start()


# Load the stress model
def load_model():
    with open("./artifacts/model_stress.pickle", "rb") as f:
        data = pickle.load(f)
        return data["cls"]


# Predict stress level
def predict_stress_from_data(data):
    df = pd.DataFrame(data)

    TEMP = df["TEMP"].values
    if sum((TEMP < 26) | (TEMP > 38)):
        return {"status": "failure", "message": "The temperature should be between 26 and 38"}

    X = df.drop(columns=["datetime"]).values
    model = load_model()
    P = model.predict(X)

    avg_stress = np.sum(P) / (len(P) * 2)

    if avg_stress < 0.25:
        status = "Resilient"
    elif avg_stress < 0.75:
        status = "Adaptive"
    else:
        status = "Overwhelmed"

    avg_stress = avg_stress * 100
    return {
        "status": "success",
        "stress": status,
        "average_stress": f"{avg_stress:.2f}%"
    }


# Routes for Biometric Gaze
@app.route('/startLoggingCam', methods=['POST'])
def start_logging_endpoint():
    data = request.get_json()
    if not data or 'name' not in data:
        return "Error: 'name' field is required.", 400
    name = data['name']
    response = start_logging(name)
    return response


@app.route('/stopLoggingCam', methods=['GET'])
def stop_logging_endpoint():
    response = stop_logging()
    return jsonify({"message": response})


@app.route('/video_feed')
def video_feed():
    """Video streaming route with continuous feed."""
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Routes for Stress Deployment
@app.route('/startLogging', methods=['POST'])
def start_logging_stress():
    global is_logging, buffered_data, current_id
    body = request.get_json()
    current_id = body.get('id')

    if not current_id:
        return jsonify({"status": "failure", "message": "ID is required"}), 400

    is_logging = True
    buffered_data = []
    return jsonify({"status": "success", "message": f"Logging started for ID {current_id}"}), 200


@app.route('/stopLogging', methods=['POST'])
def stop_logging_stress():
    global is_logging, buffered_data, current_id

    if is_logging:
        stress_result = predict_stress_from_data(buffered_data)

        candidate_record = {
            "candidateId": current_id,
            "averageStress": stress_result["average_stress"],
            "stressStatus": stress_result["stress"],
            "biometrics": buffered_data
        }

        collection.insert_one(candidate_record)
        buffered_data = []
        is_logging = False
        current_id = None

        return jsonify({
            "status": "success",
            "message": "Logging stopped and data saved to MongoDB",
            "stress_result": stress_result
        }), 200
    else:
        return jsonify({"status": "failure", "message": "Logging is not currently active"}), 400


# @app.route('/data', methods=['POST'])
# def receive_data():
#     global buffered_data
#     data = request.get_json()
#     bpm = data.get('bpm', None)
#     temp = data.get('temperature', None)
#
#     print(f"Received data: {data}")
#
#     if bpm is None or temp is None:
#         return jsonify({"status": "failure", "message": "Missing bpm or temperature"}), 400
#
#     new_entry = {
#         "HR": bpm,
#         "TEMP": temp,
#         "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     }
#
#     if is_logging:
#         buffered_data.append(new_entry)
#
#     return jsonify({"status": "success"}), 200

# @app.route('/data', methods=['GET', 'POST'])
# def receive_data():
#     global buffered_data
#
#     if request.method == 'POST':
#         # Handle POST requests to receive data
#         data = request.get_json()
#         bpm = data.get('bpm', None)
#         temp = data.get('temperature', None)
#
#         print(f"Received data: {data}")
#
#         if bpm is None or temp is None:
#             return jsonify({"status": "failure", "message": "Missing bpm or temperature"}), 400
#
#         new_entry = {
#             "HR": bpm,
#             "TEMP": temp,
#             "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
#
#         if is_logging:
#             buffered_data.append(new_entry)
#
#         return jsonify({"status": "success"}), 200
#
#     elif request.method == 'GET':
#         # Handle GET requests to fetch the most recent data
#         if buffered_data:
#             latest_data = buffered_data[-1]  # Fetch the latest entry
#             return jsonify({
#                 "status": "success",
#                 "bpm": latest_data["HR"],
#                 "temperature": latest_data["TEMP"],
#                 "datetime": latest_data["datetime"]
#             }), 200
#         else:
#             return jsonify({"status": "failure", "message": "No data available"}), 404
@app.route('/data', methods=['GET', 'POST'])
def receive_data():
    global buffered_data

    if request.method == 'POST':
        # Handle POST requests to receive real-time data from the sensor
        data = request.get_json()
        bpm = data.get('bpm', None)
        temp = data.get('temperature', None)

        print(f"Received data: {data}")

        # If bpm or temp is missing, send an error response
        if bpm is None or temp is None:
            return jsonify({"status": "failure", "message": "Missing bpm or temperature"}), 400

        # Store the latest data in buffered_data
        new_entry = {
            "HR": bpm,
            "TEMP": temp,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Append to buffered data (no need for is_logging here, as we're just collecting sensor data)
        buffered_data.append(new_entry)

        return jsonify({"status": "success"}), 200

    elif request.method == 'GET':
        # Handle GET requests to fetch the most recent BPM and temperature
        if buffered_data:
            latest_data = buffered_data[-1]  # Fetch the latest entry from buffered_data
            return jsonify({
                "status": "success",
                "bpm": latest_data["HR"],
                "temperature": latest_data["TEMP"],
                "datetime": latest_data["datetime"]
            }), 200
        else:
            return jsonify({"status": "failure", "message": "No data available"}), 404

@app.route('/startAll', methods=['POST'])
def start_all_logging():
    """Start both stress and gaze logging."""
    global is_logging, buffered_data, current_id

    # Start stress logging
    body = request.get_json()
    current_id = body.get('id')

    if not current_id:
        return jsonify({"status": "failure", "message": "ID is required"}), 400

    is_logging = True
    buffered_data = []

    # Start gaze logging
    start_logging()

    return jsonify({"status": "success", "message": "Logging started for both stress and gaze."}), 200

@app.route('/stopAll', methods=['POST'])
def stop_all_logging():
    """Stop both stress and gaze logging and save results to the database."""
    global is_logging, buffered_data, current_id

    # Stop stress logging
    if is_logging:
        stress_result = predict_stress_from_data(buffered_data)

        # Stop gaze logging
        gaze_result = stop_logging()

        # Save to database
        candidate_record = {
            "candidateId": current_id,
            "averageStress": stress_result["average_stress"],
            "stressStatus": stress_result["stress"],
            "gaze_patterns": gaze_result["gaze_patterns"],  # Store gaze data
            "focus_index": gaze_result["focus_index"],      # Average gaze percentage
            "biometrics": buffered_data                     # Stress data
        }
        collection.insert_one(candidate_record)

        # Reset logging state
        buffered_data = []
        is_logging = False
        current_id = None

        return jsonify({
            "status": "success",
            "message": "Logging stopped and data saved to MongoDB",
            "stress_result": stress_result,
            "gaze_result": gaze_result["focus_index"]
        }), 200
    else:
        return jsonify({"status": "failure", "message": "Logging is not currently active"}), 400

if __name__ == '__main__':
    # Start the camera process
    start_camera_process()

    # Start the Flask app
    app.run(host='0.0.0.0', port=5000)
