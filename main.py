import matplotlib
from bottle import Bottle, request, response, run
import threading
import time
import json
import requests
from nixtla_prediction_generator import start_nixtla
from darts_prediction_generator import start_darts
from normalization import setup_and_normalize_data
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

app = Bottle()
URL = "http://127.0.0.1:8082/set_dle_predictions/1/1"


def create_combined_json_object(darts_prediction, nixtla_prediction):
    combined_predictions = {**darts_prediction, **nixtla_prediction}
    return combined_predictions


def run_prediction_task(models_list, session_id, pattern_id, callback_url, normalize_method='z-score'):
    models_list = None
    try:
        url = f"http://127.0.0.1:8082/get_pattern_dle_data/{session_id}/{pattern_id}"
        data_response = requests.get(url)
        if data_response.status_code == 200:
            data = data_response.json()
        else:
            print(f"Request failed with status code {data_response.status_code}")

        data = setup_and_normalize_data(data, normalize_method)
        darts_prediction = start_darts(data, models_list, session_id, pattern_id, normalize_method)
        nixtla_prediction = start_nixtla(data,models_list, session_id, pattern_id, normalize_method)
        json_pred = json.dumps((create_combined_json_object(darts_prediction, nixtla_prediction)))
        res = post_prediction(json_pred)
        plot_model_predictions(darts_prediction, nixtla_prediction)


    except Exception as e:
        print(f"Error in prediction task: {e}")


def plot_model_predictions(darts_prediction, nixtla_prediction):
    plt.figure(figsize=(12, 6))
    for model, values in darts_prediction.items():
        plt.plot(values, label=f"Darts - {model}")

    for model, values in nixtla_prediction.items():
        plt.plot(values, label=f"Nixtla - {model}")

    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title("Model Predictions")

    plt.legend()
    plt.grid(True)
    plt.savefig("C:/Users/adird/Documents/plot.png")


def post_prediction(body):
    try:

        response = requests.post(URL, data=body, headers={"Content-Type": "application/json"})
        return {
            "status_code": response.status_code,
            "response_body": response.json() if response.content else "No content"
        }

    except Exception as e:
        return {"error": str(e)}


@app.get('/start-task')
def start_task():
    try:
        # Extract parameters from GET request
        neural_networks = request.query.get("models_list", "").split(",")
        session_id = request.query.get("session_id")
        pattern_id = request.query.get("pattern_id")
        callback_url = request.query.get("callback_url")
        normalize_method = request.query.get("normalize_method", "min-max")

        if not neural_networks or not session_id or not pattern_id or not callback_url:
            response.status = 400
            return {"error": "Missing required parameters"}

        # Start the background task
        task_thread = threading.Thread(
            target=run_prediction_task,
            args=(neural_networks, session_id, pattern_id, callback_url)
        )
        task_thread.start()

        # Respond immediately that the task has started
        return {"status": "Task started", "session_id": session_id, "pattern_id": pattern_id}
    except Exception as e:
        response.status = 500
        return {"error": str(e)}


run(app, host='0.0.0.0', port=8080)
