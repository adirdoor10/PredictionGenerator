import pandas as pd
from darts import TimeSeries
from darts.models import RNNModel
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
import matplotlib.pyplot as plt
from darts.models import BlockRNNModel
import requests
import numpy as np
from darts.utils import torch
from pytorch_lightning import Callback
from bottle import Bottle, run
from darts.models import TiDEModel
import json
from normalization import normalize_data, reverse_normalize_data, setup_and_normalize_data
import torch

#http://localhost:8000/start-task?models_list=None&session_id=1&pattern_id=1&normalize_method=min-max&quantile=1.0
class EpochLossLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch + 1} - Loss: {trainer.logged_metrics['train_loss']}")


def create_and_fit_TiDEModel(ocr_list, covariates_list, input_chunk_length, output_chunk_length):
    model = TiDEModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        n_epochs=1,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": [0]
        }
    )

    return model.fit(series=ocr_list, past_covariates=covariates_list)


def create_and_fit_rnn_model(ocr_list, covariates_list, input_chunk_length, output_chunk_length):
    optimizer = torch.optim.Adam
    optimizer_params = {"lr": 0.0001}

    # Create the RNN model with custom optimizer
    model = BlockRNNModel(
        model="LSTM",
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        n_epochs=1,
        n_rnn_layers=7,
        dropout=0.1,
        optimizer_cls=optimizer,
        optimizer_kwargs=optimizer_params,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": [0]
        }
    )

    return model.fit(series=ocr_list, past_covariates=covariates_list)


def create_and_fit_nbeats_model(ocr_list, covariates_list, input_chunk_length, output_chunk_length):
    model = NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        n_epochs=1,
        generic_architecture=True,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": [0]
        }

    )
    return model.fit(series=ocr_list, past_covariates=covariates_list)


model_functions = {
    "Rnn": create_and_fit_rnn_model,
    "NBEATS": create_and_fit_nbeats_model,
    "TiDEModel": create_and_fit_TiDEModel
}


def create_models(ocr_list, covariates_list, models_list, input_chunk_length, output_chunk_length):
    trained_models = {}
    for model_name in models_list:
        if model_name in model_functions:
            trained_models[model_name] = model_functions[model_name](ocr_list, covariates_list, input_chunk_length,
                                                                     output_chunk_length)
        else:
            print(f"Model '{model_name}' is not defined in darts models list ")

    return trained_models


def predict_with_models(trained_models, n_predictions, input_series, covariates_list):
    predictions = {}
    for model_name, model in trained_models.items():
        print(f"Making predictions with {model_name}...")
        if covariates_list:
            predictions[model_name] = model.predict(n=n_predictions, series=input_series,
                                                    past_covariates=covariates_list).values().reshape(-1)

        else:
            predictions[model_name] = model.predict(n=n_predictions, series=input_series).values().reshape(-1)

    return predictions


def save_to_file(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


def setup_data(data):
    covariates_list = []
    ocrs = data['ocrs']
    target_dim_name = data['target_dim_name']
    head_data, stacked_covariates_head, input_chunk_length, output_chunk_length = setup_head_data(data, target_dim_name)
    ocrs_list = []
    for ocr in ocrs:
        stacked_covariates = None

        y_values = np.array(ocr["y_data"][target_dim_name])
        if len(y_values) >= input_chunk_length + output_chunk_length:
            ts = TimeSeries.from_values(values=y_values)
            ocrs_list.append(ts)

            for dim in ocr["y_data"]:
                if dim != target_dim_name:
                    covariates_value = np.array(ocr["y_data"][dim])
                    covariate_series = TimeSeries.from_values(values=covariates_value)
                    if stacked_covariates is None:
                        stacked_covariates = covariate_series
                    else:
                        stacked_covariates = stacked_covariates.stack(covariate_series)
            if stacked_covariates!= None :
                covariates_list.append(stacked_covariates)

            if len(ocr["y_data"]) == 1:
                covariates_list = None

    return ocrs_list, input_chunk_length, covariates_list, output_chunk_length, head_data, stacked_covariates_head


def setup_head_data(data, target_dim_name):
    head_data = TimeSeries.from_values(values=(np.array(data['head']['y_data'][target_dim_name])))
    stacked_covariates = None

    for dim in data['head']['y_data']:

        if dim != target_dim_name:
            covariates_value = np.array(data['head']['y_data'][dim])
            covariate_series = TimeSeries.from_values(values=covariates_value)
            if stacked_covariates is None:
                stacked_covariates = covariate_series
            else:
                stacked_covariates = stacked_covariates.stack(covariate_series)

    return head_data, stacked_covariates, len(head_data), len(data['real_response'])


def start_darts(data, models_list=None, session_id=1, pattern_id=1, normalize_method='min-max'):
    if models_list is None:
        models_list = ["NBEATS"]
    ocr_list, input_chunk_length, covariates_list, output_chunk_length, head_data, stacked_covariates_head = setup_data(
        data)

    save_to_file(data, 'normalized_data.json')

    models = create_models(ocr_list, covariates_list, models_list, input_chunk_length, output_chunk_length)
    predictions = predict_with_models(models, output_chunk_length, head_data, stacked_covariates_head)
    market_response = data['real_response']
    pred = reverse_normalize_data(predictions, data['normalization params'][data['target_dim_name']])
    #pred['real_response'] = data['real_response']
    save_to_file(pred, 'denormalized_pred.json')
    print("finish")
    return pred

#start()
#    market_response_series = pd.Series(market_response, index=pred.time_index)

#plt.figure(figsize=(10, 6))
#pred.plot(label="Prediction")
#market_response_series.plot(label="Actual")
#plt.show()
