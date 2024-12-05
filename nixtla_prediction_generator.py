import numpy as np
import pandas as pd
from neuralforecast.auto import NHITS, BiTCN
from neuralforecast.core import NeuralForecast
import pandas as pd
import requests
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import NBEATS, RNN

from normalization import reverse_normalize_data

def create_and_fit_Lstm_model(df, input_chunk_length, output_chunk_length, exogenous_vars):
    model = (
        LSTM(
            h=output_chunk_length,
            input_size=input_chunk_length,
            hist_exog_list=exogenous_vars,
            max_steps=1,  # Increased training steps for more thorough optimization
            encoder_n_layers=2,
            encoder_hidden_size=128,
            context_size=10,
            decoder_hidden_size=128,
            decoder_layers=2,
        )
        )

    return model


def create_and_fit_nbeats_model(df, input_chunk_length, output_chunk_length, exogenous_vars):
    model = (
        NBEATS
        (input_size=input_chunk_length,
         h=output_chunk_length,
         max_steps=1,
         )
    )
    return model


def create_and_fit_Rnn(df, input_chunk_length, output_chunk_length, exogenous_vars):
    model =(
        RNN
        (input_size=input_chunk_length,
         h=output_chunk_length,
         hist_exog_list=exogenous_vars,
         max_steps=1,

         )
    )
    return model


model_functions = {
    "LSTM": create_and_fit_Lstm_model,
    "NBEATS": create_and_fit_nbeats_model,
    "Rnn": create_and_fit_Rnn
}


def create_models(df, models_list, input_chunk_length, output_chunk_length, exogenous_vars):
    trained_models = []
    for model_name in models_list:
        if model_name in model_functions:
            trained_models.append(model_functions[model_name](df, input_chunk_length, output_chunk_length,
                                                              exogenous_vars))
        else:
            print(f"Model '{model_name}' is not defined.")

    return trained_models


def setup_data(data):
    head_df, output_len, input_size = setup_head_data(data)
    target_dim_name = data["target_dim_name"]
    ocrs = data["ocrs"]
    rows = []
    for ocr_idx, ocr in enumerate(ocrs):
        unique_id = f"ocr_{ocr_idx + 1}"  # Create a unique_id for each OCR
        y_data = ocr["y_data"]  # y_data contains target and exogenous
        y_values = y_data[target_dim_name]
        exogenous_dims = {dim: values for dim, values in y_data.items() if dim != target_dim_name}

        for i, y_value in enumerate(y_values):
            row = {
                "unique_id": unique_id,
                "ds": i + 1,  # Numeric timestamp starting at 1 for each OCR
                "y": y_value,  # Target value
            }
            for dim, values in exogenous_dims.items():
                row[dim] = values[i]
            rows.append(row)

    ocr_df = pd.DataFrame(rows)
    return ocr_df, head_df, output_len, input_size


def predict_with_models(head_df, trained_models):
    predictions = {}
    for model_name, model in trained_models.items():
        print(f"Making predictions with {model_name}...")
        predictions[model_name] = model.predict(df=head_df)

    return predictions


def setup_head_data(data):
    target_dim_name = data["target_dim_name"]
    head_y_data = data["head"]["y_data"]

    target_values = head_y_data[target_dim_name]
    input_size = len(target_values)
    output_len = len(data["real_response"])

    exogenous_vars = {key: values for key, values in head_y_data.items() if key != target_dim_name}

    head_df = pd.DataFrame({
        "unique_id": ["head"] * len(target_values),
        "ds": range(1, len(target_values) + 1),
        "y": target_values,
        **exogenous_vars
    })
    return head_df, output_len, input_size


def start_nixtla(data, models_list=None, session_id=1, pattern_id=1,
                 normalize_method='min-max'):

    if models_list is None:
        models_list = ["Rnn", "LSTM"]
    ocr_df, head_df, output_len, input_size = setup_data(data)
    exogenous_vars = [col for col in ocr_df.columns if col not in ["unique_id", "ds", "y"]]
    horizon = len(data["real_response"])
    models = create_models(ocr_df, models_list, input_size, output_len, exogenous_vars)
    nf = NeuralForecast(models=models, freq=1)
    nf.fit(df=ocr_df)
    predictions = nf.predict(df=head_df).drop(columns=["ds"], errors="ignore").to_dict(orient="list")
    predictions = reverse_normalize_data(predictions, data['normalization params'][data['target_dim_name']], True)
    return predictions
