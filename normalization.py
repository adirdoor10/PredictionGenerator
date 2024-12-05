import numpy as np
import pandas as pd

# Define normalization and reverse functions
def min_max_normalize(flat_array):
    global_min = flat_array.min()
    global_max = flat_array.max()
    normalized = (flat_array - global_min) / (global_max - global_min)
    return normalized, {"min": global_min, "max": global_max}

def min_max_reverse(flat_array, params):
    return flat_array * (params["max"] - params["min"]) + params["min"]

def z_score_normalize(flat_array):
    mean = flat_array.mean()
    std = flat_array.std()
    normalized = (flat_array - mean) / std
    return normalized, {"mean": mean, "std": std}

def z_score_reverse(flat_array, params):
    return flat_array * params["std"] + params["mean"]

def log_normalize(flat_array):
    normalized = np.log(flat_array + 1)  # Avoid log(0) by adding 1
    return normalized, {}

def log_reverse(flat_array, params):
    return np.exp(flat_array) - 1

# Define normalization methods dictionary
NORMALIZATION_METHODS = {
    "min-max": {
        "normalize": min_max_normalize,
        "reverse": min_max_reverse
    },
    "z-score": {
        "normalize": z_score_normalize,
        "reverse": z_score_reverse
    },
    "log": {
        "normalize": log_normalize,
        "reverse": log_reverse
    }
}

def setup_and_normalize_data(data, normalize_method):
    ocrs = data["ocrs"]
    ocrs.append(data["head"])
    dimension_names = data["ocrs"][0]["y_data"].keys()
    normalized_ocrs = {}
    normalization_params = {}
    for dim in dimension_names:
        dim_values = [ocr["y_data"][dim] for ocr in data["ocrs"]]
        [normalized_ocrs[dim], normalization_params[dim]] = normalize_data(dim_values, normalize_method)

    num_ocrs = len(next(iter(normalized_ocrs.values())))
    ocrs = [{"y_data": {}} for _ in range(num_ocrs)]
    for dim, ocr_values in normalized_ocrs.items():
        for ocr_index, values in enumerate(ocr_values):
            ocrs[ocr_index]["y_data"][dim] = values

    head = ocrs.pop()
    data['normalization params'] = normalization_params
    data['head']['y_data'] = head['y_data']
    data['ocrs'] = ocrs
    return data


def normalize_data(nested_array, method="min-max"):
    if method not in NORMALIZATION_METHODS:
        raise ValueError(f"Unsupported normalization method: {method}")

    flat_array = np.concatenate(nested_array)
    normalize_func = NORMALIZATION_METHODS[method]["normalize"]
    normalized_flat, params = normalize_func(flat_array)
    params["method"] = method

    lengths = [len(sub_array) for sub_array in nested_array]
    normalized_nested = np.split(normalized_flat, np.cumsum(lengths)[:-1])

    return [list(arr) for arr in normalized_nested], params

def reverse_normalize_data(normalized_dict, params, convert_to_numpy=False):

    method = params["method"]
    if method not in NORMALIZATION_METHODS:
        raise ValueError(f"Unsupported normalization method: {method}")

    reverse_func = NORMALIZATION_METHODS[method]["reverse"]

    for key, normalized_array in normalized_dict.items():
        if convert_to_numpy:
            normalized_array = np.array(normalized_array)
            denormalized_array = reverse_func(normalized_array, params)
            normalized_dict[key] = denormalized_array.tolist()
        else:
            denormalized_array = reverse_func(normalized_array, params)
            #denormalized_array = denormalized_array.values().to_list()
            normalized_dict[key] = denormalized_array.tolist()

    return normalized_dict


