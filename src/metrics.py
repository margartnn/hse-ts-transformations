import numpy as np

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float: 
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = 100 * np.mean(numerator / (denominator + eps))

    return smape



