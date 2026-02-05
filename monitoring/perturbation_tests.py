# perturbation_tests.py

import numpy as np

def add_gaussian_noise(X, std=1e-3):
    X = np.asarray(X, dtype=float)  # convert list/mixed to ndarray
    noise = np.random.normal(0, std, X.shape)
    return X + noise

def decision_flip_rate(engine, X_original: np.ndarray, X_perturbed: np.ndarray):
    """
    Measure how often predictions flip after perturbation.
    Works with StreamingInferenceEngine that returns probabilities.
    """
    # Get predictions (probabilities) from the engine
    # You may need to adjust depending on your engine API
    def get_probs(X):
        # If your engine can run on a DataFrame:
        import pandas as pd
        df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        engine.run(1, input_df=df)  # run 1 step with the new input
        pred_df = engine.get_prediction_dataframe()
        prob_cols = [c for c in pred_df.columns if c.startswith("p_class_")]
        probs = pred_df[prob_cols].to_numpy()
        return probs

    y_orig = np.argmax(get_probs(X_original), axis=1)
    y_pert = np.argmax(get_probs(X_perturbed), axis=1)

    flips = (y_orig != y_pert).sum()
    return flips / len(y_orig)

def counterfactual_stability(engine, X_original: np.ndarray, perturb_fn, n_trials=5):
    """
    Apply a perturbation function multiple times and measure output stability.
    """
    flip_rates = []

    for _ in range(n_trials):
        X_pert = perturb_fn(X_original)
        rate = decision_flip_rate(engine, X_original, X_pert)
        flip_rates.append(rate)

    return np.mean(flip_rates)


def decision_flip_rate_from_probs(y_orig_probs, y_pert_probs):
    y_orig = np.argmax(y_orig_probs, axis=1)
    y_pert = np.argmax(y_pert_probs, axis=1)
    flips = (y_orig != y_pert).sum()
    return flips / len(y_orig)
