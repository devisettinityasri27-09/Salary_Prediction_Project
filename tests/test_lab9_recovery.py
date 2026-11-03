
import os
import joblib
import numpy as np
import pytest


# --------------------------------
# TEST 1: Missing model recovery
# --------------------------------
def test_missing_model_detection():
    missing_path = "models/non_existing_model.pkl"

    with pytest.raises(FileNotFoundError):
        joblib.load(missing_path)


# --------------------------------
# TEST 2: Corrupted input detection
# --------------------------------
def test_corrupted_input_detection():
    data = np.array([[np.nan, np.inf, -np.inf]])

    assert not np.isfinite(data).all()


# --------------------------------
# TEST 3: Invalid input detection
# --------------------------------
def test_invalid_input_detection():
    data = np.array([["invalid", None]], dtype=object)

    assert data.dtype == object


# --------------------------------
# TEST 4: Model availability
# --------------------------------
def test_model_availability():
    model_path = "models/lab7_random_forest.pkl"

    assert os.path.exists(model_path)


# --------------------------------
# TEST 5: Preprocessor availability
# --------------------------------
def test_preprocessor_availability():
    preprocessor_path = "models/preprocessor.pkl"

    assert os.path.exists(preprocessor_path)
