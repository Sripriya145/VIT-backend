import joblib
import os
def load_model():
    """Function to load the trained model."""
    model_path = os.path.join(os.path.dirname(__file__), 'multiclassifiertuned1trained.pkl')
    model = joblib.load(model_path)
    return model
