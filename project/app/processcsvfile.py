# import numpy as np
# import pandas as pd

# from trainedmodel1 import load_model
# # Assuming fhr and uct are your input arrays:
# from extractfreature_function import extract_features
# def predict_on_features(features, model):
#     """Predict on the extracted features using a trained model."""
#     return model.predict(features)



# model = load_model()
# data=pd.read_csv('D:\\react-django\\project\\media\\normal_baby_data.csv')
# fhr=np.array(data['FHR'])
# uct=np.array(data['Uterine Contractions'])
# predictions = []
# # Ensure that fhr and uct have the same length
# assert len(fhr) == len(uct), "FHR and UCT arrays must have the same length."

# # Extract features using the function
# features = extract_features(fhr, uct, window_size=60)
# feature_names = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency']
# features_df = pd.DataFrame(features, columns=feature_names)
# prediction = predict_on_features(features_df, model)
# predictions.append(prediction)

# predictions_array = np.array(predictions)

# print(predictions_array)
import pandas as pd
import numpy as np
from .trainedmodel1 import load_model
from .extractfreature_function import extract_features
import matplotlib.pyplot as plt
import io
import base64


def process_and_predict(filepath):
    """Process the CSV files and make predictions using the machine learning model."""
    
    # Load the files as DataFrames
    first = filepath[0]
    second = filepath[1]
    print(first, second)
    fhr_data = pd.read_csv(first['file_path'])
    uct_data = pd.read_csv(second['file_path'])

    # Ensure that both files have the same length
    if len(fhr_data) != len(uct_data):
        raise ValueError('The two files must have the same length of data')
    
    predictions = []
    
    # Extract FHR and UCT columns
    fhr = np.array(fhr_data['FHR'])  # Assuming FHR is the column name in file1
    uct = np.array(uct_data['Uterine Contractions'])  # Assuming UCT is the column name in file2

    # Define faulty areas - customize these thresholds based on domain knowledge
    faulty_fhr_indices = np.where((fhr < 110) | (fhr > 160))[0]  # Example: FHR abnormal if <110 or >160
    faulty_uct_indices = np.where(uct > 90)[0]  # Example: UCT abnormal if >90

    # Generate the plot using matplotlib
    plt.figure(figsize=(14, 7))  # Further increase figure size if needed

    # Plot normal FHR in blue and faulty parts in red
    for i in range(len(fhr) - 1):
        if i in faulty_fhr_indices:
            plt.plot([i, i+1], [fhr[i], fhr[i+1]], color='red', zorder=1)
        else:
            plt.plot([i, i+1], [fhr[i], fhr[i+1]], color='blue', zorder=1)

    # Plot normal UCT in green and faulty parts in orange
    for i in range(len(uct) - 1):
        if i in faulty_uct_indices:
            plt.plot([i, i+1], [uct[i], uct[i+1]], color='orange', zorder=1)
        else:
            plt.plot([i, i+1], [uct[i], uct[i+1]], color='green', zorder=1)

    # Increase the size of the x-axis and y-axis labels
    plt.xlabel("Time", fontsize=18)  # Further increase font size for x-axis label
    plt.ylabel("Values", fontsize=18)  # Further increase font size for y-axis label
    plt.title("FHR and Uterine Contractions Over Time", fontsize=20)  # Increase title font size

    # Increase x-axis tick size further
    plt.xticks(fontsize=16)

    # Increase y-axis tick size
    plt.yticks(fontsize=16)

    plt.legend(['FHR', 'Uterine Contractions', 'Faulty FHR', 'Faulty UCT'], fontsize=14)

    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Encode image to base64 to send it as a string
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # Extract features
    features = extract_features(fhr, uct, window_size=60)
    feature_names = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency']
    features_df = pd.DataFrame(features, columns=feature_names)
    print(features_df)

    # Load model and make prediction
    model = load_model()
    prediction = model.predict(features_df)

    predictions.append(prediction)
    predictions_array = np.array(predictions)

    # Return prediction and plot
    return {
        "prediction": prediction.tolist(),  # Convert to list for JSON serialization
        "plot": image_base64  # Return the plot as base64 string
    }

def process_and_predict(filepath):
    """Process the CSV files and make predictions using the machine learning model."""
    
    # Load the files as DataFrames
    first = filepath[0]
    second = filepath[1]
    print(first, second)
    fhr_data = pd.read_csv(first['file_path'])
    uct_data = pd.read_csv(second['file_path'])

    # Ensure that both files have the same length
    if len(fhr_data) != len(uct_data):
        raise ValueError('The two files must have the same length of data')
    
    predictions = []
    
    # Extract FHR and UCT columns
    fhr = np.array(fhr_data['FHR'])  # Assuming FHR is the column name in file1
    uct = np.array(uct_data['Uterine Contractions'])  # Assuming UCT is the column name in file2

    # Define faulty areas - customize these thresholds based on domain knowledge
    faulty_fhr_indices = np.where((fhr < 110) | (fhr > 160))[0]  # Example: FHR abnormal if <110 or >160
    faulty_uct_indices = np.where(uct > 90)[0]  # Example: UCT abnormal if >90

    # Generate the plot using matplotlib
    plt.figure(figsize=(12, 6))  # Adjust figure size if needed

    # Plot normal FHR in blue and faulty parts in red
    for i in range(len(fhr) - 1):
        if i in faulty_fhr_indices:
            plt.plot([i, i+1], [fhr[i], fhr[i+1]], color='red', zorder=1)
        else:
            plt.plot([i, i+1], [fhr[i], fhr[i+1]], color='blue', zorder=1)

    # Plot normal UCT in green and faulty parts in orange
    for i in range(len(uct) - 1):
        if i in faulty_uct_indices:
            plt.plot([i, i+1], [uct[i], uct[i+1]], color='orange', zorder=1)
        else:
            plt.plot([i, i+1], [uct[i], uct[i+1]], color='green', zorder=1)

    plt.xlabel("Time", fontsize=14)  # Increase font size for x-axis label
    plt.ylabel("Values", fontsize=14)  # Increase font size for y-axis label
    plt.title("FHR and Uterine Contractions Over Time", fontsize=16)

    # Increase x-axis tick size
    plt.xticks(fontsize=12)

    plt.legend(['FHR', 'Uterine Contractions', 'Faulty FHR', 'Faulty UCT'])

    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Encode image to base64 to send it as a string
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # Extract features
    features = extract_features(fhr, uct, window_size=60)
    feature_names = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency']
    features_df = pd.DataFrame(features, columns=feature_names)
    print(features_df)

    # Load model and make prediction
    model = load_model()
    prediction = model.predict(features_df)

    predictions.append(prediction)
    predictions_array = np.array(predictions)

    # Return prediction and plot
    return {
        "prediction": prediction.tolist(),  # Convert to list for JSON serialization
        "plot": image_base64  # Return the plot as base64 string
    }
