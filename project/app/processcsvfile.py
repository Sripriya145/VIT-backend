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
# from .PredictiveFinal import predict_condition
from .recommendMedicine import predict_and_recommend
import matplotlib.pyplot as plt
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64

def generate_enhanced_plot(fhr, uct, faulty_fhr_indices, faulty_uct_indices):
    """Generate an enhanced plot with better visuals for FHR and UCT data."""
    
    # Create a figure and axis
    plt.figure(figsize=(16, 8))  # Increase the figure size for better readability

    # Plot FHR with normal parts in blue and faulty parts in red, with smooth lines and thicker strokes
    plt.plot(fhr, color='blue', lw=2, label="Normal FHR")  # Smooth and thicker lines for normal FHR
    
    # Highlight faulty FHR areas with red points
    plt.scatter(faulty_fhr_indices, fhr[faulty_fhr_indices], color='red', s=50, zorder=5, label="Faulty FHR", marker='x')
    
    # Plot UCT with normal parts in green and faulty parts in orange
    plt.plot(uct, color='green', lw=2, label="Normal UCT")  # Smooth and thicker lines for normal UCT
    
    # Highlight faulty UCT areas with orange points
    plt.scatter(faulty_uct_indices, uct[faulty_uct_indices], color='orange', s=50, zorder=5, label="Faulty UCT", marker='x')
    
    # Add shaded regions for faulty FHR and UCT for better visual separation
    plt.fill_between(range(len(fhr)), fhr, where=np.isin(range(len(fhr)), faulty_fhr_indices), 
                     color='red', alpha=0.1, label='Faulty FHR Region')
    plt.fill_between(range(len(uct)), uct, where=np.isin(range(len(uct)), faulty_uct_indices), 
                     color='orange', alpha=0.1, label='Faulty UCT Region')

    # Add a grid for better readability of the data points
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.6)

    # Increase the size of the x-axis and y-axis labels for readability
    plt.xlabel("Time (seconds)", fontsize=18)
    plt.ylabel("FHR and UCT values", fontsize=18)
    plt.title("FHR and Uterine Contractions Over Time", fontsize=20, fontweight='bold')

    # Add a legend to explain which line represents what
    plt.legend(fontsize=14)

    # Adjust tick sizes for better clarity
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Encode the image to base64 to send it as a string
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return image_base64

# Example usage within the existing process_and_predict function:

def process_and_predict(filepath):
    """Process the CSV files and make predictions using the machine learning model."""
    
    # Load the files as DataFrames
    first=filepath[0]
    second=filepath[1]
    print(first,second)
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
    faulty_uct_indices = np.where(uct > 60)[0]  # Example: UCT abnormal if >90

    # Generate the enhanced plot
    image_base64 = generate_enhanced_plot(fhr, uct, faulty_fhr_indices, faulty_uct_indices)

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
    
    # Get the predicted condition and recommendations (optional part of your code)
    r = predict_and_recommend(predictions_array)
    predicted_condition, recommended_medicine, suggested_diet = r
    print(predicted_condition, recommended_medicine, suggested_diet)
    
    return predictions_array, features_df, predicted_condition, recommended_medicine, suggested_diet, image_base64
