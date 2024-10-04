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
    # print(predictions_array)
    # final_result=predict_condition(predictions_array)
    #Problem
    r = predict_and_recommend(predictions_array)
    predicted_condition, recommended_medicine, suggested_diet = r
    print(predicted_condition, recommended_medicine, suggested_diet)
    return predictions_array,features_df,predicted_condition, recommended_medicine, suggested_diet
