
# # from .processcsvfile import process_and_predict


# # from rest_framework.views import APIView
# # from rest_framework.response import Response
# # from rest_framework import status
# # from django.core.files.storage import FileSystemStorage
# # import csv
# # import os

# # class CsvUploadView(APIView):
# #     def post(self, request, *args, **kwargs):
# #         files = request.FILES.getlist('files')  # Get a list of uploaded files
# #         if not files:
# #             return Response({'error': 'No files provided'}, status=status.HTTP_400_BAD_REQUEST)

# #         uploaded_files_info = []
# #         try:
# #             # Process each file in the uploaded files list
# #             file1_path='D:\\IEEEproject\\MedicalHackBackEnd\\project\\media\\FSH-norm.csv'
# #             file2_path='D:\\IEEEproject\\MedicalHackBackEnd\\project\\media\\Contra-norm.csv'
# #             for file in files:
# #                 # Save the file locally
# #                 fs = FileSystemStorage()
# #                 filename = fs.save(file.name, file)
# #                 file_path = fs.path(filename)
                



# #                 # Check if the file is a CSV
# #                 file_extension = os.path.splitext(file.name)[1]
# #                 if file_extension == '.csv':
# #                     # Process the CSV file (example: read and print the contents)
# #                     with open(file_path, newline='') as csvfile:
# #                         reader = csv.reader(csvfile)
                        
                           
# #                 else:
# #                     # Skip or handle non-CSV files
# #                     print(f"Skipping non-CSV file: {file.name}")

# #                 # Store file information in the response
# #                 uploaded_files_info.append({
# #                     'filename': file.name,
# #                     'file_path': file_path,
# #                     'file_type': file_extension
# #                 })
# #                 print(uploaded_files_info)

# #             features_list, prediction_list = process_and_predict(file1_path,file2_path)
# #             return Response({
# #                 'features': features_list,
# #                 'predictions': prediction_list
# #             }, status=status.HTTP_200_OK)
# #             return Response({'message': 'Files uploaded and processed successfully', 'files': uploaded_files_info}, status=status.HTTP_200_OK)

# #         except Exception as e:
# #             return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# # views.py
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# import pandas as pd
# import numpy as np
# from .trainedmodel1 import load_model
# from .extractfreature_function import extract_features

# class PredictionView(APIView):
#     def get(self, request, *args, **kwargs):
#         try:
#             # Paths to your local files
#             file1_path='D:\\IEEEproject\\MedicalHackBackEnd\\project\\media\\FSH-norm.csv'
#             file2_path='D:\\IEEEproject\\MedicalHackBackEnd\\project\\media\\Contra-norm.csv'
            
#             # Process and predict using your function
#             fhr_data = pd.read_csv(file1_path)
#             uct_data = pd.read_csv(file2_path)

#             if len(fhr_data) != len(uct_data):
#                 return Response({'error': 'The two files must have the same length of data'}, status=status.HTTP_400_BAD_REQUEST)

#             fhr = np.array(fhr_data)
#             uct = np.array(uct_data)
            
#             # Extract features
#             features = extract_features(fhr, uct, window_size=60)
#             feature_names = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency']
#             features_df = pd.DataFrame(features, columns=feature_names)

#             # Load model and make predictions
#             model = load_model()
#             prediction = model.predict(features_df)

#             # Return both features and predictions
#             return Response({
#                 'features': features_df.to_dict(orient='records'),
#                 'predictions': prediction.tolist(),
#             }, status=status.HTTP_200_OK)
        
#         except Exception as e:
#             return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import FileSystemStorage
import csv
import os
from .processcsvfile import process_and_predict
class CsvUploadView(APIView):
    def post(self, request, *args, **kwargs):
        files = request.FILES.getlist('files')
        if not files:
            return Response({'error': 'No files provided'}, status=status.HTTP_400_BAD_REQUEST)

        uploaded_files_info = []
        try:
            for file in files:
                fs = FileSystemStorage()
                filename = fs.save(file.name, file)
                file_path = fs.path(filename)
                file_extension = os.path.splitext(file.name)[1]

                if file_extension == '.csv':
                    uploaded_files_info.append({
                        'filename': file.name,
                        'file_path': file_path,
                        'file_type': file_extension
                    })

            # Process the uploaded CSV files and generate plot
            result = process_and_predict(uploaded_files_info)
            
            return Response({
                'message': 'Files uploaded and processed successfully',
                'files': uploaded_files_info,
                'result': result['prediction'],
                'plot': result['plot']  # Base64-encoded image string
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def post(self, request, *args, **kwargs):
        files = request.FILES.getlist('files')  # Get a list of uploaded files
        if not files:
            return Response({'error': 'No files provided'}, status=status.HTTP_400_BAD_REQUEST)

        uploaded_files_info = []
        try:
            # Process each file in the uploaded files list
            for file in files:
                # Save the file locally
                fs = FileSystemStorage()
                filename = fs.save(file.name, file)
                file_path = fs.path(filename)

                # Check if the file is a CSV
                file_extension = os.path.splitext(file.name)[1]
                if file_extension == '.csv':
                    # Process the CSV file (example: read and print the contents)
                    with open(file_path, newline='') as csvfile:
                        reader = csv.reader(csvfile)
                        
                           
                else:
                    # Skip or handle non-CSV files
                    print(f"Skipping non-CSV file: {file.name}")

                # Store file information in the response
                uploaded_files_info.append({
                    'filename': file.name,
                    'file_path': file_path,
                    'file_type': file_extension
                })
            print(uploaded_files_info)
            result=process_and_predict(uploaded_files_info)
            print(result)

            return Response({'message': 'Files uploaded and processed successfully', 'files': uploaded_files_info,'result':result}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)