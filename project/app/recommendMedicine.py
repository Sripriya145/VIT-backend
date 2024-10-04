import numpy as np
from collections import Counter

def classify_fetal_condition(most_freq, nsp_code):
    """
    Classify the fetal condition based on the most frequent value in the predictions and the NSP code.
    """
    # FIGO Guidelines Based on NSP Code
    if nsp_code == 1:
        return "Normal: Fetus has no hypoxia or acidosis, no intervention needed."

    elif nsp_code == 2:
        if most_freq in [1, 2, 3, 4]:
            return "Suspect: Constant monitoring required, but no immediate intervention."
        elif most_freq == 5:
            return "Suspect: Accelerative/Decelerative pattern detected, suggesting fetal stress. Close monitoring is needed."
        elif most_freq == 6:
            return "Suspect: Decelerative pattern detected, possible vagal stimulation. Recommend further observation."
        elif most_freq == 7:
            return "Suspect: Largely decelerative pattern detected, monitor carefully."
        elif most_freq == 10:
            return "Suspect: Pattern shifts, might indicate stress. Monitoring is necessary."

    elif nsp_code == 3:
        if most_freq == 9:
            return "Pathologic: Flat-sinusoidal pattern detected, pathological state. Immediate intervention required."
        elif most_freq == 6:
            return "Pathologic: Decelerative pattern with high probability of hypoxia and acidosis. Immediate intervention needed."
        elif most_freq == 7:
            return "Pathologic: Largely decelerative pattern, high risk of fetal distress. Emergency intervention required."
        else:
            return "Pathologic: Critical condition, immediate action needed."

    return "Condition not clearly identifiable, further diagnostics required."

def predict_condition(result_array):
    """
    Predict the fetal condition based on the most frequent value in the result array.
    """
    # Flatten the array and count the most frequent value
    flattened = result_array.flatten()
    freq_counter = Counter(flattened)
    most_freq, count = freq_counter.most_common(1)[0]  # Get the most common value and its count

    # Logic to classify the condition
    n = len(flattened)
    if most_freq == 1:
        return classify_fetal_condition(most_freq, 1)  # Normal
    elif count > n / 2 or most_freq == 3:
        return classify_fetal_condition(most_freq, 3)  # Abnormal
    else:
        return classify_fetal_condition(most_freq, 2)  # Suspicious

def class_find(result_array):
        zero_index_values = result_array[:, :, 0].flatten()  # This gives a 1D array of the 0th elements
        freq_counter = Counter(zero_index_values)
        most_freq, count = freq_counter.most_common(1)[0] 
        print("Most Frequent 0th Element:", most_freq)
        print("Count of Most Frequent Element:", count)
        return most_freq

def recommend_medicine(class_code, nsp_code):
    """
    Recommend suitable medicine based on the classified fetal condition.

    Parameters:
    - class_code: The CTG morphological class (1 to 10, corresponding to classes A to SUSP).
    - nsp_code: The NSP classification (1: Normal, 2: Suspect, 3: Pathologic).

    Returns:
    - A string describing the recommended medicine.
    """

    if nsp_code == 2:
        if class_code in [5, 6, 7, 10]:
            if class_code == 5:
                return "Consider suitable medicine like Propranolol, to be administered under medical supervision."
            elif class_code == 6:
                return "Consider suitable medicine like Metoprolol, to be administered under medical supervision."
            elif class_code == 7:
                return "Consider suitable medicine like Atenolol, to be administered under medical supervision."
            elif class_code == 10:
                return "Consider suitable medicine like Nadolol, to be administered under medical supervision."

    elif nsp_code == 3:
        if class_code in [6, 7, 9]:
            if class_code == 6:
                return "Immediate medical intervention required. Administer Magnesium Sulfate under medical supervision."
            elif class_code == 7:
                return "Immediate medical intervention required. Administer Nifedipine under medical supervision."
            elif class_code == 9:
                return "Immediate medical intervention required. Administer Diazepam under medical supervision."

    return "No specific medicine recommended. Consult with a healthcare professional."

def suggest_diet(class_code, nsp_code):
    """
    Suggest suitable diet based on the classified fetal condition.

    Parameters:
    - class_code: The CTG morphological class (1 to 10, corresponding to classes A to SUSP).
    - nsp_code: The NSP classification (1: Normal, 2: Suspect, 3: Pathologic).

    Returns:
    - A string describing the recommended diet.
    """

    if nsp_code == 2:
        if class_code in [1, 2, 3, 4]:
            return "Monitor and maintain a balanced diet with emphasis on hydration. Include vegetables like spinach and fruits like oranges to boost nutrition."
        elif class_code == 5:
            return "Monitor closely and consider a diet rich in magnesium and potassium. Include bananas and leafy greens like kale."
        elif class_code == 6:
            return "Monitor closely and consider a diet with reduced caffeine and increased hydration. Include foods like oats and berries."
        elif class_code == 7:
            return "Monitor closely and ensure a diet with adequate calcium and potassium. Include dairy products and bananas."
        elif class_code == 10:
            return "Monitor closely and consider a diet with stress-relief nutrients like vitamin C and B complex. Include citrus fruits and nuts."

    elif nsp_code == 3:
        if class_code in [6, 7, 9]:
            if class_code == 6:
                return "Immediate medical intervention required. Consult with a healthcare professional for specific dietary needs."
            elif class_code == 7:
                return "Immediate medical intervention required. Consult with a healthcare professional for specific dietary needs."
            elif class_code == 9:
                return "Immediate medical intervention required. Consult with a healthcare professional for specific dietary needs."

    return "No specific diet recommended. Consult with a healthcare professional."

# def predict_and_recommend(result_array):
#     """
#     Predict the fetal condition, recommend medicine, and suggest diet.
#     """
#     predicted_condition = predict_condition(result_array)
#     print(predicted_condition.split(":")[1].split()[0])
#     class_code = int(predicted_condition.split(":")[0])
#     nsp_code = int(predicted_condition.split(":")[1].split()[0])
#     recommended_medicine = recommend_medicine(class_code, nsp_code)
#     suggested_diet = suggest_diet(class_code, nsp_code)
#     return predicted_condition, recommended_medicine, suggested_diet
def predict_and_recommend(result_array):
    """
    Predict the fetal condition, recommend medicine, and suggest diet.
    """
    # Get the predicted condition string
    predicted_condition = predict_condition(result_array)
    
    # Print the predicted condition for debugging
    # print("Predicted Condition:", predicted_condition)
    
    # Extract NSP classification (e.g., "Normal", "Suspect", "Pathologic")
    nsp_classification = predicted_condition.split(":")[0].strip()
    
    # Map the NSP classification to a numeric code
    nsp_mapping = {
        'Normal': 1,
        'Suspect': 2,
        'Pathologic': 3
    }
    
    # Use the mapping to get the NSP code
    nsp_code = nsp_mapping.get(nsp_classification, None)
    if nsp_code is None:
        print(f"Error: Unknown NSP classification '{nsp_classification}'")
        return None, None, None  # Return early if NSP classification is invalid
    
    # Now assign class_code based on some logic (e.g., based on further analysis)
    # For now, I'll assume the class_code is fixed, or you could have additional logic to determine this.
    # Since class_code isn't present in the string, you'll need to define it in some way.
    
    # Example: Assign a default class_code (this could be more dynamic)
    class_code = class_find(result_array)
    print(class_code)
    # Call the recommend_medicine function with the parsed class_code and nsp_code
    recommended_medicine = recommend_medicine(class_code, nsp_code)
    # Call the suggest_diet function with the parsed class_code and nsp_code
    suggested_diet = suggest_diet(class_code, nsp_code)
    # print(suggested_diet)
    return predicted_condition, recommended_medicine, suggested_diet


