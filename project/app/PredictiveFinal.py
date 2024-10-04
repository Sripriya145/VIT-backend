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



