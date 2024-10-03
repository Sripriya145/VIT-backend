import numpy as np

# Constants
window_size = 60  # Analyze data in 1-minute windows

def extract_baseline_fhr(window_fhr):
    """Extract baseline FHR (LB)."""
    return np.mean(window_fhr)

def extract_accelerations(window_fhr, avg_fhr):
    """Extract accelerations (AC)."""
    accel = np.sum(window_fhr > avg_fhr + 15)
    return 1 if accel >= 15 else 0

def extract_decelerations(window_fhr, avg_fhr):
    """Extract decelerations (DL)."""
    decel = np.sum(window_fhr < avg_fhr - 15)
    return 1 if decel >= 15 else 0

def extract_uterine_contractions(window_uct):
    """Extract uterine contractions (UC)."""
    return np.count_nonzero(window_uct > np.mean(window_uct) + 10)

def extract_severe_decelerations(window_fhr, avg_fhr):
    """Extract severe decelerations (DS)."""
    severe_decel = np.sum(window_fhr < avg_fhr - 30)
    return 1 if severe_decel >= 30 else 0

def extract_prolonged_decelerations(window_fhr, avg_fhr):
    """Extract prolonged decelerations (DP)."""
    prolonged_decel = np.sum(window_fhr < avg_fhr - 15)
    return 1 if prolonged_decel >= 120 else 0

def extract_repetitive_decelerations(window_fhr, avg_fhr):
    """Extract repetitive decelerations (DR)."""
    rep_decel = np.sum(window_fhr < avg_fhr - 15)
    return 1 if rep_decel >= 5 else 0

def extract_variability(window_fhr):
    """Extract short and long term variability (ASTV, MSTV, ALTV, MLTV)."""
    short_term_var = np.std(np.diff(window_fhr))  # Beat-to-beat variability
    long_term_var = np.std(window_fhr)
    astv = np.mean(np.abs(np.diff(window_fhr)) > 10)
    altv = np.mean(window_fhr > np.mean(window_fhr) + 10)
    return astv, short_term_var, altv, long_term_var

def extract_histogram_features(window_fhr):
    """Extract histogram features (Width, Min, Max, Nmax, Nzeros)."""
    hist_min = np.min(window_fhr)
    hist_max = np.max(window_fhr)
    hist_width = hist_max - hist_min
    
    peaks = np.sum((window_fhr[1:-1] > window_fhr[:-2]) & (window_fhr[1:-1] > window_fhr[2:]))
    
    baseline = np.mean(window_fhr)
    zero_crossings = np.sum(np.diff(np.sign(window_fhr - baseline)) != 0)
    
    return hist_width, hist_min, hist_max, peaks, zero_crossings

def extract_central_tendency(window_fhr):
    """Extract central tendency features (Mode, Mean, Median, Variance, Tendency)."""
    mode_fhr = np.bincount(window_fhr.astype(int)).argmax()
    mean_fhr = np.mean(window_fhr)
    median_fhr = np.median(window_fhr)
    variance_fhr = np.var(window_fhr)
    
    skewness = np.mean((window_fhr - mean_fhr) ** 3)
    tendency_fhr = -1 if skewness < 0 else (1 if skewness > 0 else 0)
    
    return mode_fhr, mean_fhr, median_fhr, variance_fhr, tendency_fhr

def extract_features(digital_input, contraction_wave, window_size=60):
    """Main function to extract features from CTG data."""
    num_windows = len(digital_input) // window_size

    # Initialize feature lists
    baseline_fhr = []
    accelerations = []
    decelerations = []
    severe_decelerations = []
    prolonged_decelerations = []
    repetitive_decelerations = []
    uterine_contractions = []
    astv = []
    mstv = []
    altv = []
    mltv = []
    hist_width = []
    hist_min = []
    hist_max = []
    nmax = []
    nzeros = []
    mode_fhr = []
    mean_fhr = []
    median_fhr = []
    variance_fhr = []
    tendency_fhr = []

    # Feature extraction loop
    for i in range(num_windows):
        window_fhr = digital_input[i * window_size:(i + 1) * window_size]
        window_uct = contraction_wave[i * window_size:(i + 1) * window_size]
        
        avg_fhr = extract_baseline_fhr(window_fhr)
        
        baseline_fhr.append(avg_fhr)
        accelerations.append(extract_accelerations(window_fhr, avg_fhr))
        decelerations.append(extract_decelerations(window_fhr, avg_fhr))
        uterine_contractions.append(extract_uterine_contractions(window_uct))
        severe_decelerations.append(extract_severe_decelerations(window_fhr, avg_fhr))
        prolonged_decelerations.append(extract_prolonged_decelerations(window_fhr, avg_fhr))
        repetitive_decelerations.append(extract_repetitive_decelerations(window_fhr, avg_fhr))
        
        astv_val, mstv_val, altv_val, mltv_val = extract_variability(window_fhr)
        astv.append(astv_val)
        mstv.append(mstv_val)
        altv.append(altv_val)
        mltv.append(mltv_val)
        
        width, min_fhr, max_fhr, n_max, n_zeros = extract_histogram_features(window_fhr)
        hist_width.append(width)
        hist_min.append(min_fhr)
        hist_max.append(max_fhr)
        nmax.append(n_max)
        nzeros.append(n_zeros)
        
        mode, mean, median, var, tendency = extract_central_tendency(window_fhr)
        mode_fhr.append(mode)
        mean_fhr.append(mean)
        median_fhr.append(median)
        variance_fhr.append(var)
        tendency_fhr.append(tendency)
    
    # Combine features into a single feature matrix
    features = np.column_stack((
        baseline_fhr, accelerations, decelerations, severe_decelerations, prolonged_decelerations,
        repetitive_decelerations, uterine_contractions, astv, mstv, altv, mltv, hist_width, hist_min, hist_max,
        nmax, nzeros, mode_fhr, mean_fhr, median_fhr, variance_fhr, tendency_fhr))
    
   
    return features
