import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils import resample
from mne.decoding import SlidingEstimator, cross_val_multiscore
import mne

def balance_classes(X: np.ndarray, y: np.ndarray, groups: np.ndarray = None) -> tuple:
    """
    Balances classes by downsampling the majority class(es) to match the minority class.
    
    Args:
        X: Data array (epochs, channels, times)
        y: Labels array (epochs)
        groups: Optional groups array (e.g. subjects) for cross-validation
        
    Returns:
        Balanced X, y, (and groups if provided)
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_count = np.min(class_counts)
    
    X_balanced = []
    y_balanced = []
    groups_balanced = []
    
    for cls in unique_classes:
        idx = np.where(y == cls)[0]
        # Downsample without replacement
        idx_resampled = resample(idx, replace=False, n_samples=min_count, random_state=42)
        
        X_balanced.append(X[idx_resampled])
        y_balanced.append(y[idx_resampled])
        if groups is not None:
            groups_balanced.append(groups[idx_resampled])
            
    X_bal = np.concatenate(X_balanced, axis=0)
    y_bal = np.concatenate(y_balanced, axis=0)
    
    if groups is not None:
        groups_bal = np.concatenate(groups_balanced, axis=0)
        return X_bal, y_bal, groups_bal
        
    return X_bal, y_bal

def compute_time_resolved_decoding(
    X: np.ndarray, 
    y: np.ndarray, 
    groups: np.ndarray = None, 
    balance: bool = True,
    n_jobs: int = 1
) -> np.ndarray:
    """
    Executes a time-resolved MVPA decoding pipeline using Linear Discriminant Analysis (LDA).
    Uses Leave-One-Group-Out cross-validation if groups (e.g. subjects) are provided,
    allowing for true inter-subject decoding.
    
    Args:
        X: Array of shape (n_epochs, n_features, n_times)
        y: 1D array of labels/classes (e.g. 0 for Fast, 1 for Slow)
        groups: 1D array of group identifiers (e.g. subject IDs)
        balance: If True, randomly downsamples classes to ensure perfect balance
        n_jobs: Number of parallel jobs for cross-validation
        
    Returns:
        Array of shape (n_splits, n_times) containing decoding scores.
    """
    
    if balance:
        if groups is not None:
            X, y, groups = balance_classes(X, y, groups)
        else:
            X, y = balance_classes(X, y)
            
    # Initialize the base classifier
    clf = LinearDiscriminantAnalysis()
    
    # Wrap in SlidingEstimator to apply the classifier at every time point independently
    time_decod = SlidingEstimator(clf, n_jobs=n_jobs, scoring='accuracy', verbose=False)
    
    # Set up Cross-Validation
    if groups is not None:
        cv = LeaveOneGroupOut()
        splits = cv.split(X, y, groups=groups)
    else:
        # Fallback if no groups provided (e.g. single subject decoding)
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5)
        splits = cv.split(X, y)
        
    # Run the time-resolved cross-validation
    scores = cross_val_multiscore(
        time_decod, 
        X, 
        y, 
        cv=splits, 
        n_jobs=n_jobs
    )
    
    return scores

def compute_decoding_permutations(
    X: np.ndarray, 
    y: np.ndarray, 
    groups: np.ndarray = None,
    balance: bool = True,
    n_permutations: int = 100,
    n_jobs: int = 1
) -> tuple:
    """
    Runs time-resolved MVPA decoding along with N permutations to calculate a 
    statistical null distribution and Family-Wise Error (FWE) corrected thresholds.
    
    Args:
        X: Array of shape (n_epochs, n_features, n_times)
        y: 1D array of labels/classes
        groups: 1D array of group identifiers
        balance: If True, randomly downsamples classes
        n_permutations: Number of label scrambles to run
        n_jobs: Number of parallel jobs
        
    Returns:
        Tuple of (true_scores, perm_scores, fwe_threshold_95)
    """
    
    # Run the true classification
    true_scores = compute_time_resolved_decoding(X, y, groups, balance, n_jobs)
    
    # Run permutations
    perm_scores = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        p_scores = compute_time_resolved_decoding(X, y_perm, groups, balance, n_jobs)
        # Store the mean across CV splits
        perm_scores.append(np.mean(p_scores, axis=0))
        
    perm_scores = np.array(perm_scores)
    
    # Max statistic across time for Family-Wise Error Correction
    max_perm_dist = np.max(perm_scores, axis=1)
    fwe_threshold_95 = np.percentile(max_perm_dist, 95)
    
    return true_scores, perm_scores, fwe_threshold_95

def compute_spatial_decoding(
    X: np.ndarray, 
    y: np.ndarray, 
    groups: np.ndarray = None, 
    balance: bool = True,
    n_jobs: int = 1
) -> np.ndarray:
    """
    Executes a spatial (searchlight) MVPA decoding pipeline.
    Trains a separate classifier for each feature independently.
    
    Args:
        X: Array of shape (n_epochs, n_features)
        y: 1D array of labels/classes
        groups: 1D array of group identifiers
        
    Returns:
        Array of shape (n_features,) containing mean cross-validation decoding scores per feature.
    """
    if balance:
        if groups is not None:
            X, y, groups = balance_classes(X, y, groups)
        else:
            X, y = balance_classes(X, y)
            
    if groups is not None:
        from sklearn.model_selection import LeaveOneGroupOut
        cv = LeaveOneGroupOut()
        splits = list(cv.split(X, y, groups=groups))
    else:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5)
        splits = list(cv.split(X, y))
        
    from sklearn.model_selection import cross_val_score
    scores = []
    
    # Train univariate classifier for each feature independently
    for f in range(X.shape[1]):
        X_f = X[:, f:f+1] # Shape (epochs, 1)
        clf = LinearDiscriminantAnalysis()
        score = cross_val_score(clf, X_f, y, cv=splits, n_jobs=n_jobs)
        scores.append(np.mean(score))
        
    return np.array(scores)

def compute_spatial_decoding_permutations(
    X: np.ndarray, 
    y: np.ndarray, 
    groups: np.ndarray = None,
    balance: bool = True,
    n_permutations: int = 100,
    n_jobs: int = 1
) -> tuple:
    """
    Runs spatial decoding along with N permutations.
    """
    true_scores = compute_spatial_decoding(X, y, groups, balance, n_jobs)
    
    perm_scores = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        p_scores = compute_spatial_decoding(X, y_perm, groups, balance, n_jobs)
        perm_scores.append(p_scores)
        
    perm_scores = np.array(perm_scores) # Shape (n_permutations, n_features)
    max_perm_dist = np.max(perm_scores, axis=1) # Family-Wise Error corrected
    fwe_threshold_95 = np.percentile(max_perm_dist, 95)
    
    return true_scores, perm_scores, fwe_threshold_95
