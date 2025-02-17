from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def normalize_data(data):
    '''
    Applies Min-Max normalization (scales data between 0 and 1).
    
    Parameters:
        data (numpy array): The input data.
    
    Returns:
        Normalized data
    '''
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def standardize_data(data):
    '''
    Applies standardization (zero mean, unit variance).
    
    Parameters:
        data (numpy array): The input data.
    
    Returns:
        Standardized data
    '''
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def preprocess_images(images):
    '''
    Normalizes image data to the range [-1, 1].
    
    Parameters:
        images (numpy array): Image dataset.
    
    Returns:
        Normalized images.
    '''
    return images / 255.0 * 2 - 1
