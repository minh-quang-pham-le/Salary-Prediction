import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import joblib

def load_data(filepath):
    data = pd.read_csv(filepath)
    print('Data Loaded Successfully.')
    return data

def preprocess_data(data):
    # Take the "YearsExperience" column as the feature and "Salary" as the target 
    X = data.iloc[:,0].to_numpy()
    y = data.iloc[:,-1].to_numpy()
    
    # Convert to 2D numpy array
    X = X.reshape(-1,1)
    y = y.reshape(-1,1)
    
    # Split the dataset (80% training, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    print("Data Preprocessing  Completed.")
    
    # Calculate X_range for visualization later
    X_range = np.linspace(np.min(X_train),np.max(X_train),100).reshape(-1,1)
    
    # Scale the data
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    X_range_scaled = scaler_X.transform(X_range)
    
    joblib.dump(scaler_X,'scaler_X') # Save the scaler_X in "scaler_X" file
    
    return X_train, X_test, y_train, y_test, X_range

def poly(X_train, X_test, X_range, degree): # Function to add polynomial feature to the data set
    poly = PolynomialFeatures(degree=degree, include_bias= False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    X_range_poly = poly.transform(X_range)
    
    joblib.dump(poly, 'poly')
    
    return X_train_poly, X_test_poly, X_range_poly 

def scale(X_train, X_test, X_range): # Function to scale the dataset
    scaler_poly = StandardScaler()
    X_train_scaled = scaler_poly.fit_transform(X_train)
    X_test_scaled = scaler_poly.transform(X_test)
    X_range_scaled = scaler_poly.transform(X_range)
    
    joblib.dump(scaler_poly, 'scaler_poly')
    return X_train_scaled, X_test_scaled, X_range_scaled

if __name__ == '__main__':
    data = load_data('Data/Salary.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)