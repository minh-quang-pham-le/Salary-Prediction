import joblib
import numpy as np

scaler_X = joblib.load('scaler_X')
scaler_poly = joblib.load('scaler_poly')
poly = joblib.load('poly')

def lr_input(number):
    value = np.array([number]).reshape(-1,1)
    return scaler_X.transform(value)

def pl_input(number):
    value = np.array([number]).reshape(-1,1)
    value_poly = poly.transform(value)
    return scaler_poly.transform(value_poly)

def nn_input(number):
    value = np.array([number]).reshape(-1,1)
    return scaler_X.transform(value)