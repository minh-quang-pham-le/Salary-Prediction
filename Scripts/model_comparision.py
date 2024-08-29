from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

# Calculate the MSE and R^2 Score
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):

    # Check if model is a neural network then "verbose = 0"
    if isinstance(model, (tf.keras.Model, tf.keras.Sequential)):
        y_train_pred = model.predict(X_train, verbose = 0)
        y_test_pred = model.predict(X_test, verbose = 0)
    else:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
    mse_train = mean_squared_error(y_train, y_train_pred) / 2
    mse_test = mean_squared_error(y_test, y_test_pred) / 2
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    print(f'- {model_name}:\n + Training set: Mean Squared Error: {mse_train}, R^2 Score: {r2_train}\
        \n + Test set    : Mean Squared Error: {mse_test}, R^2 Score: {r2_test}')
    
    return y_train_pred