import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LinearRegression

def train_linear_regression(X_train, y_train):
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    print('Linear Regression Model Trained.')
    
    return model

def train_neural_network(X_train, y_train):
    
    tf.random.set_seed(42)
    model = Sequential([
        Dense(units = 2, activation = 'relu'),
        Dense(units = 1, activation = 'linear')
    ])
    model.compile(loss = tf.keras.losses.MeanSquaredError(), 
                  optimizer = tf.keras.optimizers.Adam(learning_rate = 7.50))
    model.fit(X_train, y_train, epochs = 100, verbose = 0)
    print('Neural Network Model Trained.')
    
    return model