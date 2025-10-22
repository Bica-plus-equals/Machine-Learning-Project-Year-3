#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def fit(X, weights):
    '''
    Map features onto weights using dot product, (after dimentions matching)

    Parameters: 
        X:ndarray feature vector shape(n_samples, n_features)
        weights:ndarray , shape(n_features,) weights vector
    Returns:    
        Linear combination, dot product (n_samples,) 
    '''
    z = np.dot(X, weights)
    return z

def sigmoid(x):
    '''
    Apply sigmoid activation function to get probabilities.
    
    Parameters: 
    x: ndarray , linear combination, shape (n_samples,)
    
    Returns:
    ndarray , probabilities between 0 and 1
    '''
    return 1 / (1 + np.exp(-x))

def loss(y_pred, y_true):
    '''
    Compute the binary cross-entropy loss between predicted probabilities and true labels.

    Parameters:
        y_pred (numpy.ndarray): Predicted probabilities, shape (n_samples,).
        y_true (numpy.ndarray): True binary labels (0 or 1), shape (n_samples,).

    Returns:
        float: The average binary cross-entropy loss value.
    '''
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def accuracy(y_true, y_pred):
    """
    Calculate the classification accuracy of predictions.

    Parameters:
        y_true (numpy.ndarray): True binary labels (0 or 1), shape (n_samples,).
        y_pred (numpy.ndarray): Predicted binary labels (0 or 1), shape (n_samples,).

    Returns:
        float: The fraction of correctly classified samples.
    """
    accuracy = np.mean(y_true == (y_pred >= 0.5).astype(int))
    print(f'Accuracy: {accuracy}')
    return accuracy

def train_logistic(X_train, y_train, num_classes=3, learning_rate=0.001, epochs=1000):
    """
    Train a logistic regression model for classification using gradient descent optimization.

    Parameters:
        X_train (numpy.ndarray): Training feature data, shape (n_samples, n_features).
        y_train (numpy.ndarray): Training binary labels (0 or 1), shape (n_samples,).
        num_classes: number of classes.
        learning_rate (float): Step size used for gradient descent updates. Default is 0.00001.
        epochs (int): Number of iterations for training. Default is 1000.

    Returns:
        numpy.ndarray: Trained model weights of shape (n_features,).
    """
    num_samples, num_features = X_train.shape
    weights = np.zeros((num_classes, num_features))

    for epoch in range(epochs):
        y_pred = fit(X_train, weights)

        # Compute gradient
        gradient = np.dot(X_train.T, (y_pred - (y_train == np.arange(num_classes)))) / num_samples

        # Update weights
        weights -= learning_rate * gradient.T

        if epoch % 100 == 0:
            loss_value = loss(y_pred, y_train)
            print(f'Epoch {epoch}, Loss: {loss_value}')

    return weights

def predict(X, weights):
    """
    Predict binary labels (0 or 1) for input data using trained logistic regression weights.

    Parameters:
        X (numpy.ndarray): Input feature data, shape (n_samples, n_features).
        weights (numpy.ndarray): Trained model weights, shape (n_features,).

    Returns:
        numpy.ndarray: Predicted binary labels (0 or 1), shape (n_samples,).
    """
    z = fit(X, weights)
    y_pred = sigmoid(z)
    # Convert probabilities to binary outputs
    y_pred_binary = (y_pred >= 0.5).astype(int)
    return y_pred_binary

#Load data file
X, y = pd.read_excel("Galaxy_data_5.xlsx", skiprows=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression for 3 classes
weights = train_logistic(X_train, y_train)

# Make predictions
y_pred_train = np.argmax(fit(X_train, weights), axis=1)
y_pred_test = np.argmax(fit(X_test, weights), axis=1)

# Calculate accuracy
accuracy_train = accuracy(y_train, y_pred_train)
accuracy_test = accuracy(y_test, y_pred_test)

print(f'Training Accuracy: {accuracy_train:.2f}')
print(f'Testing Accuracy: {accuracy_test:.2f}')
