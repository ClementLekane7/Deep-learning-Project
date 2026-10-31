import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score


def initialization(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)


# Function to define the model (perceptron) we will use
def model(X, W, b):
    Z = X @ W + b
    A = 1 / (1 + np.exp(-Z))
    return A


# Function to measure the loss (protected against log(0))
def log_loss(A, y):
    m = len(y)
    eps = 1e-15
    A = np.clip(A, eps, 1 - eps)
    return (-1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))


# Function to calculate gradient descent
def gradients(A, X, y):
    dw = (1 / len(y)) * ((X.T) @ (A - y))
    db = (1 / len(y)) * np.sum(A - y)
    return (dw, db)


# Function to make predictions
def predict(X, W, b):
    A = model(X, W, b)
    return A >= 0.5


# Function to optimize the weights and bias
def optimisation(dw, db, W, b, learning_rate):
    W = W - learning_rate * dw
    b = b - learning_rate * db
    return (W, b)


# Main machine learning function gathering all the functions above
def artificial_neuron(X, y, X_test,Y_test,learning_rate = 0.1, nb_iteration = 100):
    W, b = initialization(X)
    loss_value = []
    loss_test_value = []
    accuracy_trainset = []
    accuracy_testset = []
    for i in tqdm(range(nb_iteration), desc ="Entertainment..."):
        A = model(X, W, b)
        A_test = model(X_test, W, b)
        dw, db = gradients(A, X, y)
        W, b = optimisation(dw, db, W, b, learning_rate)
        if(i % 10 == 0):

            # TRAIN 
            # Calculus of the train_loss function for the plot
            Loss = log_loss(A, y)
            loss_value.append(Loss)
            # Calculus of the prediction on the trainsetfor the plot
            y_predict = predict(X, W, b)
            Acc = accuracy_score(y, y_predict)
            accuracy_trainset.append(Acc)

            # TEST 
            # Calculus of the test_loss function for the plot
            Loss_test = log_loss(A_test, Y_test)
            loss_test_value.append(Loss_test)
            # Calculus of the prediction on the testset for the plot
            y_predict = predict(X_test, W, b)
            Acc_test = accuracy_score(Y_test, y_predict)
            accuracy_testset.append(Acc_test)
            
        time.sleep(0.01)
    plt.figure(figsize = (15, 5))
    plt.subplot(1,2,1)
    plt.plot(loss_value, label = 'Loss_train function evolution')
    plt.plot(loss_test_value, label = 'Loss_test function evolution')
    plt.subplot(1,2,2)
    plt.plot(accuracy_testset, label = 'accuracy_testset evolution')
    plt.plot(accuracy_trainset, label = 'accuracy_trainset evolution')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return W, b
