# import Python Libraries 
import numpy as np 
from matplotlib import pyplot as plt 
  
# Sigmoid Function 
def sigmoid(z): 
    return 1 / (1 + np.exp(-z)) 
  
# Initialising all the weights and biases in the range of between 0 and 1 
w1 = np.random.randn(4, 2) 
w2 = np.random.randn(1, 4) 
b1 = np.random.randn(4, 1)
b2 = np.random.randn(1, 1)
      
params = {"w1" : w1, "b1": b1, 
              "w2" : w2, "b2": b2} 
  
# Forward Propagation 
def forward(X, Y, params): 
    N = X.shape[1] 
    w1 = params["w1"] 
    w2 = params["w2"] 
    b1 = params["b1"] 
    b2 = params["b2"] 
  
    Z1 = np.dot(w1, X) + b1 
    A1 = sigmoid(Z1) 
    Z2 = np.dot(w2, A1) + b2 
    A2 = sigmoid(Z2) 
  
    store = (Z1, A1, w1, b1, Z2, A2, w2, b2) 
    log_sum = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y)) 
    cost = -np.sum(log_sum) / N 
    return cost, store, A2 
  
# Backward Propagation 
def back_prop(X, Y, store): 
    N = X.shape[1] 
    (Z1, A1, w1, b1, Z2, A2, w2, b2) = store 
      
    dZ2 = A2 - Y 
    dw2 = np.dot(dZ2, A1.T) / N 
    db2 = np.sum(dZ2, axis = 1, keepdims = True) / N
      
    dA1 = np.dot(w2.T, dZ2) 
    dZ1 = np.multiply(dA1, A1 * (1- A1)) 
    dw1 = np.dot(dZ1, X.T) / N 
    db1 = np.sum(dZ1, axis = 1, keepdims = True) / N 
      
    grads = {"dZ2": dZ2, "dw2": dw2, "db2": db2, 
                 "dZ1": dZ1, "dw1": dw1, "db1": db1} 
    return grads 
  
# Updating the weights based on the negative grads 
def update_param(params, grads, lr): 
    params["w1"] = params["w1"] - lr * grads["dw1"] 
    params["w2"] = params["w2"] - lr * grads["dw2"] 
    params["b1"] = params["b1"] - lr * grads["db1"] 
    params["b2"] = params["b2"] - lr * grads["db2"] 
    return params 
  
# Model to learn the XOR truth table  
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]) # XOR input 
Y = np.array([[0, 1, 1, 0]]) # XOR output 
  
# Define model params 
epochs = 100000
lr = 0.01
losses = np.zeros((epochs, 1)) 
  
for i in range(epochs): 
    losses[i, 0], store, A2 = forward(X, Y, params) 
    grads = back_prop(X, Y, store) 
    params = update_param(params, grads, lr) 
  
# Evaluating the performance 
plt.figure() 
plt.plot(losses) 
plt.title('LOSS VS EPOCHS')
plt.xlabel("epochs") 
plt.ylabel("Loss value") 
plt.show() 

# Testing 
X = np.array([[1, 1, 0, 0], [1, 1, 1, 1]]) # XOR input 
cost, _, A2 = forward(X, Y, params) 
print(cost)
prediction = (A2 > 0.5) * 1.0
# print(A2) 
print(prediction) 
    



    
