import numpy as np
import pandas as pd
import matplotlib as plt

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases for hidden layer
        self.weights_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros(hidden_size)

        # Initialize weights and biases for output layer
        self.weights_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros(output_size)

    def forward(self, X):
        # Forward pass through the network
        # Input to hidden layer
        z_hidden = np.dot(X, self.weights_hidden) + self.bias_hidden
        a_hidden = self.sigmoid(z_hidden)

        # Hidden to output layer
        z_output = np.dot(a_hidden, self.weights_output) + self.bias_output
        a_output = self.softmax(z_output)
        return a_output

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        # Softmax activation function
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def train(self, X, y, learning_rate=0.01, epochs=100):
        # One-hot encode the target labels
        y_onehot = np.eye(self.output_size)[y]

        # Training loop
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Backpropagation
            delta_output = output - y_onehot
            delta_hidden = np.dot(delta_output, self.weights_output.T) * (output * (1 - output))

            # Update weights and biases
            self.weights_output -= learning_rate * np.dot(output.T, delta_output)
            self.bias_output -= learning_rate * np.sum(delta_output, axis=0)
            self.weights_hidden -= learning_rate * np.dot(X.T, delta_hidden)
            self.bias_hidden -= learning_rate * np.sum(delta_hidden, axis=0)

            # Print loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                loss = self.cross_entropy(output, y_onehot)
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

    def cross_entropy(self, y_pred, y_true):
        # Cross-entropy loss
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

class NeuralNetwork:
    # Code for the NeuralNetwork class as provided

    def train_and_evaluate_nn(num_nodes):
        MyNN = NeuralNetwork(9, num_nodes, 3)  # Creating NN with specified number of nodes
        MyNN.train(X_train, y_train, learning_rate=0.00001, epochs=100)

        predictions = np.argmax(MyNN.forward(X_test), axis=1)
        accuracy = np.mean(predictions == y_test)
        return accuracy

#Load data file
data = pd.read_excel("Galaxy_data_5.xlsx", skiprows=0)
split_index = int(len(data) * 0.8) 

train_data = data[:split_index] 
test_data = data[split_index:] #getting np array form of variables 

X_train = train_data[['P_EL','P_CW','P_ACW','P_EDGE','P_DK','P_MG','P_CS','P_EL_DEBIASED','P_CS_DEBIASED']] 
y_train = train_data['label']

X_test = test_data[['P_EL','P_CW','P_ACW','P_EDGE','P_DK','P_MG','P_CS','P_EL_DEBIASED','P_CS_DEBIASED']] 
y_test = test_data['label']


MyNN = NeuralNetwork(9,3,3) 
MyNN.train(X_train, y_train, learning_rate= 0.00001, epochs = 100)
#The mlp returns a 3d probability vector, the index of the point with highest 
# #probability corresponds with the class label {0,1,2} 

##Note dependence testing

# Define range of nodes for testing
num_nodes_list = [5, 10, 15, 20, 25]

# Train NN with different number of nodes and collect accuracies
accuracies = [NeuralNetwork.train_and_evaluate_nn(num_nodes) for num_nodes in num_nodes_list]

# Plot accuracy vs. number of nodes in log scale
plt.semilogx(num_nodes_list, accuracies, marker='o', label='Accuracy')

# Fit a line of best fit (simple linear regression)
slope, intercept = np.polyfit(np.log(num_nodes_list), accuracies, 1)
plt.semilogx(num_nodes_list, intercept + slope * np.log(np.array(num_nodes_list)), linestyle='--', label='Line of Best Fit')

plt.title('Accuracy vs. Number of Nodes in Hidden Layer (Log Scale)')
plt.xlabel('Number of Nodes')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

##Learning rates analysis
# Define range of learning rates for testing
learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]

# Train NN with different learning rates and collect accuracies
accuracies = [NeuralNetwork.train_and_evaluate_nn(learning_rate) for learning_rate in learning_rates]

# Plot accuracy vs. learning rate in log scale
plt.semilogx(learning_rates, accuracies, marker='o', label='Accuracy')

plt.title('Accuracy vs. Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()