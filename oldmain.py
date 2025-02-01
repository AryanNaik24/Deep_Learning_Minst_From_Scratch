import cupy as cp
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Normalize pixel values (0-255) to (0-1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Flatten images (28x28 → 784)
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Convert labels to one-hot encoding
Y_train = to_categorical(Y_train, num_classes=10)
Y_test = to_categorical(Y_test, num_classes=10)

# Move data to GPU
X_train, X_test = cp.array(X_train), cp.array(X_test)
Y_train, Y_test = cp.array(Y_train), cp.array(Y_test)

# Activation Functions
def relu(x):
    return cp.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(cp.float32)

def softmax(x):
    exp_x = cp.exp(x - cp.max(x, axis=1, keepdims=True))  # Stability trick
    return exp_x / cp.sum(exp_x, axis=1, keepdims=True)

# Initialize Weights and Biases
def initialize_weights_biases(layers):
    cp.random.seed(42)
    weights = {}
    for i in range(1, len(layers)):
        weights[f"W{i}"] = cp.random.randn(layers[i-1], layers[i]) * cp.sqrt(2 / layers[i-1])  # He Initialization
        weights[f"b{i}"] = cp.zeros((1, layers[i]))
    return weights

# Forward Propagation
def forward_propagation(X, weights):
    activation = {"A0": X}
    L = len(weights) // 2

    for i in range(1, L):
        Z = cp.dot(activation[f"A{i-1}"], weights[f"W{i}"]) + weights[f"b{i}"]
        A = relu(Z)
        activation[f"Z{i}"], activation[f"A{i}"] = Z, A

    # Output Layer (Softmax Activation)
    Z_final = cp.dot(activation[f"A{L-1}"], weights[f"W{L}"]) + weights[f"b{L}"]
    A_final = softmax(Z_final)
    activation[f"Z{L}"], activation[f"A{L}"] = Z_final, A_final

    return activation

# Compute Loss (Categorical Cross-Entropy)
def compute_loss(Y, A_final):
    return -cp.mean(cp.sum(Y * cp.log(A_final + 1e-8), axis=1))  # Avoid log(0) error

# Backward Propagation
def backward_propagation(X, Y, weights, activation):
    gradients = {}
    L = len(weights) // 2
    m = X.shape[0]

    # Output Layer Gradient
    dZ = activation[f"A{L}"] - Y
    for i in range(L, 0, -1):
        dW = cp.dot(activation[f"A{i-1}"].T, dZ) / m
        db = cp.sum(dZ, axis=0, keepdims=True) / m
        gradients[f"dW{i}"], gradients[f"db{i}"] = dW, db

        if i > 1:
            dZ = cp.dot(dZ, weights[f"W{i}"].T) * relu_derivative(activation[f"Z{i-1}"])

    return gradients

# Update Weights
def update_weights(weights, gradients, learning_rate):
    for key in weights.keys():
        weights[key] -= learning_rate * gradients["d" + key]
    return weights

# Training Function
def train(X, Y, layers, learning_rate=0.01, epochs=5000):
    weights = initialize_weights_biases(layers)
    
    for i in range(epochs):
        activation = forward_propagation(X, weights)
        gradients = backward_propagation(X, Y, weights, activation)
        weights = update_weights(weights, gradients, learning_rate)

        if i % 500 == 0:
            loss = compute_loss(Y, activation[f"A{len(layers) - 1}"])
            print(f"Epoch {i}: Loss = {loss:.4f}")

    return weights

# Prediction Function
def predict(X, weights):
    activation = forward_propagation(X, weights)
    return cp.argmax(activation[f"A{len(weights) // 2}"], axis=1)

# Define Network Architecture
layers = [784, 128, 64, 32, 10]  # Input: 784 (28x28), Hidden: 128-64-32, Output: 10 classes
weights = train(X_train, Y_train, layers, learning_rate=0.01, epochs=5000)

# Evaluate on Test Set
predictions = predict(X_test, weights)
accuracy = cp.mean(predictions == cp.argmax(Y_test, axis=1))
print(f"Test Accuracy: {accuracy * 100:.2f}%")
