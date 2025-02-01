# Deep Learning MNIST From Scratch

This project implements a simple deep learning model from scratch using **CuPy** for GPU acceleration. The model is trained on the MNIST dataset, which consists of handwritten digits, and uses a fully connected neural network with ReLU and Softmax activation functions.

## Features
- Uses **CuPy** for fast computations on the GPU.
- Implements a **fully connected neural network**.
- Uses **He Initialization** for better weight initialization.
- Implements **Forward Propagation, Backward Propagation, and Weight Updates** manually.
- Uses **Categorical Cross-Entropy Loss** for multi-class classification.
- Achieves high accuracy on MNIST dataset.

## Prerequisites
Make sure you have the following installed:
- Python 3.x
- TensorFlow
- CuPy (for GPU acceleration)

### Install Dependencies
To install the required dependencies, run:
```sh
pip install tensorflow cupy
```

## Dataset
The project uses the **MNIST dataset**, which is automatically downloaded using TensorFlow:
```python
from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
```

## Neural Network Architecture
The model consists of the following layers:
- Input layer: 784 neurons (28x28 images flattened)
- Hidden layers: 128, 64, 32 neurons (ReLU activation)
- Output layer: 10 neurons (Softmax activation)

## Training the Model
To train the model, run:
```python
layers = [784, 128, 64, 32, 10]
weights = train(X_train, Y_train, layers, learning_rate=0.01, epochs=5000)
```

## Testing and Accuracy
To evaluate the model on the test set:
```python
predictions = predict(X_test, weights)
accuracy = cp.mean(predictions == cp.argmax(Y_test, axis=1))
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

## Functions Implemented
### Activation Functions
- **ReLU** (Rectified Linear Unit)
- **Softmax** (for multi-class classification)

### Neural Network Operations
- **Forward Propagation**
- **Loss Computation** (Categorical Cross-Entropy)
- **Backward Propagation**
- **Weight Updates** (Gradient Descent)

## Example Output
```sh
Epoch 0: Loss = 2.3012
Epoch 500: Loss = 0.3456
Epoch 1000: Loss = 0.1984
...
Test Accuracy: 98.5%
```

## Future Improvements
- Implement additional optimization techniques (Adam, RMSprop).
- Add convolutional layers for better image classification.
- Test on other datasets like Fashion MNIST.

## License
This project is open-source

