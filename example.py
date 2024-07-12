import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Hàm kích hoạt
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

# Derivatives
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Forward propagation
def forward_propagation(X, weights, biases):
    Z1 = np.dot(X, weights[0]) + biases[0]
    A1 = relu(Z1)
    Z2 = np.dot(A1, weights[1]) + biases[1]
    A2 = relu(Z2)
    Z3 = np.dot(A2, weights[2]) + biases[2]
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# Backward propagation
def backward_propagation(X, Y, Z1, A1, Z2, A2, Z3, A3, weights, biases, learning_rate):
    m = X.shape[0]
    dZ3 = A3 - Y
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    dA2 = np.dot(dZ3, weights[2].T)
    dZ2 = dA2 * relu_derivative(A2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, weights[1].T)
    dZ1 = dA1 * relu_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    weights[2] -= learning_rate * dW3
    biases[2] -= learning_rate * db3
    weights[1] -= learning_rate * dW2
    biases[1] -= learning_rate * db2
    weights[0] -= learning_rate * dW1
    biases[0] -= learning_rate * db1

    return weights, biases

# Khởi tạo tham số
def initialize_parameters(input_size, hidden_sizes, output_size):
    np.random.seed(42)
    weights = [
        np.random.randn(input_size, hidden_sizes[0]) * 0.01,
        np.random.randn(hidden_sizes[0], hidden_sizes[1]) * 0.01,
        np.random.randn(hidden_sizes[1], output_size) * 0.01
    ]
    biases = [
        np.zeros((1, hidden_sizes[0])),
        np.zeros((1, hidden_sizes[1])),
        np.zeros((1, output_size))
    ]
    return weights, biases

# Huấn luyện mạng
def train(X, Y, input_size, hidden_sizes, output_size, epochs, learning_rate):
    weights, biases = initialize_parameters(input_size, hidden_sizes, output_size)
    for i in range(epochs):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X, weights, biases)
        weights, biases = backward_propagation(X, Y, Z1, A1, Z2, A2, Z3, A3, weights, biases, learning_rate)
        if i % 100 == 0:
            loss = -np.mean(np.sum(Y * np.log(A3), axis=1))
            print(f'Epoch {i}, Loss: {loss}')
    return weights, biases

# Hàm dự đoán
def predict(X, weights, biases):
    _, _, _, _, _, A3 = forward_propagation(X, weights, biases)
    return np.argmax(A3, axis=1)

# Load dữ liệu MNIST
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28*28).astype('float32') / 255
X_test = X_test.reshape(-1, 28*28).astype('float32') / 255
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

# Chia dữ liệu train và validation
split_index = int(0.8 * X_train.shape[0])
X_train, X_val = X_train[:split_index], X_train[split_index:]
Y_train, Y_val = Y_train[:split_index], Y_train[split_index:]

# Định nghĩa các tham số mạng
input_size = 784 # Kích thước của ảnh MNIST là 28x28
hidden_sizes = [512, 256] # Tăng số lượng nơ-ron trong các lớp ẩn
output_size = 10 # 10 lớp cho 10 chữ số

# Huấn luyện
weights, biases = train(X_train, Y_train, input_size, hidden_sizes, output_size, epochs=1000, learning_rate=0.01)

# Dự đoán trên tập validation
val_predictions = predict(X_val, weights, biases)
val_accuracy = np.mean(val_predictions == np.argmax(Y_val, axis=1))
print(f'Validation Accuracy: {val_accuracy}')

# Dự đoán trên tập test
test_predictions = predict(X_test, weights, biases)
test_accuracy = np.mean(test_predictions == np.argmax(Y_test, axis=1))
print(f'Test Accuracy: {test_accuracy}')
