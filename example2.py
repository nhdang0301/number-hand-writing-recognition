import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Hàm kích hoạt và đạo hàm
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def softmax_derivative(softmax_output, Y):
    return softmax_output - Y

# Khởi tạo tham số với khởi tạo He
def initialize_parameters_he(input_size, hidden_sizes, output_size):
    np.random.seed(42)
    weights = [
        np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2. / input_size),
        np.random.randn(hidden_sizes[0], hidden_sizes[1]) * np.sqrt(2. / hidden_sizes[0]),
        np.random.randn(hidden_sizes[1], output_size) * np.sqrt(2. / hidden_sizes[1])
    ]
    biases = [np.zeros((1, size)) for size in hidden_sizes + [output_size]]
    return weights, biases

# Forward propagation
def forward_propagation(X, weights, biases):
    Z1 = X.dot(weights[0]) + biases[0]
    A1 = relu(Z1)
    Z2 = A1.dot(weights[1]) + biases[1]
    A2 = relu(Z2)
    Z3 = A2.dot(weights[2]) + biases[2]
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# Backward propagation
def backward_propagation(X, Y, weights, biases, Z1, A1, Z2, A2, Z3, A3):
    m = X.shape[0]
    dZ3 = softmax_derivative(A3, Y)
    dW3 = (1/m) * A2.T.dot(dZ3)
    db3 = (1/m) * np.sum(dZ3, axis=0, keepdims=True)

    dA2 = dZ3.dot(weights[2].T)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = (1/m) * A1.T.dot(dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2.dot(weights[1].T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1/m) * X.T.dot(dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

# Cập nhật tham số
def update_parameters(weights, biases, dW1, db1, dW2, db2, dW3, db3, learning_rate):
    weights[0] -= learning_rate * dW1
    biases[0] -= learning_rate * db1
    weights[1] -= learning_rate * dW2
    biases[1] -= learning_rate * db2
    weights[2] -= learning_rate * dW3
    biases[2] -= learning_rate * db3
    return weights, biases

# Huấn luyện mạng
def train(X, Y, input_size, hidden_sizes, output_size, epochs, learning_rate):
    weights, biases = initialize_parameters_he(input_size, hidden_sizes, output_size)
    for i in range(epochs):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X, weights, biases)
        dW1, db1, dW2, db2, dW3, db3 = backward_propagation(X, Y, weights, biases, Z1, A1, Z2, A2, Z3, A3)
        weights, biases = update_parameters(weights, biases, dW1, db1, dW2, db2, dW3, db3, learning_rate)
        if i % 100 == 0:
            loss = -np.mean(Y * np.log(A3 + 1e-8))
            print(f"Epoch {i}, Loss: {loss}")
    return weights, biases

# Hàm dự đoán
def predict(X, weights, biases):
    _, _, _, _, _, A3 = forward_propagation(X, weights, biases)
    return np.argmax(A3, axis=1)

# Chuẩn bị dữ liệu
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"Training data shape: {X_train.shape}, Training labels shape: {Y_train.shape}")
print(f"Test data shape: {X_test.shape}, Test labels shape: {Y_test.shape}")
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)

# Chia dữ liệu train và validation
split_index = int(0.8 * X_train.shape[0])
X_train, X_val = X_train[:split_index], X_train[split_index:]
Y_train, Y_val = Y_train[:split_index], Y_train[split_index:]

# Huấn luyện mô hình
input_size = 784
hidden_sizes = [128, 64]
output_size = 10
epochs = 1000
learning_rate = 0.01
weights, biases = train(X_train, Y_train, input_size, hidden_sizes, output_size, epochs, learning_rate)

# Dự đoán trên tập validation
val_predictions = predict(X_val, weights, biases)
val_accuracy = np.mean(val_predictions == np.argmax(Y_val, axis=1))
print(f'Validation Accuracy: {val_accuracy}')

# Dự đoán trên tập test
test_predictions = predict(X_test, weights, biases)
test_accuracy = np.mean(test_predictions == np.argmax(Y_test, axis=1))
print(f'Test Accuracy: {test_accuracy}')