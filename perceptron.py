import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

# Two layer Perceptron implemented using Duda's Pattern Classification book notation
class Perceptron:
    def __init__(self, n_features, n_outputs, n_units_hidden_layer=3, initizalize_weights_between=(0,1)):
        low = initizalize_weights_between[0]
        high = initizalize_weights_between[1]
        self.wji = np.random.uniform(low=low, high=high, size=(n_units_hidden_layer, n_features)) # input to hidden layer weights
        self.wj0 = np.random.uniform(low=low, high=high, size=(n_units_hidden_layer, 1)) # input to hidden layer bias
        self.wkj = np.random.uniform(low=low, high=high, size=(n_outputs, n_units_hidden_layer))  # hidden layer to output weights
        self.wk0 = np.random.uniform(low=low, high=high, size=(n_outputs, 1)) # hidden layer to output bias
    
    def activation(self, X):
        return np.tanh(X)
    
    def activation_derivative(self, X):
        tanh = self.activation(X)
        return 1 - np.multiply(tanh,tanh)
    
    def forward(self, X):
        net_j = np.dot(self.wji, X) + self.wj0
        yj = self.activation(net_j)
        net_k = np.dot(self.wkj, yj) + self.wk0
        zk = self.activation(net_k)
        return net_j, yj, net_k, zk
    
    def gradient_descent(self, x, y, net_j, yj, net_k, zk):
        sensitivity_k = np.array(np.multiply((y - zk), self.activation_derivative(net_k)))
        dwkj = np.dot(sensitivity_k, yj.T)
        dwk0 = np.sum(sensitivity_k, axis=1)
        sensitivity_j = np.multiply(self.activation_derivative(net_j), np.dot(self.wkj.T, sensitivity_k))
        dwj0 = np.sum(sensitivity_j, axis=1)
        dwji = np.dot(sensitivity_j, x.T)
        return dwkj, dwk0[:, np.newaxis], dwji, dwj0[:, np.newaxis]

    def train(self, X, Y, learning_rate=0.001, epochs=300, batch_size=100):
        assert X.shape[1] == Y.shape[1]
        n_samples = X.shape[1]
        for i in range(epochs):
            for j, index in enumerate(range(0, n_samples, batch_size)):
                x_batch = X[:, index:min(index+batch_size, n_samples)]
                y_batch = Y[:, index:min(index+batch_size, n_samples)]
                net_j, yj, net_k, zk = self.forward(x_batch)
                dwkj, dwk0, dwji, dwj0 = self.gradient_descent(x_batch, y_batch, net_j, yj, net_k, zk)
                self.wji += learning_rate * dwji
                self.wj0 += learning_rate * dwj0
                self.wkj += learning_rate * dwkj
                self.wk0 += learning_rate * dwk0
        return self.wji, self.wj0, self.wkj, self.wk0,

    def predict(self, X):
        _, _, _, z = self.forward(X)
        return np.argmax(z, 0)

    def get_accuracy(self, X, Y):
        predictions = self.predict(X)
        return np.sum(predictions == Y) / Y.size

def plot_decision_boundary(X, Y, make_predictions):
    h = .001
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    Z = make_predictions(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)
    plt.scatter(X[0, :], X[1, :], c=np.where(Y > 0, Y, 0), cmap=plt.cm.Paired)
    plt.show()

if __name__ == '__main__':
    X_train = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]).T
    Y_train = np.array([[1, -1], [-1, 1], [-1, 1], [1, -1]]).T
    n_classes = len(np.unique(Y_train))
    n_samples = X_train.shape[1]
    nn = Perceptron(X_train.shape[0], n_classes, n_units_hidden_layer=3, initizalize_weights_between=(-1, 1))   
    nn.train(X_train, Y_train, learning_rate=0.1, epochs=100, batch_size=n_samples)
    print(f'Accuracy: {nn.get_accuracy(X_train, np.argmax(Y_train, axis=0))}')
    plot_decision_boundary(X_train, Y_train[1], nn.predict)


