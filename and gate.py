import numpy as np
class Perceptron(object):
    """Implements a perceptron network"""
    def activation_fn(self, x):
        return 1 if x >= 0 else 0
    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a
    def __init__(self, input_size, lr=1, epochs=10):
        self.W = np.zeros(input_size+1)
        # add one for bias
        self.epochs = epochs
        self.lr = lr
    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                y = self.predict(X[i])
                e = d[i] - y
                self.W = self.W + self.lr * e * np.insert(X[i], 0, 1)
if __name__ == '__main__':
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    d = np.array([0, 0, 0, 1])
    perceptron = Perceptron(input_size=2)
    perceptron.fit(X, d)
    print("b=",perceptron.W[0])
    print("w1=", perceptron.W[1])
    print("w2=", perceptron.W[2])
    for i in range(4):
        print ("case no ",i+1)
        print ("inputs are :",*X[i])
        print("predicted output :",perceptron.predict(X[i]))
        print ("real value is :",d[i])
