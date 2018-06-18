import numpy as np
import matplotlib.pyplot as plt

def readData(path):
    f = open(path, 'r')

    X = []
    Y = []
    for line in f:
        tok = line.split(',')
        # normaliza para 0.0 < x < 1.0 e adiciona termo de bias
        X.append([int(x)/255.0 for x in tok[:-1]]+[1])
        Y.append(int(tok[-1]))

    return np.array(X), np.array(Y)

class Perceptron():
    def fit(self, X, Y, Xt, Yt, learning_rate=1e-2, epochs=1000, show=False):
        N, D = X.shape
        self.w = np.zeros(D)

        s = []
        for epoch in range(1, epochs+1):
            pY = self.predict(X)
            #pY = self.forward(X)
            #pY = sigmoid(self.forward(X))
            self.w += learning_rate*X.T.dot(Y-pY)

            score = self.score(Xt, Yt)
            #print('({}/{}) score: {}'.format(epoch, epochs, score))

            s.append(score)

        print('best score:', max(s))
        if show:
            plt.plot(s)
            plt.show()

    def forward(self, X):
        return X.dot(self.w)

    def predict(self, X):
        return np.heaviside(self.forward(X), 1)

    def score(self, X, Y):
        return np.mean(self.predict(X) == Y)

print('Loading data...')
X, Y = readData('../data/proc_db.dat')

N = Y.shape[0]
pct_test = 0.20
Ntest = round(pct_test*N)
Xtest, Ytest = X[:Ntest,:], Y[:Ntest]
Xtrain, Ytrain = X[Ntest:,:], Y[Ntest:]

model = Perceptron()
print('Training model...')
model.fit(Xtrain, Ytrain, Xtest, Ytest, learning_rate=1, epochs=40, show=True)
