import numpy as np
import matplotlib.pyplot as plt

def cut(t, a=-1, b=1):
    if t <= a:
        return 0.0
    elif t >= b:
        return 1.0
    else:
        return (t-a)/(b-a)
cut = np.vectorize(cut)

class Perceptron():
    def fit(self, X, Y, Xt, Yt, learning_rate=1, epochs=1000, show=False):
        N, D = X.shape
        self.w = np.zeros(D)

        s = []
        for epoch in range(1, epochs+1):
            pY = self.forward(X) # Utiliza funcao Cut
            #pY = np.heaviside(X.dot(self.w), 1) # Utiliza funcao degrau
            self.w += learning_rate*X.T.dot(Y-pY)

            score = self.score(Xt, Yt)
            #print('({}/{}) score: {}'.format(epoch, epochs, score))

            s.append(100*score)

        print('best score:', max(s))
        if show:
            plt.plot(s, linewidth=2, marker='o')
            plt.title('Taxa de acerto em função do número de épocas utilizando Função Cut (Taxa de aprendizado = 1e-6)')
            plt.xlabel('Número de épocas')
            plt.ylabel('Taxa de acerto (%)')
            plt.show()

    def forward(self, X):
        return cut(X.dot(self.w))

    def predict(self, X):
        return np.round(self.forward(X))

    def score(self, X, Y):
        return np.mean(self.predict(X) == Y)

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

if __name__ == '__main__':
    print('Loading data...')
    X, Y = readData('../data/proc_db.dat')

    N = Y.shape[0]
    pct_test = 0.15
    Ntest = round(pct_test*N)
    Xtest, Ytest = X[:Ntest,:], Y[:Ntest]
    Xtrain, Ytrain = X[Ntest:,:], Y[Ntest:]

    model = Perceptron()
    print('Training model...')
    #model.fit(Xtrain, Ytrain, Xtest, Ytest, learning_rate=1e-5, epochs=150, show=True)
    model.fit(Xtrain, Ytrain, Xtest, Ytest, learning_rate=1e-6, epochs=150, show=True)
    #model.fit(Xtrain, Ytrain, Xtest, Ytest, learning_rate=1e-10, epochs=150, show=True)
