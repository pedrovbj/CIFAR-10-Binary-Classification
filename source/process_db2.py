import numpy as np
from util import readData
from sklearn.utils import shuffle

path_in = '../data/dBAlltrSt2.dat'
path_out = '../data/proc_db.dat'

# Carrega a base de dados
f_in = open(path_in, 'r')

X = []
Y = []
for line in f_in:
    tok = line.split(',')
    X.append([int(x) for x in tok[:-9]]+[1])
    # grupo 9 -> ultima coluna
    Y.append(int(tok[-1]))

# Transforma em np.array para manipulacao mais facil
X = np.array(X)
Y = np.array(Y)

## Balanceamento do numero de casos 1
## que na base eh muito menor que os casos com 0
# Todos os X classificados como 0
X0 = X[Y==0, :]
# Todos os X classificados como 1
X1 = X[Y==1, :]

N0 = X0.shape[0]
N1 = X1.shape[0]

# Downsample para o mesmo numero de 1s e 0s, N0 >> N1
X0 = X0[::N0//N1]

# Reconstrui a base com os dados balanceados
X = np.vstack([X0, X1])
Y = np.array([0]*len(X0) + [1]*len(X1))

# Embaralha a base de dados
X, Y = shuffle(X, Y)

# Grava a base de dados no disco
f_out = open(path_out, 'w')
for i, x in enumerate(X):
    tok = [*map(str, x)] # transforma os floats em string
    tok += [str(Y[i])] # adiciona a classificacao
    line = ','.join(tok) # aglutina com separador
    line += '\n'# adiciona a quebra de linha
    f_out.write(line)
f_out.close()
