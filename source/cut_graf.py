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

x = np.linspace(-1.5, 1.5, 1000)
y = cut(x)

plt.plot(x, y, linewidth=2)
plt.xlabel('x')
plt.ylabel('cut(x)')
plt.title('Gráfico da função cut(x, a=-1, b=1)')
plt.show()
