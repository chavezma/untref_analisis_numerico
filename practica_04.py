import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import scipy.linalg as la

def df(t, x):
    return np.cos(t) - np.sin(x) + t**2

def df2(t, x):
    return -np.sin(t) - (df(t, x) * np.cos(x)) + (2 * t)

def df3(t, x):
    return -np.cos(t) - (df2(t, x) * np.cos(x)) + ((df(t, x)**2) * np.sin(x)) + 2

def df4(t, x):
    return np.sin(t) + (df(t, x)**3 - df3(t, x)) * np.cos(x) + 3 * df(t, x) * df2(t, x) * np.sin(x)

def f(t, x, h):
    return x + h * (df(t, x) + (h / 2) * (df2(t, x) + (h / 3) * (df3(t, x) + (h / 4) * df4(t, x))))

def solve_taylor(t_cero, x_cero, paso, iteraciones):
    lst_f = []
    lst_t = []
    x = x_cero
    t = t_cero

    iter = 0
    while iter < iteraciones:
        iter += 1
        x = f(t, x, paso)
        t = t + paso
        lst_t.append(t)
        lst_f.append(x)

    return lst_t, lst_f

def runge_kuta_4(x_cero, t_ini, t_fin, h, f):
    lst_t = []
    lst_fx = []

    x = x_cero
    t = t_ini

    while t <= t_fin:
        lst_t.append(t)
        lst_fx.append(x)

        f1 = h * f(t, x)
        f2 = h * f(t + (0.5 * h), x + (0.5 * f1))
        f3 = h * f(t + (0.5 * h), x + (0.5 * f2))
        f4 = h * f(t + h, x + f3)

        x = x + ((f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6)
        t = t + h

    return lst_t, lst_fx

def ej3_f(t, x):
    return (1 / np.power(t, 2)) * (t * x - np.power(x, 2))

def ej4_f(x):
    return np.sin(np.pi * x)

def solve_metodo_implicito_homogeneo(f, h, k, tf):

    s = k / np.power(h, 2)
    n = int(1 / h)
    M = int(tf / k)

    list_t = [i * k for i in range(M + 1)]
    list_x = [i * h for i in range(n + 1)]

    U_actual = [f(x) for x in list_x[1:len(list_x) - 1]]
    U = [[0] + U_actual + [0]]

    A = np.array([
        [0] + (n - 2) * [-s],
        (n - 1) * [1 + 2 * s],
        (n - 2) * [-s] + [0]
    ])

    for i in range(M):
        U_next = la.solve_banded((1, 1), A, U_actual)
        U.append([0] + list(U_next) + [0])
        U_actual = U_next

    return U, list_x, list_t

if __name__ == '__main__':

    # x, fx = taylor(-1, 3, 0.01, 200)

    # """ https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
    # """
    # cs = CubicSpline(x, fx)
    # fig, ax = plt.subplots(figsize=(6.5, 4))
    # ax.axhline(0, color='black')
    # ax.axvline(0, color='black')
    # ax.plot(x, fx, 'o', label='data')
    # ax.plot(x, cs(x), label="S")
    # ax.legend(loc='upper left', ncol=2)
    # plt.show()

    # x, fx = runge_kuta_4(2, 1, 3, 0.005, ej3_f)

    # cs = CubicSpline(x, fx)
    # fig, ax = plt.subplots(figsize=(6.5, 4))
    # # ax.axhline(0, color='black')
    # # ax.axvline(0, color='black')
    # ax.plot(x, fx, 'o', label='data')
    # ax.plot(x, cs(x), label="S")
    # ax.legend(loc='upper left', ncol=2)
    # plt.show()

    h = 0.01
    k = 0.01
    t_final = 2
    U, X, T = solve_metodo_implicito_homogeneo(ej4_f, h, k, t_final)

    """ https://matplotlib.org/2.0.2/mpl_toolkits/mplot3d/tutorial.html
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, T = np.meshgrid(X, T)
    Z = np.array(U)
    surf = ax.plot_surface(X, T, Z, cmap="RdPu")
    fig.colorbar(surf, shrink=0.5, aspect=5)
#
    plt.show()
