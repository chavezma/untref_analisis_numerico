import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def df(t, x):
    return np.cos(t) - np.sin(x) + t**2

def df2(t, x):
    return -np.sin(t) - (df(t, x) * np.cos(x)) + (2 * t)

def df3(t, x):
    return -np.cos(t) - (df2(t, x) * np.cos(x)) + ((df(t, x)**2) * np.sin(x)) + 2

def df4(t, x):
    return np.sin(t) + (((df(t, x)**3) - df3(t, x)) * np.cos(x)) + (3 * df(t, x) * df2(t, x) * np.sin(x))

def f(t, x, h):
    return x + h * (df(t, x) + (h / 2) * (df2(t, x) + (h / 3) * (df3(t, x) + ((h / 4) * df4(t, x)))))

def taylor(t_cero, x_cero, paso, iteraciones):
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

if __name__ == '__main__':

    # x, fx = taylor(-1, 3, 0.01, 200)
#
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

    x, fx = runge_kuta_4(2, 1, 3, 0.005, ej3_f)

    cs = CubicSpline(x, fx)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.plot(x, fx, 'o', label='data')
    ax.plot(x, cs(x), label="S")
    ax.legend(loc='upper left', ncol=2)
    plt.show()
