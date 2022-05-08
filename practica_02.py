import random
import numpy as np
from operator import itemgetter

from pyparsing import java_style_comment
import scipy
import scipy.linalg  # SciPy Linear Algebra Library
from numpy import identity, matmul


def normaInfinitoCerrada(matriz):
    res = []
    cant_filas = len(matriz)
    for i in range(cant_filas):
        res.append(sum(map(abs, matriz[i])))

    return max(res)


def generacionVectoresAleatorios(tamanio_matriz):
    vector_x = np.random.uniform(-1, 1, tamanio_matriz)
    # Cargamos un 1 para que la norma infinito resulte 1, ya que todos seran menores o iguales a 1 (en modulo)
    vector_x[random.randint(0, tamanio_matriz - 1)] = 1
    return vector_x


def normaInfinitoAproximada(matriz, pasadas):
    max_norma = 0
    for r in range(pasadas + 1):
        cant_filas = len(matriz)
        vec_x = generacionVectoresAleatorios(cant_filas)
        vec_b = np.dot(matriz, vec_x)
        norma_infinito = max(map(abs, vec_b))
        max_norma = max(max_norma, norma_infinito)

    print("Norma Infinito de A (formula aproximada)(", pasadas, ") = ", max_norma)


def resolverMatrizTriangularSuperior(matriz, b, p):
    n = len(matriz)
    res = [None for i in range(n)]

    # primer elemento:
    res[n - 1] = b[p[n - 1]] / matriz[p[n - 1]][n - 1]

    # Recorremos la matriz desde abajo hacia arriba, y de izquierda a derecha, comenzando desde la diagonal + 1.
    for i in range(n - 2, -1, -1):
        acum = 0
        for j in range(i + 1, n):
            acum = acum + matriz[p[i]][j] * res[j]

        res[i] = (b[p[i]] - acum) / matriz[p[i]][i]
        # Validación para que el resultado no me dé -0
        if res[i] == -0:
            res[i] = 0.0

    return res


def solve_pivot(matriz, n, j, s, p):
    radios = [(i, p[i], abs(matriz[i][j]) / s[i]) for i in p[j:]]
    pivot = max(radios, key=itemgetter(2))[1]
    p[j], p[pivot] = p[pivot], p[j]
    return pivot


# Matriz cuadrada
def solve_gauss(matriz, b):
    n = len(matriz)
    s = [max(map(abs, row)) for row in matriz]
    p = [i for i in range(n)]

    # recorro por columna, calculo pivot, aplico eliminacion
    for j in range(n - 1):
        #  para la columna j, obtengo que fila es pivot
        pivot = solve_pivot(matriz, n, j, s, p)

        # una vez tengo el pivot, tengo que aplicar la eliminacion para las filas diferentes al pivot
        for i in range(j + 1, n):
            factor = matriz[p[i]][j] / matriz[pivot][j]
            b[p[i]] = b[p[i]] - factor * b[pivot]
            for k in range(n):
                matriz[p[i]][k] = matriz[p[i]][k] - factor * matriz[pivot][k]
            matriz[p[i]][j] = factor
    # Armamos L, U y P segun A
    mp = [[0 for _ in range(n)] for _ in range(n)]
    ml = [[0 for _ in range(n)] for _ in range(n)]
    mu = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        ml[i][i] = 1
        mp[i][p[i]] = 1
        for j in range(0, i):
            ml[i][j] = matriz[p[i]][j]
        for j in range(i, n):
            mu[i][j] = matriz[p[i]][j]

    return matriz, mp, ml, mu, b, p


def solve_richardson(A, b, r):

    mR = identity(len(A)) - A
    vector_x = b

    for i in range(r):
        vector_x = matmul(mR, vector_x) + b
        print("Iteración n°: ", i, "X: ", vector_x)


def solve_jacobi(A, b, r):
    n = len(A)
    acum = 0
    temp = [0.0 for _ in range(len(A))]
    x = [0.0 for _ in range(len(A))]

    print(f"{1} \t {x[0]:.6f} \t {x[1]:.6f} \t {x[2]:.6f}  ")
    for k in range(2, r):
        x = temp
        for i in range(0, n):
            acum = 0
            for j in range(0, n):
                if j != i:
                    acum += A[i][j] * x[j]
                    # print("acum: ", acum)
                temp[i] = (b[i] - acum) / A[i][i]

        print(f"{k} \t {x[0]:.6f} \t {x[1]:.6f} \t {x[2]:.6f}  ")


def solve_gauss_seidel(A, b, r):
    n = len(A)
    acum = 0
    temp = [0.0 for _ in range(len(A))]
    x = [0.0 for _ in range(len(A))]

    print(f"{1} \t {x[0]:.6f} \t {x[1]:.6f} \t {x[2]:.6f}  ")

    for k in range(0, r):
        for i in range(0, n):
            acum = 0
            for j in range(0, n):
                if j != i:
                    acum += A[i][j] * x[j]
                    x[i] = (b[i] - acum) / A[i][i]

        print(f"{k} \t {x[0]:.6f} \t {x[1]:.6f} \t {x[2]:.6f}  ")


if __name__ == "__main__":
    matriz = [[3, 5, 7], [2, 6, 4], [0, 2, 8]]
    print("Norma Infinito Cerrada: ", normaInfinitoCerrada(matriz))

    intentos = [10, 100, 1000, 10000, 10000]
    for i in range(len(intentos)):
        normaInfinitoAproximada(matriz, intentos[i])

    matriz = [[4, -3, 2], [-1, 0, 5], [2, 6, -2]]
    print("Norma Infinito Cerrada: ", normaInfinitoCerrada(matriz))

    intentos = [10, 100, 1000, 10000, 10000]
    for i in range(len(intentos)):
        normaInfinitoAproximada(matriz, intentos[i])

    matriz_1 = [[-1, 1, -4], [2, 2, 0], [3, 3, 2]]
    b = [0, 1, 0.5]
    p = []
    a, mp, ml, mu, b, p = solve_gauss(matriz_1, b)

    print("A => \n", np.matrix(a))
    print("B => \n", np.matrix(b))
    print("P => \n", np.matrix(mp))
    print("L => \n", np.matrix(ml))
    print("U => \n", np.matrix(mu))

    print(resolverMatrizTriangularSuperior(a, b, p))

    print("--------------------------------------")

    matriz_2 = [[1, 6, 0], [2, 1, 0], [0, 2, 1]]
    b = [3, 1, 1]
    p = []
    a, mp, ml, mu, b, p = solve_gauss(matriz_2, b)
    print("A => \n", np.matrix(a))
    print("B => \n", np.matrix(b))
    print("p => \n", np.matrix(p))
    print("P => \n", np.matrix(mp))
    print("L => \n", np.matrix(ml))
    print("U => \n", np.matrix(mu))

    print(resolverMatrizTriangularSuperior(a, b, p))

    permutacion = [0, 2, 1]
    matriz_a = [[4, -3, 2], [0, 0, 2], [0, -1, 5]]
    vector_b = [6, 2, 5]
    print("res: ", resolverMatrizTriangularSuperior(matriz_a, vector_b, permutacion))

    matriz_A = [[1, 1 / 2, 1 / 3],
                [1 / 3, 1, 1 / 2],
                [1 / 2, 1 / 3, 1]]

    vector_B = [11 / 18, 11 / 18, 11 / 18]

    solve_richardson(matriz_A, vector_B, 100)

    matriz_A = [[2, -1, 0], [1, 6, -2], [4, -3, 8]]
    vector_B = [2, -4, 5]
    solve_jacobi(matriz_A, vector_B, 11)
    solve_gauss_seidel(matriz_A, vector_B, 11)
