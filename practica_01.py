from IPython.display import display, Markdown, Latex
import math
import platform

def ejercicio_05(valor_referencia, suma_objetivo):

    list_res = []
    for i in range(0, 10):
        epsilon = 2**(-i)
        suma_acumulada = valor_referencia
        numeros_sumas = 1
        n = 1
        while (valor_referencia**n > epsilon):
            numeros_sumas = numeros_sumas + 1
            suma_acumulada = suma_acumulada + (valor_referencia)**n
            n = n + 1

        diff = abs(suma_acumulada - suma_objetivo)

        list_res.append((numeros_sumas, epsilon, diff))

    return list_res

# res = ejercicio_01(0.99, 99)

def ejercicio_06():
    def f(x):
        return math.sqrt(x**2 + 1) - 1

    def g(x):
        return (x**2) / (math.sqrt(x**2 + 1) + 1)

    i = 1
    while(i <= 10):
        display(Latex('$S_{' + str(i) + '}=f(8^{' + str(-i) + '}) = ' + str(f(8**(-i))) + ' \implies g(8^{' + str(-i) + '})$ = ' + str(g(8**(-i)))))
        i += 1

def ejercicio_07():
    eps = 2**-1
    n = 1
    while 1 + eps != 1:
        n = n + 1
        eps = eps / 2

    eps = eps / 2
    display(Latex('$\epsilon_M = 2^{-' + str(n) + '} = ' + str(eps) + '$ para maquina de ' + str(platform.architecture()[0])))

# Ejercicio 8

x = [2.718281828, -3.141592654, 1.414213562, 0.5772156649, 0.3010299957]
y = [1486.2497, 878366.9879, -22.37429, 4773714.647, 0.000185049]

def punto_a():
    x_por_y_a = 0
    for i in range(len(x)):
        valor = x[i] * y[i]
        print(f"iteracion [{i+1}] suma acumulada [{x_por_y_a:+}] valor a sumar [{valor:+}]")
        x_por_y_a += x[i] * y[i]

    print(f"resultado final {x_por_y_a:+}\n")

def punto_b():
    x_por_y_b = 0
    for i in range(len(x), 0, -1):
        valor = x[i - 1] * y[i - 1]
        print(f"iteracion [{-1*(i-6)}] suma acumulada [{x_por_y_b:+}] valor a sumar [{valor:+}]")
        x_por_y_b += x[i - 1] * y[i - 1]
    print(f"resultado final {x_por_y_b:+}\n")

def punto_c():
    pos = []
    neg = []
    for i in range(len(x)):
        x_por_y = x[i] * y[i]
        if(x_por_y > 0):
            pos.append(x_por_y)
        else:
            neg.append(x_por_y)

    pos.sort()  # ordeno creciente
    pos.reverse()  # luego decreciente
    neg.sort()  # ordeno creciente

    xy_list = list(pos) + list(neg)
    print("xy_list: ", xy_list)
    x_por_y_c = 0
    for i in range(len(xy_list)):
        valor = xy_list[i]
        print(f"iteracion [{i+1}] suma acumulada [{x_por_y_c:+}] valor a sumar [{valor:+}]")
        x_por_y_c += xy_list[i]

    print(f"resultado final {x_por_y_c:+}\n")

def punto_d():
    pos = []
    neg = []
    for i in range(len(x)):
        x_por_y = x[i] * y[i]
        if(x_por_y > 0):
            pos.append(x_por_y)
        else:
            neg.append(x_por_y)

    pos.sort()  # ordeno creciente
    neg.sort()  # ordeno creciente
    neg.reverse()  # luego decreciente

    xy_list = list(pos) + list(neg)
    print("xy_list: ", xy_list)
    x_por_y_d = 0
    for i in range(len(xy_list)):
        valor = xy_list[i]
        print(f"iteracion [{i}] suma acumulada [{x_por_y_d:+}] valor a sumar [{valor:+}]")
        x_por_y_d += xy_list[i]

    print(f"resultado final {x_por_y_d:+}\n")

# Ejercicio 9

def ej9_s_n(n):
    return (1 / 3)**n

def ej9_r_n(n):
    if n == 0:
        return 1
    elif n == 1:
        return 1 / 3
    else:
        return (13 / 3) * ej9_r_n(n - 1) - (4 / 3) * ej9_r_n(n - 2)


# Ejercicio 10

def ej10_s_n(n):
    return ((1 - math.sqrt(5)) / 2)**n

def ej10_r_n(n):
    if n == 0:
        return 1
    elif n == 1:
        return (1 - math.sqrt(5)) / 2
    else:
        return ej10_r_n(n - 1) + ej10_r_n(n - 2)

# Ejercicio 11

def y_n(n):
    if n == 0:
        return (math.e - 1)
    elif n == 1:
        return 1
    else:
        return (math.e - n * (y_n(n - 1)))
