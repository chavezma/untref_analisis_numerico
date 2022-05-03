from IPython.display import display, Markdown, Latex
import math
from mpmath import mpf

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
        # display(Latex('$S_{' + str(i) + '}=f(8^{' + str(-i) + '}) = ' + str(f(8**(-i))) + '\implies g(8^{' + str(-i) + '})$ = ' + str(g(8**(-i)))))
        i += 1

def ejercicio_07():
    eps = 2**-1
    n = 1
    while 1 + eps != 1:
        n = n + 1
        eps = eps / 2

    eps = eps / 2

# Ejercicio 8

res_real = mpf(0.0008909545339942893)
res_punto_a = mpf(0)
res_punto_b = mpf(0)
res_punto_c = mpf(0)
res_punto_d = mpf(0)

x = [mpf(2.718281828), mpf(-3.141592654), mpf(1.414213562), mpf(0.5772156649), mpf(0.3010299957)]
y = [mpf(1486.2497), mpf(878366.9879), mpf(-22.37429), mpf(4773714.647), mpf(0.000185049)]

x_por_y = []

x_por_y_ref = [mpf(4040.0455513804516), mpf(-2759471.2767027468866), mpf(-31.64202435812098), mpf(2755462.8740109737903), mpf(5.57052996742893e-5)]

def multiplicar(arr_a, arr_b):
    res = []
    for i in range(len(x)):
        res.append(x[i] * y[i])

    return res

def punto_a(arr):
    valor = mpf(0)
    acum = mpf(0)
    for i in range(len(arr)):
        temp_acum = acum
        valor = arr[i]
        acum += arr[i]
        print(f"iteracion [{i+1}] suma previa [{repr(temp_acum)}], valor a sumar [{repr(valor)}] nuevo valor acumulada [{repr(acum)}] ")

    print(f"resultado final {repr(acum)}")
    return acum

def punto_b(arr):
    acum = mpf(0)
    valor = mpf(0)
    for i in range(len(arr), 0, -1):
        temp_acum = acum
        valor = arr[i - 1]
        acum += arr[i - 1]
        print(f"iteracion [{i+1}] suma previa [{repr(temp_acum)}], valor a sumar [{repr(valor)}] nuevo valor acumulada [{repr(acum)}] ")

    print(f"resultado final {repr(acum)}")
    return acum


def punto_c(arr):
    pos = []
    neg = []
    item = mpf(0)
    for i in range(len(arr)):
        item = arr[i]
        if(item > 0):
            pos.append(item)
        else:
            neg.append(item)

    # Ordeno el array de positivos de Mayor a Menor
    pos_sorted = sorted(pos, reverse=True)

    # Ordeno el array de negativos de Menor a Mayor
    neg_sorted = sorted(neg, reverse=False)

    print(f"pos_sorted {pos_sorted}")
    print(f"neg_sorted {neg_sorted}")

    total = mpf(0)
    valor = mpf(0)
    total_pos = mpf(0)
    total_neg = mpf(0)

    for i in range(len(pos_sorted)):
        temp_acum = total_pos
        valor = pos_sorted[i]
        total_pos += pos_sorted[i]
        print(f"iteracion [{i+1}] suma previa [{repr(temp_acum)}], valor a sumar [{repr(valor)}] nuevo valor acumulada [{repr(total_pos)}] ")

    print("")

    for i in range(len(neg_sorted)):
        temp_acum = total_neg
        valor = neg_sorted[i]
        total_neg += neg_sorted[i]
        print(f"iteracion [{i+1}] suma previa [{repr(temp_acum)}], valor a sumar [{repr(valor)}] nuevo valor acumulada [{repr(total_neg)}] ")

    print(f"ultima operacion {repr(total_pos)} + {repr(total_neg)}")
    total = total_pos + total_neg
    print(f"resultado final {repr(total)}")
    return total

def punto_d(arr):
    pos = []
    neg = []
    item = mpf(0)
    for i in range(len(arr)):
        item = arr[i]
        if(item > 0):
            pos.append(item)
        else:
            neg.append(item)

    # Ordeno el array de positivos de Menor a Mayor
    pos_sorted = sorted(pos, reverse=False)
    # Ordeno el array de negativos de Mayor a Menor
    neg_sorted = sorted(neg, reverse=True)

    print(f"pos_sorted {pos_sorted}")
    print(f"neg_sorted {neg_sorted}")

    total = mpf(0)
    valor = mpf(0)
    total_pos = mpf(0)
    total_neg = mpf(0)

    for i in range(len(pos_sorted)):
        temp_acum = total_pos
        valor = pos_sorted[i]
        total_pos += pos_sorted[i]
        print(f"iteracion [{i+1}] suma previa [{repr(temp_acum)}], valor a sumar [{repr(valor)}] nuevo valor acumulada [{repr(total_pos)}] ")

    print("")

    for i in range(len(neg_sorted)):
        temp_acum = total_neg
        valor = neg_sorted[i]
        total_neg += neg_sorted[i]
        print(f"iteracion [{i+1}] suma previa [{repr(temp_acum)}], valor a sumar [{repr(valor)}] nuevo valor acumulada [{repr(total_neg)}] ")

    print(f"ultima operacion {repr(total_pos)} + {repr(total_neg)}")
    total = total_pos + total_neg
    print(f"resultado final {repr(total)}")
    return total

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


if __name__ == "__main__":
    pass
