
MAX_DEEP = 5

def buscar_rango_valido(f, a, b, deep, rama="Inicial"):
    """ A partir de un rango [a, b] valida que el mismo cumpla la condicion
         de que f(a) * f(b) < 0 o algun flag de salida se cumpla
    """
    if deep >= MAX_DEEP or b - a < 0.01:
        return None

    f_x_a = f(a)
    f_x_b = f(b)

    bandera = f_x_a * f_x_b

    if bandera < 0:
        return (a, b)

    mid = (a + b) / 2

    res1 = buscar_rango_valido(f, a, mid, deep + 1, f"izq_{deep}")
    res2 = buscar_rango_valido(f, mid, b, deep + 1, f"der_{deep}")

    return res1 if not None else res2

def biseccion(f, a, b):
    """ Dada una funcion f y un rango [a, b] que cumpla que f(a) * f(b) < 0
        parte el rango a la mitad y retorna el nuevo rango tal que se sigue cumpliendo f(a) * f(b) < 0
    """
    pivot = (a + b) / 2
    f_x_a = f(a)
    f_x_p = f(pivot)

    bandera_izq = f_x_a * f_x_p

    if bandera_izq < 0:
        return (a, pivot)
    else:
        return (pivot, b)

def newton_raphson(f, df, x0):
    f_x_x0 = f(x0)
    df_x_x0 = df(x0)

    x_nr = x0 - (f_x_x0 / df_x_x0)

    return x_nr


def solve_root(f, df, ra, rb, times, xtol):
    """ Dada una funcion y su derivada, un rango y una tolerancia
        1) si f(ra) * f(rb) > 0 entonces llama a buscar_rango_valido hasta encontrar un rango que cumpla la hipotesis.
        2) Luego busca como candidato de raiz segun el algoritmo de Newton Raphson
        3) Si el candidato esta dentro del rango, evaluamos el candidato
            si cumple la tolerancia, retornamos solucion
            sino, se actualiza rango cumpliendo las hipotesis
        4) Si el candidato no esta dentro del rango, entonces se aplica biseccion para achicar y volver a evaluar.
    """
    temp_ra = ra
    temp_rb = rb

    (temp_ra, temp_rb) = buscar_rango_valido(f, ra, rb, 0)

    if temp_ra is None and temp_rb is None:
        return None

    candidato = (temp_ra + temp_rb) / 2

    cur_time = 0
    while cur_time < times:
        cur_time += 1
        candidato = newton_raphson(f, df, candidato)

        if not (temp_ra <= candidato <= temp_rb):
            (temp_ra, temp_rb) = biseccion(f, temp_ra, temp_rb)
        else:
            f_x_candidato = f(candidato)

            if f_x_candidato == 0.0:
                return candidato
            elif abs(f_x_candidato) < xtol:
                return candidato

            f_ra = f(temp_ra)

            if (f_ra * f_x_candidato) < 0:
                temp_rb = candidato
            else:
                temp_ra = candidato


if __name__ == '__main__':
    a = -10
    b = 10
    x0 = (a + b) / 2

    def f(x):
        return x**3 + 4 * x**2 + x - 5

    def df(x):
        return 3 * x**2 + 8 * x + 1

    print("===" * 30)
    print("===" * 30)
    for (a, b) in [(-4, -3), (-2, -1), (0, 2)]:
        root = solve_root(f, df, a, b, 10, xtol=1e-14)
        print(f"resultado a={a} b={b} root={root} f({root})={f(root)}")
