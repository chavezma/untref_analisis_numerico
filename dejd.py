import math
import time

import scipy.optimize
import numpy.linalg.linalg as linalg
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import gridspec
from scipy.misc import derivative

from practica_03 import solve_root

# S(t) denotes the firm’s equity price per share
# ν(t) is variance of returns on the firm’s assets, asset volatility
# r(t) is a deterministic risk-free interest rate
# d(t) is a deterministic dividend yield on the firm’s assets.
# B(t) denotes the firm’s total debt per share
# R (0 < R ≤ 1) denotes the recovery rate of firm’s debt

class DEJD:

    def __init__(self, ni_inf, ni_cero, k_v, eta_mas, eta_menos, q_mas, lamda):
        self.method = "brentq"
        self.max_intervalo = 100000
        self.lamda = lamda
        self.eta_mas = eta_mas
        self.eta_menos = eta_menos
        self.q_mas = q_mas
        self.q_menos = 1 - q_mas

        self.K = 30  # no se de donde sale
        self.ni_inf = ni_inf  # initial variance
        self.ni_cero = ni_cero  # initial variance
        self.k_v = k_v  # reversion speed to the long-term mean
        # self.S = 0  # Valor del Stock
        self.root_finding_benchmark = []
        self.conditional_number = []

        # self.B_cero = 65.0  # valor tomado de pagina 17
        self.lamda_coefficients = [
            0,  # Para indexar de 1 a 14
            0.00277778, -6.40277778, 924.05000000, -34597.92777778, 540321.11111111, -4398346.36666666, 21087591.77777770,
            -63944913.04444440, 127597579.55000000, -170137188.08333300, 150327467.03333300, -84592161.50000000,
            27478884.76666660, -3925554.96666666
        ]

    ''' parametros del mercado '''
    def set_empresa(self, nombre, S_cero, d, D_cero, r, R):
        self.nombre = nombre
        self.R = R
        self.S_cero = S_cero
        # self.T0 = T0
        self.d = d  # dividendos para el caso particular General Motors
        self.D_cero = D_cero
        self.r = r  # Tasa libre de Riesgo

    def set_root_finding_properties(self, method, tol, max_iter):
        self.method = method
        self.tolerance = tol
        self.max_iter = max_iter

    def get_tau(self, T):
        """ Integral de la volatilidad entre T y t con t=0
            Formula 7.1 - Pág. 20
        """
        return self.ni_inf * T - ((self.ni_cero - self.ni_inf) / self.k_v) * (math.exp(-self.k_v * T) - 1.0)

    def D(self, T):
        """deterministic barrier D(t)
        """
        return self.R * self.B(T)

    def B(self, T):
        """ # B(t) denotes the firm’s total debt per share
            B_t = B(0)*e^(Integral r(s)-d(s) de 0 a t (con t=0))
            int de 0 a t => ( r(s)-d(s) ) * s => ( r(s)-d(s) )*t - (r(s)-d(s))*0 => (r(s)-d(s))*t
        """
        B_cero = self.D_cero / 0.5
        return B_cero * np.exp((self.r - self.d) * T)

    def get_y(self, T):
        # Primer termino en pizarron dice D(0) pero otros han hecho D(T)
        # return math.log((self.S_cero + self.D(T)) / self.D(T), math.e) - self.get_a(T)
        # return math.log((self.S_cero + self.D_cero) / self.D_cero, math.e) - \
        #    self.get_a(T)

        return np.log((self.S_cero + self.D_cero) / self.D_cero) - \
            self.get_a(T)

    def get_v(self, f_u, tau, y, T):
        """ Inverse Transform using Stehfest:
        """

        factor = (math.log(2.0, math.e) / tau)

        acum = 0.0
        for j in range(1, len(self.lamda_coefficients)):
            acum = acum + self.lamda_coefficients[j] * f_u(j * factor, y, T)

        return factor * acum

    def calcular_raiz(self, f, df, a, b):

        if(self.method == "bisect"):
            raiz = scipy.optimize.bisect(f, a, b, maxiter=self.max_iter, xtol=self.tolerance)

        if(self.method == "brentq"):
            raiz = scipy.optimize.brentq(f, a, b, maxiter=self.max_iter, xtol=self.tolerance)

        if(self.method == "manual"):
            raiz = solve_root(f, df, a, b, times=self.max_iter, xtol=self.tolerance)

        return raiz

    def get_quartic_equation(self, p):
        coeficientes = []

        coef_gr_4 = 0.5 * self.eta_menos * self.eta_mas
        coef_gr_3 = (self.get_mu() * self.eta_menos * self.eta_mas) - (0.5 * (self.eta_menos - self.eta_mas))
        coef_gr_2 = -(0.5 + self.get_mu() * (self.eta_menos - self.eta_mas) + ((p + self.lamda) * self.eta_menos * self.eta_mas))
        coef_gr_1 = -self.get_mu() + (p + self.lamda) * (self.eta_menos - self.eta_mas) - self.lamda * (self.q_mas * self.eta_menos - self.q_menos * self.eta_mas)
        coef_gr_0 = p
        coeficientes = [coef_gr_0, coef_gr_1, coef_gr_2, coef_gr_3, coef_gr_4]

        def f(x):
            total = 0.0
            for i in range(len(coeficientes)):
                total = total + coeficientes[i] * (x ** i)
            return total

        return f

    def get_derived_quartic_equation(self, p):
        coeficientes = []
        coef_gr_3 = 4 * (0.5 * self.eta_menos * self.eta_mas)
        coef_gr_2 = 3 * ((self.get_mu() * self.eta_menos * self.eta_mas) - (0.5 * (self.eta_menos - self.eta_mas)))
        coef_gr_1 = 2 * (-(0.5 + self.get_mu() * (self.eta_menos - self.eta_mas) + ((p + self.lamda) * self.eta_menos * self.eta_mas)))
        coef_gr_0 = 1 * (-self.get_mu() + (p + self.lamda) * (self.eta_menos - self.eta_mas) - self.lamda * (self.q_mas * self.eta_menos - self.q_menos * self.eta_mas))
        coeficientes = [coef_gr_0, coef_gr_1, coef_gr_2, coef_gr_3]

        def df(x):
            total = 0.0
            for i in range(len(coeficientes)):
                total = total + coeficientes[i] * (x ** i)
            return total

        return df

    def get_mu(self):
        return -0.5 - self.lamda * self.get_alpha()

    def get_alpha(self):
        return (self.q_mas / (1 - self.eta_mas)) + (self.q_menos / (1 + self.eta_menos)) - 1

    def get_a(self, T):
        # return math.log((self.D(T) + self.K) / self.D(T), math.e)
        return np.log((self.D(T) + self.K) / self.D(T))

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    def solve_stock_price(self, T):
        """ Fórmula 3.16 Pág. 11
            W(t,S) = ( D(T) + K ) * e^(-integral{t}{T} r(s)ds) * inversa Laplace [U(p,y)]
        """
        self.root_finding_benchmark = []
        self.conditional_number = []
        tau = self.get_tau(T)
        # y = self.get_y(T)
        y = self.get_y(T)

        res = (self.D(T) + self.K) * math.exp(-(self.r * T)) * self.get_v(self.f_u_call, tau, y, T)

        return res

    def f_u_call(self, p, y, T):

        f = self.get_quartic_equation(p)
        df = self.get_derived_quartic_equation(p)

        """Obtenemos las raices del polinomio cuadrático
            Fórmula 3.13 Pág. 10
        """
        t_ini = time.time()

        psi_3 = self.calcular_raiz(f, df, -self.max_intervalo, -(1.0 / self.eta_menos))
        psi_2 = self.calcular_raiz(f, df, -(1.0 / self.eta_menos), 0.0)
        psi_1 = self.calcular_raiz(f, df, 0.0, (1.0 / self.eta_mas))
        psi_0 = self.calcular_raiz(f, df, (1.0 / self.eta_mas), self.max_intervalo)

        t_fin = time.time()
        self.root_finding_benchmark.append(t_fin - t_ini)

        """Armamos sistema de ecuacion para hallar C_0, C_1, C_2 y C_3.
            Fórmula 3.12 Pág. 10
        """

        matriz_a = np.array([
            [1.0, 1.0, -1.0, -1.0],
            [psi_0, psi_1, -psi_2, -psi_3],
            [1.0 / (psi_0 * self.eta_menos + 1.0), 1.0 / (psi_1 * self.eta_menos + 1.0), -1.0 / (psi_2 * self.eta_menos + 1.0), -1.0 / (psi_3 * self.eta_menos + 1.0)],
            [1.0 / (psi_0 * self.eta_mas - 1.0), 1.0 / (psi_1 * self.eta_mas - 1.0), -1.0 / (psi_2 * self.eta_mas - 1.0), -1.0 / (psi_3 * self.eta_mas - 1.0)]])

        self.conditional_number.append(linalg.cond(matriz_a))

        vector_b = np.array([
            0.0,
            1.0 / p,
            (1.0 / (p * (self.eta_menos + 1.0))) - (1.0 / p),
            (1.0 / (p * (self.eta_mas - 1.0))) + (1.0 / p)
        ])

        [C_0, C_1, C_2, C_3] = np.linalg.solve(matriz_a, vector_b)

        """ Calculamos A2_mas y A3_menos
            Fórmula 3.15 Pág. 11
        """
        a = self.get_a(T)
        b = -a
        A2_mas = ((1.0 + psi_2 * self.eta_menos) / (psi_3 - psi_2)) * (((psi_0 - psi_3) * C_0 * math.exp((psi_0 - psi_2) * b)) / (psi_0 * self.eta_menos + 1.0) + ((psi_1 - psi_3) * C_1 * math.exp((psi_1 - psi_2) * b)) / (psi_1 * self.eta_menos + 1.0))
        A3_menos = -((1.0 + psi_3 * self.eta_menos) / (psi_3 - psi_2)) * (((psi_0 - psi_2) * C_0 * math.exp((psi_0 - psi_3) * b)) / (psi_0 * self.eta_menos + 1.0) + ((psi_1 - psi_2) * C_1 * math.exp((psi_1 - psi_3) * b)) / (psi_1 * self.eta_menos + 1.0))

        """ Calculamos finalmente U(p, y)
            Fórmula 3.11 Pág. 10
        """
        if y <= 0:
            res = C_0 * math.exp(psi_0 * y) + C_1 * math.exp(psi_1 * y) + A2_mas * math.exp(psi_2 * y) + A3_menos * math.exp(psi_3 * y)
        else:
            res = C_2 * math.exp(psi_2 * y) + C_3 * math.exp(psi_3 * y) + A2_mas * math.exp(psi_2 * y) + A3_menos * math.exp(psi_3 * y) + (math.exp(y) / p - 1.0 / p)

        return res

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    def solve_survival(self, T):
        """ Fórmula 4.15 Pág. 14
            Q(t,S) = inversa Laplace [U(p,y)]
        """
        self.root_finding_benchmark = []
        self.conditional_number = []
        tau = self.get_tau(T)
        y = math.log((self.S_cero + self.D_cero) / self.D_cero)

        return self.get_v(self.f_u_survival, tau, y, T)

    def f_u_survival(self, p, y, T):
        f = self.get_quartic_equation(p)
        df = self.get_derived_quartic_equation(p)

        """Obtenemos las raices del polinomio cuadrático
            Fórmula 3.13 Pág. 10
        """
        t_ini = time.time()

        psi_3 = self.calcular_raiz(f, df, -self.max_intervalo, -(1.0 / self.eta_menos))
        psi_2 = self.calcular_raiz(f, df, -(1.0 / self.eta_menos), 0.0)

        t_fin = time.time()
        self.root_finding_benchmark.append(t_fin - t_ini)

        A_dos = -(1 / p) * ((psi_3 * (1 + self.eta_menos * psi_2)) / (psi_3 - psi_2))
        A_tres = (1 / p) * ((psi_2 * (1 + self.eta_menos * psi_3)) / (psi_3 - psi_2))

        res = A_dos * np.exp(psi_2 * y) + A_tres * np.exp(psi_3 * y) + (1 / p)

        return res


if __name__ == '__main__':

    gm_model = DEJD(
        ni_inf=0.0151,
        ni_cero=0.0260,
        k_v=1.2433,
        eta_mas=0.0443,
        eta_menos=0.1181,
        q_mas=0.4894,
        lamda=162.5382
    )

    gm_model.set_empresa(
        nombre='General Motors',
        S_cero=25.86,
        d=0.078,
        D_cero=32.5,
        r=0.01,
        R=0.5
    )

    tiempo = [t for t in range(1, 60)]

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(3, 2, width_ratios=[0.2, 0.2])

    subplots = [
        [plt.subplot(gs[0]), plt.subplot(gs[1])],
        [plt.subplot(gs[2]), plt.subplot(gs[3])],
        [plt.subplot(gs[4]), plt.subplot(gs[5])]]

    row = 0

    total_avg_stock = []
    total_avg_prob = []
    total_cond_1 = []
    total_cond_2 = []

    metodo = []
    tolerancia = []
    tipo = []
    tiempo_prom = []
    cond_num = []

    for method in ("manual", "bisect", "brentq"):
        subplots[row][0].axhline(0, color='black')
        subplots[row][0].axvline(0, color='black')

        subplots[row][0].set_ylabel("Call Price")
        subplots[row][0].set_xlabel("Time")
        subplots[row][0].grid()

        subplots[row][1].set_ylabel("Default probability")
        subplots[row][1].set_xlabel("Time")
        subplots[row][1].grid()

        for tol in [(1e-6, "10^{-06}"), (1e-7, "10^{-07}"), (1e-8, "10^{-08}"), (1e-9, "10^{-09}"), (1e-10, "10^{-10}")]:
            gm_model.set_root_finding_properties(method=method, max_iter=10000, tol=tol[0])
            stock_val = []
            prob_val = []
            total_avg_stock = []
            total_avg_prob = []
            for t in tiempo:
                stock_val.append(gm_model.solve_stock_price(t))
                total_avg_stock.extend(gm_model.root_finding_benchmark)
                total_cond_1.extend(gm_model.conditional_number)

                prob_val.append(1 - gm_model.solve_survival(t))
                total_avg_prob.extend(gm_model.root_finding_benchmark)

            subplots[row][0].plot(tiempo, stock_val, label="$" + "xtol = " + tol[1].zfill(2) + "$")
            subplots[row][1].plot(tiempo, prob_val, label="$" + "xtol = " + tol[1].zfill(2) + "$")

            metodo.append(method)
            tolerancia.append(tol[1].zfill(2))
            tipo.append("call")
            print(f"{method} avg gm_model.conditional_number: ", np.average(total_cond_1))
            print(f"{method} max gm_model.conditional_number: ", np.max(total_cond_1))
            print(f"{method} min gm_model.conditional_number: ", np.max(total_cond_1))
            tiempo_prom.append(np.average(total_avg_stock))
            cond_num.append(np.average(total_cond_1))

            metodo.append(method)
            tolerancia.append(tol[1].zfill(2))
            tipo.append("default")
            tiempo_prom.append(np.average(total_avg_prob))
            cond_num.append(0)

        subplots[row][1].legend()
        subplots[row][0].legend()
        row = row + 1

    plt.tight_layout()

    df = pd.DataFrame(
        list(zip(metodo, tipo, tolerancia, tiempo_prom, cond_num)),
        columns=['Metodo', 'Tipo', 'Tolerancia', 'tiempo', 'cond']
    )

    df_p = df.pivot_table(['tiempo', 'cond'], ['Tipo', 'Tolerancia'], 'Metodo')

    print(df_p)
