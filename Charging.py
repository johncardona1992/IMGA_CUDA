import numpy as np
import pandas as pd
from scipy.optimize import newton
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
import math


# known constants
D = 0.01  # diameter of the pipe in meters
L = 1  # length of the pipe in meters
epsilon = 0.00015  # roughness of the pipe in meters
T1 = 298.15  # temperature in K (25C)
T2 = 298.15  # temperature in K (25C)
m2max = 1128.766  # tube trailer max capacity in kg
Qmax = 41.667/3600  # max output flow in kg/s at the storage
Imax = 41.667/3600  # max input flow in kg/s at the storage
Tenv = 298.15  # ambient temperature in K
Tmax = 333  # max temperature in the tube trailer in K
Trestart = 323  # temperature to restart filling the tube trailer in K
k = 0.01  # heat transfer coefficient
V_tank1 = 17500e-03  # volume of tank 1 in m3
V_tank2 = 45150e-03  # volume of tank 2 in m3
p1 = 500 * 1e5  # initial pressure in tank 1 in Pa (500 bar)
p2 = 40 * 1e5  # initial pressure in tank 2 in Pa (40 bar)
p2max = 381 * 1e5  # max pressure in the tube trailer
dt = 1  # time step in seconds
molar_mass_H2 = PropsSI('molar_mass', 'H2')
R = PropsSI('gas_constant', 'Hydrogen')


def zFactor(temp, pressure):
    rho = PropsSI('D', 'T', temp, 'P', pressure, 'Hydrogen')
    # Calcular Z
    return pressure * molar_mass_H2 / (rho * R * temp)


z1 = zFactor(T1, p1)
z2 = zFactor(T2, p2)
# initial amounts of H2 in each tank (kg)
m1 = p1 * V_tank1*molar_mass_H2 / (R * T1*z1)
m2 = p2 * V_tank2*molar_mass_H2 / (R * T2*z2)

# cross-sectional area of the pipe
A = np.pi * (D/2)**2

# defining the equation to solve


def equation(V, D, rho, epsilon, L, delta_p):
    mu = PropsSI('V', 'T', T1, 'P', p1, 'Hydrogen')
    Re = rho * V * D / mu
    f = 0.25 / (np.log10(epsilon/(3.7*D) + 5.74/(Re**0.9)))**2
    return V - np.sqrt((2 * delta_p * D) / (f * L * rho))


# initial guess for V
V0 = 100
# Qmax=Imax
t = 0

# Lists to store values for plotting
p1_list = []
p2_list = []
m1_list = []
m2_list = []
T2_list = []
Q_list = []
t_list = []

while True:
    # calculate density of H2 in the storage
    rho1 = m1 / V_tank1

    # calculate pressure difference between Storage and tube trailer
    delta_p = p1 - p2
    if delta_p/1e5 <= 2:
        Qreal = 0
    else:
        # solve for V using the Newton-Raphson method
        V = newton(equation, V0, args=(D, rho1, epsilon, L, delta_p))
        # calculate flow rate
        Qreal = V * A * rho1

    V0 = V

    Q = min(Qreal, Qmax)
    # update amounts of H2 in each tank
    dm = Q * dt
    m1 = m1 - dm + Imax
    m2 += dm

    # update pressure in the storage
    p1 = m1*(R * T1 * z1)/(V_tank1*molar_mass_H2)

    # update temperature in the tube trailer

    # 1) Newton's law of cooling
    T2 = Tenv + (T2-Tenv)*math.exp(-k*dt)

    # 2) Joule Thomson effect
    # calculate Joule-Thomson coefficient
    jouleThomson = PropsSI('d(T)/d(P)|H', 'T', T2, 'P', p2, 'Hydrogen')
    dtemp = jouleThomson*(p2 - p1)
    proportion = dm/m2
    T2 = T2*(1-proportion) + (T2+dtemp)*proportion

    # if the max temperature is reached, pause filling the tube trailer until the temperature decrease to restart point
    if T2 > Tmax:
        t += (-1)*math.log((Trestart-Tenv)/(T2-Tenv))/k
        T2 = Trestart

    # update pressure in the tube trailer
    p2 = m2*(R * T2*z2)/(V_tank2*molar_mass_H2)

    # update z factor
    z1 = zFactor(T1, p1)
    z2 = zFactor(T2, p2)

    # print results
    t += 1
    # print(f"t = {t} s, p1 = {p1/1e5} bar, p2 = {p2/1e5} bar, m1 = {m1} kg, m2 = {m2} kg, T2 = {T2} K")

    # store values in lists
    t_list.append(t)
    p1_list.append(p1)
    p2_list.append(p2)
    m1_list.append(m1)
    m2_list.append(m2)
    T2_list.append(T2)
    Q_list.append(dm)

    # stop if the tube trailer is full
    if m2 >= m2max:
        break

# convert lists to arrays for plotting
t = np.array(t_list)
p1 = np.array(p1_list)
p2 = np.array(p2_list)
m1 = np.array(m1_list)
m2 = np.array(m2_list)
T2 = np.array(T2_list)
Q = np.array(Q_list)

# Create a dictionary where keys will be the column names and values will be the data in these columns
data = {'t': t, 'p1': p1, 'p2': p2, 'm1': m1, 'm2': m2, 'T2': T2, 'Q': Q}

# Create a DataFrame
df = pd.DataFrame(data)

# Export to Excel
df.to_excel(f'cargue_data.xlsx', index=False, engine='openpyxl')

# plot p1 and p2 vs t
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(t, p1/1e5, label='p1')
plt.plot(t, p2/1e5, label='p2')
plt.xlabel('Time (s)')
plt.ylabel('Pressure (bar)')
plt.legend()

# plot m1 and m2 vs t
plt.subplot(2, 2, 2)
plt.plot(t, m1, label='m1')
plt.plot(t, m2, label='m2')
plt.xlabel('Time (s)')
plt.ylabel('mass (kg)')
plt.legend()

# plot T2 vs t
plt.subplot(2, 2, 3)
plt.plot(t, T2)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')

# plot Q vs t
plt.subplot(2, 2, 4)
plt.plot(t, Q)
plt.xlabel('Time (s)')
plt.ylabel('Flow rate (kg/s)')

plt.tight_layout()
plt.show()
