import matplotlib.pyplot as plt
import numpy as np
import pybamm
from scipy.optimize import curve_fit
import pandas as pd
from scipy.optimize import least_squares

df = pd.read_excel('/Users/mac/Documents/MA/current.xlsx')
df_n = df.to_numpy()
xdata = np.array(df_n).flatten()

df1 = pd.read_excel('/Users/mac/Documents/MA/voltage.xlsx')
df1_n = df1.to_numpy()
ydata = np.array(df1_n).flatten()

t = np.linspace(0.0, 3600.0, len(ydata))

def call_model(xdata, p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15):
    model = pybamm.lithium_ion.DFN()
    geometry = model.default_geometry
    param = model.default_parameter_values
    param.process_geometry(geometry)
    param.process_model(model)
    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 30, var.x_s: 30, var.x_p: 30, var.r_n: 10, var.r_p: 10}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
    timescale = param.evaluate(model.timescale)
    t_end = 0.19 * timescale
    t_eval = np.linspace(0, t_end, len(ydata))
    solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-3)
    solution = solver.solve(
        model, t_eval,
        inputs={"Lower voltage cut-off [V]": p1, "Electrode height [m]": p2, "Ambient temperature [K]": p3,
                "Negative electrode thickness [m]": p4,
                "Initial concentration in negative electrode [mol.m-3]": p5,
                "Electrode width [m]": p6,
                "Maximum concentration in positive electrode [mol.m-3]": p7,
                "Current function [A]": p8,
                "Positive particle radius [m]": p9,
                "Negative electrode active material volume fraction": p10,
                "Initial concentration in positive electrode [mol.m-3]": p11,
                "Negative electrode electrons in reaction": p12,
                "Positive electrode active material volume fraction": p13,
                "Positive electrode diffusivity [m2.s-1]": p14,
                "Negative electrode diffusivity [m2.s-1]": p15}
    )
    return solution["Terminal voltage [V]"].data



popt, pcov = curve_fit(call_model, xdata, ydata)
print(popt)
plt.plot(t, call_model(xdata, *popt), 'r-', label='fit')
plt.plot(t, ydata, 'b-', label='data')

plt.xlabel('time')
plt.ylabel('voltage')
plt.legend()
plt.show()