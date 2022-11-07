import math
import numpy as np
from lmfit import Parameters, fit_report, minimize
import pandas as pd
import pybamm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_excel('/Users/mac/Documents/MA/current.xlsx')
df_n = df.to_numpy()
xdata = np.array(df_n).flatten()

df1 = pd.read_excel('/Users/mac/Documents/MA/ocvvoltage.xlsx')
df1_n = df1.to_numpy()
ydata = np.array(df1_n).flatten()

t = np.linspace(0.0, 3600.0, len(ydata))

params = Parameters()
params.add_many(
                ('Initial_concentration_in_negative_electrode', 29866, True, 28372.7, 31359.3),
                ('Maximum_concentration_in_positive_electrode', 63104, True, 59948.8, 66259.2),
                ('Negative_electrode_active_material_volume_fraction', 0.75, True, 0.7125, 0.7875),
                ('Initial_concentration_in_positive_electrode', 17038, True, 16186.1, 17889.9),
                ('Negative_electrode_electrons_in_reaction', 1, True, 0.95, 1.05),
                ('Positive_electrode_active_material_volume_fraction', 0.665, True, 0.63175, 0.69825),
                ('Maximum_concentration_in_negative_electrode', 33133.0, True, 31476.35, 34789.65),
                ('Negative_electrode_porosity', 0.25, True, 0.2375, 0.2625),
                ('Positive_electrode_porosity', 0.335, True, 0.31825, 0.35175),
                ('Initial_concentration_in_electrolyte', 1000.0, True, 950, 1050),
                ('Positive_electrode_conductivity', 0.18, True, 0.171, 0.189),
                ('Separator_porosity', 0.47, True, 0.4465, 0.4935),
                ('Positive_electrode_electrons_in_reaction', 1.0, True, 0.95, 1.05)
                )

model = pybamm.lithium_ion.DFN()
geometry = model.default_geometry
chemistry = pybamm.parameter_sets.Chen2020
param = pybamm.ParameterValues(chemistry=chemistry)
param.process_geometry(geometry)
param.process_model(model)
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 30, var.x_s: 30, var.x_p: 30, var.r_n: 10, var.r_p: 10}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)


def rmse(params, yinterp, ydata):
    sim = pybamm.Simulation(model, parameter_values=params)
    sim.solve([0, 3600])
    ysim = sim.solution["Terminal voltage [V]"].data
    xp = np.linspace(0.0, 3600.0, len(ysim))
    yp = ysim
    xvals = np.linspace(min(xp), max(xp), 280)
    yinterp = np.interp(xvals, xp, yp)
    return math.sqrt(mean_squared_error(ydata, yinterp))


mini = minimize(rmse, params, method='powell', args=(xdata, ydata))
print(fit_report(mini))


final = mini.residual
print(final)


new_sim = pybamm.Simulation(model, parameter_values=mini.params)
new_sim.solve([0, 3600])
y = new_sim.solution["Terminal voltage [V]"].data
xp = np.linspace(0.0, 3600.0, len(y))
yp = y
xvals = np.linspace(min(xp), max(xp), 280)
yinterp = np.interp(xvals, xp, yp)

plt.plot(t, yinterp, 'r-', label='model')
plt.plot(t, ydata, "g-", label='experiment')
plt.xlabel(r'$t$')
plt.ylabel('Terminal voltage')
plt.show()
