import matplotlib.pyplot as plt
import numpy as np
import pybamm
import scipy.optimize
from math import sqrt
import pandas as pd
from sklearn.metrics import mean_squared_error

#df = pd.read_excel('/Users/mac/Documents/MA/current1.xlsx')
#df_n = df.to_numpy()
#xdata = np.array(df_n).flatten()

df1 = pd.read_excel('/Users/mac/Documents/MA/voltage.xlsx')
df1_n = df1.to_numpy()
ydata = np.array(df1_n).flatten()

model = pybamm.lithium_ion.DFN()
geometry = model.default_geometry
param = model.default_parameter_values
param.update({"Current function [A]": "[input]"})
param.process_geometry(geometry)
param.process_model(model)
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 30, var.x_s: 30, var.x_p: 30, var.r_n: 10, var.r_p: 10}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)
t_eval = np.linspace(0, 3600, len(ydata))
solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-3)
solution = solver.solve(
    model, t_eval,
    inputs={"Current function [A]": 5})

def rmse(parameters):
    print('solving for "Current function" =', parameters[0])
    simulation = solver.solve(
        model, t_eval,
        inputs={"Current function [A]": parameters[0]}
    )["Terminal voltage"](t_eval)
    return sqrt(mean_squared_error(simulation, ydata))


bounds = (0.01, 0.6)
x0 = np.random.uniform(low=bounds[0], high=bounds[1])
print('starting parameter is', x0)
res = scipy.optimize.minimize(
    rmse, [x0], bounds=[bounds]
)
print('recovered parameter is', res.x[0])
