import numpy as np
import pybamm
from SALib.analyze import sobol
from SALib.sample import saltelli
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd

pd.set_option('display.max_rows', 4000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 4000)

problem = {
    'num_vars': 80,

    'names': [
              'Ambient temperature [K]',
              'Bulk solvent concentration [mol.m-3]',
              'Cation transference number',
              'Cell cooling surface area [m2]',
              'Cell volume [m3]',
              'Current function [A]',
              'EC diffusivity [m2.s-1]',
              'EC initial concentration in electrolyte [mol.m-3]',
              'Electrode height [m]',
              'Electrode width [m]',

              'Initial concentration in electrolyte [mol.m-3]',
              'Initial concentration in negative electrode [mol.m-3]',
              'Initial concentration in positive electrode [mol.m-3]',
              'Initial inner SEI thickness [m]',
              'Initial outer SEI thickness [m]',
              'Initial temperature [K]',
              'Inner SEI electron conductivity [S.m-1]',
              'Inner SEI lithium interstitial diffusivity [m2.s-1]',
              'Inner SEI open-circuit potential [V]',
              'Inner SEI partial molar volume [m3.mol-1]',

              'Inner SEI reaction proportion',
              'Lithium interstitial reference concentration [mol.m-3]',
              'Lower voltage cut-off [V]',
              'Maximum concentration in negative electrode [mol.m-3]',
              'Maximum concentration in positive electrode [mol.m-3]',
              'Negative current collector conductivity [S.m-1]',
              'Negative current collector density [kg.m-3]',
              'Negative current collector specific heat capacity [J.kg-1.K-1]',
              'Negative current collector thermal conductivity [W.m-1.K-1]',
              'Negative current collector thickness [m]',


              'Negative electrode active material volume fraction',
              'Negative electrode cation signed stoichiometry',
              'Negative electrode charge transfer coefficient',
              'Negative electrode conductivity [S.m-1]',
              'Negative electrode density [kg.m-3]',
              'Negative electrode diffusivity [m2.s-1]',
              'Negative electrode double-layer capacity [F.m-2]',
              'Negative electrode electrons in reaction',

              'Negative electrode porosity',
              'Negative electrode specific heat capacity [J.kg-1.K-1]',
              'Negative electrode thermal conductivity [W.m-1.K-1]',
              'Negative electrode thickness [m]',
              'Negative particle radius [m]',
              'Nominal cell capacity [A.h]',

              'Outer SEI open-circuit potential [V]',
              'Outer SEI partial molar volume [m3.mol-1]',

              'Outer SEI solvent diffusivity [m2.s-1]',
              'Positive current collector conductivity [S.m-1]',
              'Positive current collector density [kg.m-3]',
              'Positive current collector specific heat capacity [J.kg-1.K-1]',
              'Positive current collector thermal conductivity [W.m-1.K-1]',
              'Positive current collector thickness [m]',

              'Positive electrode active material volume fraction',
              'Positive electrode cation signed stoichiometry',

              'Positive electrode charge transfer coefficient',
              'Positive electrode conductivity [S.m-1]',
              'Positive electrode density [kg.m-3]',
              'Positive electrode diffusivity [m2.s-1]',
              'Positive electrode double-layer capacity [F.m-2]',
              'Positive electrode electrons in reaction',
              'Positive electrode porosity',
              'Positive electrode specific heat capacity [J.kg-1.K-1]',
              'Positive electrode thermal conductivity [W.m-1.K-1]',
              'Positive electrode thickness [m]',

              'Positive particle radius [m]',
              'Ratio of inner and outer SEI exchange current densities',
              'Reference temperature [K]',
              'SEI kinetic rate constant [m.s-1]',
              'SEI open-circuit potential [V]',
              'SEI reaction exchange current density [A.m-2]',
              'SEI resistivity [Ohm.m]',

              'Separator density [kg.m-3]',
              'Separator porosity',

              'Separator specific heat capacity [J.kg-1.K-1]',
              'Separator thermal conductivity [W.m-1.K-1]',
              'Separator thickness [m]',
              'Total heat transfer coefficient [W.m-2.K-1]',
              'Typical current [A]',
              'Typical electrolyte concentration [mol.m-3]',
              'Upper voltage cut-off [V]'],

    'bounds': [
               [295.17, 301.13],
               [2609.64, 2662.36],
               [0.2568, 0.2620],
               [0.00526, 0.00536],
               [2.40e-05, 2.44e-05],
               [4.95, 5.05],
               [1.98e-18, 2.02e-18],
               [4495.59, 4586.41],
               [0.064, 0.066],
               [1.56, 1.60],

               [990, 1010],
               [29567.34, 30164.66],
               [16867.62, 17208.38],
               [2.475e-09, 2.525e-09],
               [2.475e-09, 2.525e-09],
               [295.17, 301.13],
               [8.86e-14, 9.04e-14],
               [0.99e-20, 1.01e-20],
               [0.099, 0.101],
               [9.489e-05, 9.681e-05],

               [0.495, 0.505],
               [14.85, 15.15],
               [2.475, 2.525],
               [32801.67, 33464.33],
               [62472.96, 63735.04],
               [57826890, 58995110],
               [8870.4, 9049.6],
               [381.15, 388.85],
               [396.99, 405.01],
               [1.188, 1.212],


               [0.74, 0.76],
               [-1.01, -0.99],
               [0.495, 0.505],
               [212.85, 217.15],
               [1640.43, 1673.57],
               [3.267e-14, 3.333e-14],
               [0.198, 0.202],
               [0.99, 1.01],

               [0.2475, 0.2525],
               [693, 707],
               [1.683, 1.717],
               [8.43e-05, 8.67e-05],
               [5.80e-06, 5.92e-06],
               [4.95, 5.05],

               [0.792, 0.808],
               [9.489e-05, 9.681e-05],

               [2.475e-22, 2.525e-22],
               [36544860, 37283140],
               [2673, 2727],
               [888.03, 905.97],
               [234.63, 239.37],
               [1.584e-05, 1.616e-05],

               [0.658, 0.672],
               [-1.01, -0.99],

               [0.495, 0.505],
               [0.178, 0.182],
               [3229.38, 3294.62],
               [3.96e-15, 4.04e-15],
               [0.198, 0.202],
               [0.99, 1.01],
               [0.332, 0.338],
               [693, 707],
               [2.079, 2.121],
               [7.48e-05, 7.64e-05],

               [5.17e-06, 5.27e-06],
               [0.99, 1.01],
               [295.17, 301.13],
               [0.99e-12, 1.01e-12],
               [0.396, 0.404],
               [1.485e-07, 1.515e-07],
               [198000, 202000],

               [393.03, 400.97],
               [0.4653, 0.4747],

               [693, 707],
               [0.158, 0.162],
               [1.188e-05, 1.212e-05],
               [9.9, 10.1],
               [4.95, 5.05],
               [990, 1010],
               [4.158, 4.242]
               ]
}
param_values = saltelli.sample(problem, 1, calc_second_order=False)

model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model)
sim.solve([0, 3600])
solution = sim.solution
tv = solution["Terminal voltage [V]"].data
#tv1 = np.expand_dims(tv, axis=0).repeat(82, axis=0)


Y = np.zeros([param_values.shape[0], 100])
b = np.zeros([param_values.shape[0], 1])
chemistry = pybamm.parameter_sets.Chen2020
params = pybamm.ParameterValues(chemistry=chemistry)

for i, X in enumerate(param_values):
    n = 0
    for param_names in problem['names']:
        params[param_names] = X[n]
        n += 1

    a = pybamm.Simulation(model, parameter_values=params)
    a.solve([0, 3600])
    solution = a.solution
    res = solution["Terminal voltage [V]"].data

   # x = (0.000, 0.102, 0.200, 0.300, 0.402, 0.501, 0.600, 0.703, 0.802)
   # y = (4.014, 3.852, 3.756, 3.660, 3.544, 3.482, 3.407, 3.311, 3.215)


    xp = np.linspace(0.0, 3600.0, len(res))
    yp = res
    xvals = np.linspace(min(xp), max(xp), 100)
    yinterp = np.interp(xvals, xp, yp)
    Y[i] = yinterp

    #c = np.concatenate(Y, axis=None)

    b[i] = sqrt(mean_squared_error(Y[i], tv))
    #print(b[i])

Si = sobol.analyze(problem, b, calc_second_order=False, print_to_console=True)
print()
names = [
              'Ambient temperature [K]',
              'Bulk solvent concentration [mol.m-3]',
              'Cation transference number',
              'Cell cooling surface area [m2]',
              'Cell volume [m3]',
              'Current function [A]',
              'EC diffusivity [m2.s-1]',
              'EC initial concentration in electrolyte [mol.m-3]',
              'Electrode height [m]',
              'Electrode width [m]',

              'Initial concentration in electrolyte [mol.m-3]',
              'Initial concentration in negative electrode [mol.m-3]',
              'Initial concentration in positive electrode [mol.m-3]',
              'Initial inner SEI thickness [m]',
              'Initial outer SEI thickness [m]',
              'Initial temperature [K]',
              'Inner SEI electron conductivity [S.m-1]',
              'Inner SEI lithium interstitial diffusivity [m2.s-1]',
              'Inner SEI open-circuit potential [V]',
              'Inner SEI partial molar volume [m3.mol-1]',

              'Inner SEI reaction proportion',
              'Lithium interstitial reference concentration [mol.m-3]',
              'Lower voltage cut-off [V]',
              'Maximum concentration in negative electrode [mol.m-3]',
              'Maximum concentration in positive electrode [mol.m-3]',
              'Negative current collector conductivity [S.m-1]',
              'Negative current collector density [kg.m-3]',
              'Negative current collector specific heat capacity [J.kg-1.K-1]',
              'Negative current collector thermal conductivity [W.m-1.K-1]',
              'Negative current collector thickness [m]',


              'Negative electrode active material volume fraction',
              'Negative electrode cation signed stoichiometry',
              'Negative electrode charge transfer coefficient',
              'Negative electrode conductivity [S.m-1]',
              'Negative electrode density [kg.m-3]',
              'Negative electrode diffusivity [m2.s-1]',
              'Negative electrode double-layer capacity [F.m-2]',
              'Negative electrode electrons in reaction',

              'Negative electrode porosity',
              'Negative electrode specific heat capacity [J.kg-1.K-1]',
              'Negative electrode thermal conductivity [W.m-1.K-1]',
              'Negative electrode thickness [m]',
              'Negative particle radius [m]',
              'Nominal cell capacity [A.h]',

              'Outer SEI open-circuit potential [V]',
              'Outer SEI partial molar volume [m3.mol-1]',

              'Outer SEI solvent diffusivity [m2.s-1]',
              'Positive current collector conductivity [S.m-1]',
              'Positive current collector density [kg.m-3]',
              'Positive current collector specific heat capacity [J.kg-1.K-1]',
              'Positive current collector thermal conductivity [W.m-1.K-1]',
              'Positive current collector thickness [m]',

              'Positive electrode active material volume fraction',
              'Positive electrode cation signed stoichiometry',

              'Positive electrode charge transfer coefficient',
              'Positive electrode conductivity [S.m-1]',
              'Positive electrode density [kg.m-3]',
              'Positive electrode diffusivity [m2.s-1]',
              'Positive electrode double-layer capacity [F.m-2]',
              'Positive electrode electrons in reaction',
              'Positive electrode porosity',
              'Positive electrode specific heat capacity [J.kg-1.K-1]',
              'Positive electrode thermal conductivity [W.m-1.K-1]',
              'Positive electrode thickness [m]',

              'Positive particle radius [m]',
              'Ratio of inner and outer SEI exchange current densities',
              'Reference temperature [K]',
              'SEI kinetic rate constant [m.s-1]',
              'SEI open-circuit potential [V]',
              'SEI reaction exchange current density [A.m-2]',
              'SEI resistivity [Ohm.m]',

              'Separator density [kg.m-3]',
              'Separator porosity',

              'Separator specific heat capacity [J.kg-1.K-1]',
              'Separator thermal conductivity [W.m-1.K-1]',
              'Separator thickness [m]',
              'Total heat transfer coefficient [W.m-2.K-1]',
              'Typical current [A]',
              'Typical electrolyte concentration [mol.m-3]',
              'Upper voltage cut-off [V]'],

outputs = pd.DataFrame(index=list(names), data=Si)
outputs.to_csv("/Users/mac/Documents/MA/outputs1.csv")