import numpy as np
import pybamm
from SALib.analyze import sobol
from SALib.sample import saltelli
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from SALib.plotting.bar import plot as barplot
import matplotlib.pyplot as plot

pd.set_option('display.max_rows', 4000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 4000)

problem = {
    'num_vars': 88,

    'names': ['1 + dlnf/dlnc',
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

              'Negative electrode Bruggeman coefficient (electrode)',
              'Negative electrode Bruggeman coefficient (electrolyte)',
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
              'Number of cells connected in series to make a battery',
              'Number of electrodes connected in parallel to make a cell',
              'Outer SEI open-circuit potential [V]',
              'Outer SEI partial molar volume [m3.mol-1]',

              'Outer SEI solvent diffusivity [m2.s-1]',
              'Positive current collector conductivity [S.m-1]',
              'Positive current collector density [kg.m-3]',
              'Positive current collector specific heat capacity [J.kg-1.K-1]',
              'Positive current collector thermal conductivity [W.m-1.K-1]',
              'Positive current collector thickness [m]',
              'Positive electrode Bruggeman coefficient (electrode)',
              'Positive electrode Bruggeman coefficient (electrolyte)',
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
              'Separator Bruggeman coefficient (electrolyte)',
              'Separator density [kg.m-3]',
              'Separator porosity',

              'Separator specific heat capacity [J.kg-1.K-1]',
              'Separator thermal conductivity [W.m-1.K-1]',
              'Separator thickness [m]',
              'Total heat transfer coefficient [W.m-2.K-1]',
              'Typical current [A]',
              'Typical electrolyte concentration [mol.m-3]',
              'Upper voltage cut-off [V]'],

    'bounds':  [[0.95, 1.05],
               [283.24, 313.06],
               [2504.2, 2767.8],
               [0.24643, 0.27237],
               [0.00505, 0.00558],
               [2.3e-05, 2.541e-05],
               [4.75, 5.25],
               [1.9e-18, 2.1e-18],
               [4313.95, 4768.05],
               [0.06175, 0.06825],
               [1.501, 1.659],

               [950, 1050],
               [28372.7, 31359.3],
               [16186.1, 17889.9],
               [2.375e-09, 2.625e-09],
               [2.375e-09, 2.625e-09],
               [283.24, 313.06],
               [8.5025e-14, 9.3975e-14],
               [0.95e-20, 1.05e-20],
               [0.095, 0.105],
               [9.106e-05, 10.064e-05],

               [0.475, 0.525],
               [14.25, 15.75],
               [2.375, 2.625],
               [31476.35, 34789.65],
               [59948.8, 66259.2],
               [55490450, 61331550],
               [8512, 9408],
               [365.75, 404.25],
               [380.95, 421.05],
               [1.14e-05, 1.26e-05],

               [1.425, 1.575],
               [1.425, 1.575],
               [0.7125, 0.7875],
               [-1.05, -0.95],
               [0.475, 0.525],
               [204.25, 225.75],
               [1574.15, 1739.85],
               [3.135e-14, 3.465e-14],
               [0.19, 0.21],
               [0.95, 1.05],

               [0.2375, 0.2625],
               [665, 735],
               [1.615, 1.785],
               [8.094e-05, 8.946e-05],
               [5.567e-06, 6.153e-06],
               [4.75, 5.25],
               [0.95, 1.05],
               [0.95, 1.05],
               [0.76, 0.84],
               [9.11e-05, 10.06e-05],

               [2.375e-22, 2.625e-22],
               [35068300, 38759700],
               [2565, 2835],
               [852.15, 941.85],
               [225.15, 248.85],
               [1.52e-05, 1.68e-05],
               [1.425, 1.575],
               [1.425, 1.575],
               [0.632, 0.698],
               [-1.05, -0.95],

               [0.475, 0.525],
               [0.171, 0.189],
               [3098.9, 3425.1],
               [3.8e-15, 4.2e-15],
               [0.19, 0.21],
               [0.95, 1.05],
               [0.318, 0.352],
               [665, 735],
               [1.995, 2.205],
               [7.182e-05, 7.938e-05],

               [4.959e-06, 5.481e-06],
               [0.95, 1.05],
               [283.24, 313.06],
               [0.95e-12, 1.05e-12],
               [0.38, 0.42],
               [1.425e-07, 1.575e-07],
               [190000, 210000],
               [1.425, 1.575],
               [377.15, 416.85],
               [0.4465, 0.4935],

               [665, 735],
               [0.152, 0.168],
               [1.14e-05, 1.26e-05],
               [9.5, 10.5],
               [4.75, 5.25],
               [950, 1050],
               [3.99, 4.41]]
}
param_values = saltelli.sample(problem, 1, calc_second_order=False)

model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model)
sim.solve([0, 3600])
solution = sim.solution
tv = solution["Terminal voltage [V]"].data

# chemistry = pybamm.parameter_sets.Chen2020
# params = pybamm.ParameterValues(chemistry=chemistry)

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

    b[i] = sqrt(mean_squared_error(Y[i], tv))
    # print(b[i])

Si = sobol.analyze(problem, b, calc_second_order=False, print_to_console=True)
# print()
# print(Si['S1'])
#result = Si['ST']
#sorted(result, reverse=False)
#print(result)

names = ['1 + dlnf/dlnc',
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

              'Negative electrode Bruggeman coefficient (electrode)',
              'Negative electrode Bruggeman coefficient (electrolyte)',
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
              'Number of cells connected in series to make a battery',
              'Number of electrodes connected in parallel to make a cell',
              'Outer SEI open-circuit potential [V]',
              'Outer SEI partial molar volume [m3.mol-1]',

              'Outer SEI solvent diffusivity [m2.s-1]',
              'Positive current collector conductivity [S.m-1]',
              'Positive current collector density [kg.m-3]',
              'Positive current collector specific heat capacity [J.kg-1.K-1]',
              'Positive current collector thermal conductivity [W.m-1.K-1]',
              'Positive current collector thickness [m]',
              'Positive electrode Bruggeman coefficient (electrode)',
              'Positive electrode Bruggeman coefficient (electrolyte)',
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
              'Separator Bruggeman coefficient (electrolyte)',
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
sorted(outputs, reverse=False)
outputs.to_csv("/Users/mac/Documents/MA/outputs3.csv")

plt.subplots(figsize=(10, 12))
plt.barh(problem['names'], Si['ST'])
plt.ylabel("Parameters")
plt.xlabel("S-value")
plt.title("Sensitivity analysis of parameters")
plt.show()




# Si_df = Si.to_df()
# barplot(Si_df[0])
# plot.show()

#plt.subplots(figsize=(90, 90)) # 设置画面大小

# plt.barh(range(len(Si['S1'])), Si['S1'])
# plt.barh(range(len(Si['ST'])), Si['ST'])
# plt.legend()
# plt.ylabel("Parameters")
# plt.xlabel("S-value")
# plt.title("Sensitivity analysis of parameters")
# plt.show()
# xlab="S-value", ylab="Parameters", title="Sensitivity analysis of parameters"
