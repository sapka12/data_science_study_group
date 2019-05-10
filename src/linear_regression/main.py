import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from linear_regression.linreg import linreg, our_linreg

if __name__ == '__main__':

    dataframe = pd.read_csv("../data/grades.csv")
    base = "time_spent"
    match = "grade"

    # FILTER OUT NULL VALUES
    dataframe = dataframe[dataframe[base].notnull()]
    dataframe = dataframe[dataframe[match].notnull()]

    x_axis = dataframe[base]
    y_axis = dataframe[match]

    # LINREG
    m, b = linreg(x_axis, y_axis)
    our_m, our_b = our_linreg(x_axis.values, y_axis.values)

    # PLOT
    plt.plot(x_axis, y_axis, 'o', label='original data')

    plt.plot(x_axis, b + m * x_axis, '-', label='fitted line')
    plt.plot(x_axis, our_b + our_m * x_axis, '-', label='our fitted line')
    plt.legend()
    plt.savefig('../output/graph.png')
