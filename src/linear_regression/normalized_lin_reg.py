import matplotlib.pyplot as plt
import pandas as pd
import numpy as np




dataframe = pd.read_csv("data/data.csv")
base = "time_spent"
match = "grade"

# FILTER OUT NULL VALUES
dataframe = dataframe[dataframe[base].notnull()]
dataframe = dataframe[dataframe[match].notnull()]

x_axis = dataframe[base]
y_axis = dataframe[match]

x_norm = (x_axis - x_axis.min()) / (x_axis.max()- x_axis.min())
y_norm = (y_axis - y_axis.min()) / (y_axis.max()- y_axis.min())




def our_linreg(x, y):
    learning_rate = 0.01
    limit = 100
    b = 0.0
    m = 0.0
    for _ in range(0,limit):
        b = b - (learning_rate * gradient_b(x, y, m, b))
        m = m - (learning_rate * gradient_m(x, y, m, b))
    return m,  b


def gradient_b(x, y, m, b):

    res = (y - (b + (m*x))) * (-1)
    return np.sum(res)

def gradient_m(x, y, m, b):
    res = (y - (b + (m*x))) * (-x)
    return np.sum(res)



# LINREG
our_m, our_b = our_linreg(x_norm.values, y_norm.values)

# back normalize coefficients
re_m = our_m * (y_axis.max()- y_axis.min()) / (x_axis.max()- x_axis.min())
re_b =  y_axis.min() + our_b * (y_axis.max()- y_axis.min()) - x_axis.min() * re_m


'''
# PLOT normalized
plt.plot(x_norm, y_norm, 'o', label='original data')
plt.plot(x_norm, our_b + our_m * x_norm, '-', label='our fitted line')
plt.legend()
'''

# PLOT 
plt.plot(x_axis, y_axis, 'o', label='original data')
plt.plot(x_axis, re_b + re_m * x_axis, '-', label='our fitted line')
plt.legend()






