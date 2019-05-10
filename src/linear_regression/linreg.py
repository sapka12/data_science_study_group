from scipy import stats
import numpy as np

def linreg(x, y):
  slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
  return slope, intercept


def our_linreg(x, y):
  learning_rate = 0.0001
  limit = 10000
  b = 0.0
  m = 0.0
  for _ in range(0,limit):
    b = b - (learning_rate * gradient_b(x, y, m, b))
    m = m - (learning_rate * gradient_m(x, y, m, b))
  return m,  b


def gradient_b(x, y, m, b):
# (y - _y)^2 = (y - (b + m*x))^2
  res = (y - (b + (m*x))) * (-1)
  return np.mean(res)


def gradient_m(x, y, m, b):
  res = (y - (b + (m*x))) * (-x)
  return np.mean(res)
