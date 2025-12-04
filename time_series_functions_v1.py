# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 21:26:07 2025

@author: Robert H Lee


Useful resources:
    Econometric Methods with Applications in Business and Economics
    Heij, de Boer, Frarses, Kloek, van Dijk.
    
    Time Series Analysis and its Applications (With R Examples)
    Shumway and Stoffer
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def lag(function, a, b, k, *params):
  """
  I currently don't use this for anything
  """
  n = len(a)
  a_prime = a[:n-k] # x_{t+h}
  b_prime = b[k:] # x_t
  return function(a_prime, b_prime, *params)

def sum_product(list1, list2, c1, c2):
  list1 = np.array(list1)
  list2 = np.array(list2)
  return np.sum((list1-c1)*(list2-c2))

def autocovariance(x, k, mean = None):
  if mean == None:
    mean = np.mean(x)
  n = len(x)
  a = x[:n-k]
  b = x[k:]
  S = float(sum_product(a, b, mean, mean))
  return S/n

def cross_covariance(x, y, k, mean_x = None, mean_y = None):
  if mean_x == None:
    mean_x = np.mean(x)
  if mean_y == None:
    mean_y = np.mean(y)
  n = len(x)
  a = x[:n-k]
  b = y[k:] # oops
  S = float(sum_product(a, b, mean_x, mean_y))
  return S/n

def autocorrelation(x, k, var_x = None, mean_x = None):
  if var_x == None:
    var_x = np.var(x, ddof = 0)
  return float(autocovariance(x, k, mean_x) / var_x)

def ACF(x, k_max, var_x = None, mean_x = None):
  autocorrelations = []
  # philosophy is to keep on handing down None (if we have no choice)
  # until we get to the point where the value is actually needed;
  # then and only then do we create it if it's missing.
  # if var_x == None: # handled in autocorrelation()
    # var_x = np.var(x, ddof=0)
  for k in range(1, k_max):
    autocorrelations.append(autocorrelation(x, k, var_x, mean_x))
  return autocorrelations

def ACF_with_plot(x, k_max, var_x = None, mean_x = None):
  n = len(x)
  bound = 2/math.sqrt(n)
  autocorrs = ACF(x, k_max, var_x, mean_x)
  plt.bar(np.arange(1, k_max), autocorrs)
  plt.hlines(bound, 0, k_max, 'r', '--')
  plt.hlines(-bound, 0, k_max, 'r', '--')
  plt.show()
  return autocorrs

def cross_correlation(x, y, k, mean_x = None, mean_y = None,
                      var_x = None, var_y = None):
  if mean_x == None:
    mean_x = np.mean(x)
  if mean_y == None:
    mean_y = np.mean(y)
  if var_x == None:
    var_x = np.var(x, ddof=0)
  if var_y == None:
    var_y = np.var(y, ddof=0)
  cross_covar = cross_covariance(x, y, k, mean_x, mean_y)
  denom = math.sqrt(var_x*var_y)
  return cross_covar/denom

def CCF(x, y, k_max, mean_x = None, mean_y = None,
                      var_x = None, var_y = None):
  if mean_x == None:
    mean_x = np.mean(x)
  if mean_y == None:
    mean_y = np.mean(y)
  if var_x == None:
    var_x = np.var(x, ddof=0)
  if var_y == None:
    var_y = np.var(y, ddof=0)
  crosscorrs = []
  for k in range(1, k_max):
    crosscorrs.append(cross_correlation(x, y, k, mean_x, mean_y, var_x, var_y))
  return crosscorrs

def CCF_with_plot(x, y, k_max, mean_x = None, mean_y = None,
                      var_x = None, var_y = None):
  if mean_x == None:
    mean_x = np.mean(x)
  if mean_y == None:
    mean_y = np.mean(y)
  if var_x == None:
    var_x = np.var(x, ddof=0)
  if var_y == None:
    var_y = np.var(y, ddof=0)
  crosscorrs = []
  for k in range(1, k_max):
    crosscorrs.append(cross_correlation(x, y, k, mean_x, mean_y, var_x, var_y))
  n = len(x)
  bound = 2/math.sqrt(n)
  plt.bar(np.arange(1, k_max), crosscorrs)
  plt.hlines(bound, 0, k_max, 'r', '--')
  plt.hlines(-bound, 0, k_max, 'r', '--')
  plt.show()
  return crosscorrs

def plot_series(series, linestyle='-'):
  plt.figure(figsize=(12,2))
  plt.plot(series, linestyle)
  plt.show()

def gen_ARCH1(random_variates, sigma_0, alpha0, alpha1):
  n = len(random_variates)
  X = random_variates
  S = [sigma_0]
  Y = [X[0]*sigma_0]
  for i in range(1, n):
    sigma_squared = alpha0 + alpha1*Y[i-1]**2
    sigma = math.sqrt(sigma_squared)
    S.append(sigma)
    Y.append(sigma*X[i])
  return Y, S

def gen_GARCH11(random_variates, sigma_0, alpha0, alpha1, beta1):
  if alpha1 + beta1 >= 1:
    print("alpha1 + beta1 should be less than 1")
  n = len(random_variates)
  X = random_variates
  S = [sigma_0]
  V = [sigma_0**2]
  Y = [X[0]*sigma_0]
  for i in range(1, n):
    sigma_squared = alpha0 + alpha1*Y[i-1]**2 + beta1*V[i-1]
    sigma = math.sqrt(sigma_squared)
    V.append(sigma_squared)
    S.append(sigma)
    Y.append(sigma*X[i])
  return Y, S

def gen_AR(random_variates, coefficients, plot = True):
    """
    Y = C_1 Y_{t-1} + C_2 Y_{t-2} + ... + C_p Y_{t-p} + W_t
    """
    W = random_variates
    n = len(W)
    C = coefficients
    p = len(coefficients)
    Y = []
    
    # populate the first few observations with noise
    for i in range(p):
        Y.append(W[i])
    
    for i in range(p, n):
        y_obs = W[i]
        for j in range(p):
            y_obs += C[j]*Y[i-j-1]
        Y.append(y_obs)
    plt.figure(figsize=(10,2))
    plt.plot(Y, '-')
    plt.show()
    return Y     
 

def gen_MA(random_variates, coefficients, plot = True):
    """
    Y = C_1 W_{t-1} + C_2 W_{t-2} + ... + C_q W_{t-q} + W_t
    
    I use nested loops to calculate this rather than numpy slices:
    Y = [np.sum(C[::-1]*W[i-q:i])+W[i] for i in range(q, n)]
    as I find the loops much more readable.
    """
    W = random_variates
    n = len(W)
    C = coefficients
    q = len(coefficients)
    Y = []
    
    # populate the first few observations with noise
    for i in range(q):
        Y.append(W[i])
    
    
    for i in range(q, n):
        y_obs = W[i]
        for j in range(q):
            y_obs += C[j]*W[i-j-1]
        Y.append(y_obs)
    
    plt.figure(figsize=(10,2))
    plt.plot(Y, '-')
    plt.show()
    return Y

def gen_ARMA(random_variates, ar_coefficients, ma_coefficients, plot = True):
    W = random_variates
    n = len(W)
    arC = ar_coefficients 
    maC = ma_coefficients
    p = len(ar_coefficients)
    q = len(ma_coefficients)
    r = max(p, q)
    Y = []
    
    # populate the first few observations with noise
    for i in range(r):
        Y.append(W[i])
    
    for i in range(r, n):
        y_obs = W[i]
        for j in range(p):
            y_obs += arC[j]*Y[i-j-1]
        for j in range(q):
            y_obs += maC[j]*W[i-j-1]
        Y.append(y_obs)

    plt.figure(figsize=(10,2))
    plt.plot(Y, '-')
    plt.show()
    return Y         
    

def main():
    np.random.seed(1)
    noise = np.random.normal(size = 1000)
    # Y = gen_MA(X, [-0.5, 0.5])
    # ACF_with_plot(Y, 10)
    
    # Z = gen_AR(X, [1.5, -0.75])
    # ACF_with_plot(Z, 10)
    
    ARMA_process = gen_ARMA(noise, [0.5], [0.5])
    ACF_with_plot(ARMA_process, 20)

    AR_process = gen_AR(noise, [0.5])
    ACF_with_plot(AR_process, 20)
    
    MA_process = gen_MA(noise, [0.5])
    ACF_with_plot(MA_process, 20)

    # # [.o......XXX] : series 1
    # # [XXX.o......] : series 2
    # # hence, we cross correlate series1[i-k] with series2[i]
    # # might be nice to reformulate this, so we get both directions in 1 plot
    # _ = CCF_with_plot(noise, AR_process, 20) # noise[i-k] on AR[i]
    # _ = CCF_with_plot(AR_process, noise, 20)
    # _ = CCF_with_plot(noise, MA_process, 20)
    # _ = CCF_with_plot(MA_process, noise, 20)

if __name__ == "__main__":
    main()