# src/benchmark_functions.py
import numpy as np

def rastrigin(x):
    """
    Rastrigin function for minimization.
    Global minimum is f(0,0,...,0) = 0.
    Domain: [-5.12, 5.12]
    """
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock(x):
    """
    Rosenbrock function (banana function) for minimization.
    Global minimum is f(1,1,...,1) = 0.
    Domain: [-2.048, 2.048]
    """
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def ackley(x):
    """
    Ackley function for minimization.
    Global minimum is f(0,0,...,0) = 0.
    Domain: [-32.768, 32.768]
    """
    n = len(x)
    sum_sq_term = -0.2 * np.sqrt(np.sum(x**2) / n)
    cos_term = np.sum(np.cos(2 * np.pi * x)) / n
    return -20 * np.exp(sum_sq_term) - np.exp(cos_term) + 20 + np.e