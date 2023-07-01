import numpy as np


# Quadratic function definition
def qp_function(x):
    f = x[0] ** 2  + x[1] ** 2 + (x[2] + 1) ** 2
    g = np.array([2 * x[0], 2 * x[1], 2 * x[2] + 2])
    h = np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2]
    ])
    return f, g, h


# Quadratic function inequalties definition
def qp_ineq1(x):
    f = -x[0]
    g = np.array([-1, 0, 0])
    h = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    return f, g, h


def qp_ineq2(x):
    f = -x[1]
    g = np.array([0, -1, 0])
    h = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    return f, g, h


def qp_ineq3(x):
    f = -x[2]
    g = np.array([0, 0, -1])
    h = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    return f, g, h


# Linear function definition
def lp_function(x):
    f = -x[0] - x[1]
    g = np.array([-1, -1])
    h = np.array([
        [0,0],
        [0,0]
    ])
    return f, g, h


# Linear function inequalties definition
def lp_ineq1(x):
    # y <=1
    f = x[1] -1
    g = np.array([0, 1])
    h = np.array([
        [0, 0], 
        [0, 0]
    ])
    return f, g, h


def lp_ineq2(x):
    # x <=2 
    f = x[0] -2
    g = np.array([1, 0])
    h = np.array([
        [0, 0], 
        [0, 0]
    ])
    return f, g, h


def lp_ineq3(x):
    # y >=0
    f = -x[1]
    g = np.array([0, -1])
    h = np.array([
        [0, 0],
        [0, 0]
    ])
    return f, g, h
 
 
def lp_ineq4(x):
    # 𝑦 ≥ −𝑥 + 1 = -x -y +1 <=0
    f = -x[0] - x[1] + 1
    g = np.array([-1, -1])
    h = np.array([
        [0, 0],
        [0, 0]
    ])
    return f, g, h
