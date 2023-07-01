import matplotlib.pyplot as plt
import numpy as np


def plot_results_qp(path, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    path = np.array(path)

    ax.plot_trisurf([1, 0, 0], [0, 1, 0], [0, 0, 1], color='lightgray', alpha=0.5)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Path')
    ax.scatter(path[-1][0], path[-1][1], path[-1][2], s=50, c='gold', marker='o', label='Final candidate')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend()
    ax.view_init(45, 45)
    plt.show()


def plot_results_lp(path, title):
    fig, ax = plt.subplots(1, 1)
    path = np.array(path)

    x = np.linspace(-1, 3, 1000)
    y = np.linspace(-2, 2, 1000)
    contraints_ineq = {
        'y=0': (x, x*0),
        'y=1': (x, x*0 + 1),
        'x=2': (y*0 + 2, y),
        'y=-x+1': (x, -x + 1)
    }
    
    for f, (x, y) in contraints_ineq.items():
        ax.plot(x, y, label=f)

    # create shape with the vertices: (0, 1), (2, 1), (2, 0), (1, 0)
    ax.fill([0, 2, 2, 1], [1, 1, 0, 0], 'lightgray', label='Feasible region')
    ax.plot(path[:, 0], path[:, 1], c='k', label='Path')
    ax.scatter(path[-1][0], path[-1][1], s=50, c='gold', marker='o', label='Final candidate')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.show()


def plot_values_graph(values, title):
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(0, len(values)-1, len(values))
    ax.plot(x, values, marker='.')
    ax.set_title(title)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Objective values')
    plt.show()
