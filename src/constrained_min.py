import numpy as np
import math


def interior_pt(f, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    path_history = dict(path=[], values=[])
    t = 1

    prev_f, prev_grad, prev_hass = get_values_after_log_barrier(f, ineq_constraints, x0, t)
    
    num_of_constraints = len(ineq_constraints)
    prev_x0 = x0

    path_history['path'].append(prev_x0.copy())
    path_history['values'].append(f(prev_x0.copy())[0])

    while (num_of_constraints / t) > 1e-8:
        for i in range(10):
            dir = find_direction(prev_hass, eq_constraints_mat, prev_grad)
            step_len = wolfe_condition_with_backtracking(f, prev_x0, prev_f, prev_grad, dir, ineq_constraints, t)
            next_x0 = prev_x0 + dir * step_len
            
            next_f, next_grad, next_hass = get_values_after_log_barrier(f, ineq_constraints, next_x0, t)

            lamda = np.sqrt(np.dot(dir, np.dot(next_hass, dir.T)))
            if 0.5 * (lamda ** 2) < 1e-8:
                break

            prev_x0 = next_x0
            prev_f = next_f
            prev_grad = next_grad
            prev_hass = next_hass
        
        path_history['path'].append(prev_x0.copy())
        path_history['values'].append(f(prev_x0.copy())[0])
        t *= 10
    
    return prev_x0, f(prev_x0.copy())[0], path_history


def log_barrier(ineq_constraints, x0):
    x0_dim = x0.shape[0]
    log_f = 0
    log_g = np.zeros((x0_dim,))
    log_h = np.zeros((x0_dim, x0_dim))

    for constraint in ineq_constraints:
        f,g,h = constraint(x0)
        log_f += math.log(-f)
        log_g += (1.0 / -f) * g

        grad = g / f
        grad_dim = grad.shape[0]
        grad_tile = np.tile(grad.reshape(grad_dim, -1), (1, grad_dim)) * np.tile(grad.reshape(grad_dim, -1).T, (grad_dim, 1))
        log_h += (h * f - grad_tile) / f ** 2
    
    return -log_f, log_g, -log_h


def find_direction_eq(previous_hassian, A, previous_gradiant):
    left_mat = np.block([
        [previous_hassian, A.T],
        [A, 0],
    ])
    right_vec = np.block([[-previous_gradiant, 0]])
    right_vec_t = right_vec.T
    ans = np.linalg.solve(left_mat, right_vec_t).T[0]
    return ans[0:A.shape[1]]


def find_direction_no_eq(previous_hassian, previous_gradiant):
    left_mat = previous_hassian
    right_vec_t = -previous_gradiant
    ans = np.linalg.solve(left_mat, right_vec_t)
    return ans


def find_direction(previous_hassian, A, previous_gradiant):
    if A is not None:
        return find_direction_eq(previous_hassian, A, previous_gradiant)
    return find_direction_no_eq(previous_hassian, previous_gradiant)
 

def wolfe_condition_with_backtracking(f, x, val, gradient, direction, ineq_constraints, t, alpha=0.01, beta=0.5, max_iter=10):
    step_length = 1
    curr_val, _, _ = f(x + step_length * direction)

    iter = 0
    while iter < max_iter and curr_val > val + alpha * step_length * gradient.dot(direction):
        step_length *= beta
        curr_val, _, _ = f(x + step_length * direction)
        iter += 1

    return step_length


def get_values_after_log_barrier(f, ineq_constraints, x0, t):
    val, grad, hass = f(x0)
    log_f, log_g, log_h = log_barrier(ineq_constraints, x0)
    prev_f, prev_grad, prev_hass = t*val + log_f, t*grad + log_g, t*hass + log_h
    return prev_f, prev_grad, prev_hass
