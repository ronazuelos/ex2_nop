import unittest
import numpy as np
from src.utils import plot_results_qp, plot_results_lp, plot_values_graph
from src.constrained_min import interior_pt
from tests.examples import *


class TestMinimize(unittest.TestCase):
    def test_qp(self):
        ineq_constraints_qp = [qp_ineq1, qp_ineq2, qp_ineq3]
        A = np.array([1, 1, 1]).reshape(1, 3)
        x0 = np.array([0.1, 0.2, 0.7])
        final_candidate, final_obj, history = interior_pt(qp_function, ineq_constraints_qp, A, 0, x0)
        
        qp_ineq_containts_at_final = [c(final_candidate)[0] for c in ineq_constraints_qp]
        print('Final candidate:', final_candidate)
        print('Objective value at final candidate:', final_obj)
        print('Inequality constraints values at final candidate:', qp_ineq_containts_at_final)
        print('Equality constraints values at final candidate:', (A * final_candidate).sum())

        plot_values_graph(history['values'], 'Objective values per outer iteration number (QP)')
        plot_results_qp(history['path'], 'Feasible region and path taken by the algorithm (QP)')


    def test_lp(self):
        ineq_constraints_lp = [lp_ineq1, lp_ineq2, lp_ineq3, lp_ineq4]
        A = None
        x0 = np.array([0.5, 0.75])
        final_candidate, final_obj, history = interior_pt(lp_function, ineq_constraints_lp, A, 0, x0)

        lp_ineq_containts_at_final = [c(final_candidate)[0] for c in ineq_constraints_lp]
        print('Final candidate:', final_candidate)
        print('Objective value at final candidate:', final_obj)
        print('Inequality constraints values at final candidate:', lp_ineq_containts_at_final)

        plot_values_graph(history['values'], 'Objective values per outer iteration number (LP)')
        plot_results_lp(history['path'], 'Feasible region and path taken by the algorithm (LP)')
