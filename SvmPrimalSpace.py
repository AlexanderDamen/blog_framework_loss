"""
Very rudimentary implementation of linear support vector machine in primal space.

This solver will focus on the primal space. For now a nonlinear constraint is used, but in principle this can be
replaced by a linear constraint. Might result in a speed improvement.

"""

import numpy as np
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize


class SvmPrimalSpace:
    def __init__(self, loss_type='l2', gamma=1):
        self.w = None
        self.b = None
        self.X = None
        self.y = None
        self.gamma = gamma
        self.loss_type = loss_type  # can also be l1

    @staticmethod
    def calculate_slack_variables(X, y, w):
        # every slack variable needs to be larger or equal to 0
        slack_variables = 1 - y * (np.matmul(X, w[:-1].T) + w[-1])
        return slack_variables

    def primal_loss(self, w):
        slack_variables = self.calculate_slack_variables(self.X, self.y, w)
        l2_reg = np.sum(np.square(w))
        loss = 0.5 * l2_reg + self.gamma * 0.5 * np.sum(np.square(slack_variables[slack_variables >= 0]))
        return loss

    @staticmethod
    def single_constraint(w, x, y):
        return 1 - y * (np.matmul(w[:-1].T, x) + w[-1])

    @staticmethod
    def gen_constraints(X, y):
        constraints = []
        for i in range(len(y)):
            def constraint_i(w):
                return SvmPrimalSpace.single_constraint(w, X[i, :], y[i])
            constraints.append(NonlinearConstraint(constraint_i, lb=0, ub=np.inf))
        return constraints

    def train(self, X, y, w0):
        self.X = X
        self.y = y
        cons = SvmPrimalSpace.gen_constraints(X, y)
        res = minimize(self.primal_loss, x0=w0, constraints=cons, options={"maxiter": 5000, "disp": True})
        self.w = res.x[:-1].T
        self.b = res.x[-1]

    def fit(self, X, y):
        return self.train(X, y, w0=np.zeros(X.shape[1] + 1))

    def predict(self, X):
        pred = np.sign(np.matmul(X, self.w) + self.b)
        return pred

    def get_params(self, deep=True):
        params = {"w": self.w, "b": self.b}
        return params
