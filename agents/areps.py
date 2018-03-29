import numpy as np
from scipy.optimize import minimize
from agents.amds import AMDS
from agents.util import (get_next_theta, gauss_pdf, reps_dual)


class AREPS(AMDS):
    def __init__(self, dmps, n_updates, n_reps, n_dims, n_bfs,
                 n_times, r_gain, r_normalize):
        super().__init__(dmps, n_updates, n_reps, n_dims, n_bfs,
                         n_times, r_gain, r_normalize)
        self.min_c = self.c2 * 0.8
        self.kl_episilon = 2.0

    def update(self, r, action):
        r = np.array(r)
        act = np.array(action)
        s = np.sum(r, 1)

        res = minimize(reps_dual,
                       x0=10.0,
                       bounds=((self.min_c, None),),
                       args=(self.kl_episilon, s))

        self.c1 = res.x[0] * self.p1.epsilon / (
            1 + self.n_reps * self.p1.epsilon)
        self.c2 = res.x[0]

        if self.r_normalize:
            s = (s - min(s)) / (max(s) - min(s))
            g2 = self.cnt**(self.alpha - 1) * self.c2 / self.r * s
            g1 = self.c1 * s
        else:
            g2 = self.cnt**(self.alpha - 1) * (1 / self.c2) / self.r * s
            g1 = (1 / self.c1) * s
        g2 = (g2 * np.ones((self.n_params, 1)))
        g1 = (g1 * np.ones((self.n_params, 1)))
        act = act.reshape([self.n_reps, self.n_params]).T
        theta_z = self.theta_z.reshape(self.n_params).T
        theta_x = self.theta_x.reshape(self.n_params).T
        std = self.std_eps * np.ones_like(theta_x)

        # Discretize prob_z and prob_x
        prob_z = gauss_pdf(act, theta_z, std)
        prob_x = gauss_pdf(act, theta_x, std)
        # Normalize prob_z and prob_z
        prob_z /= (np.sum(prob_z, 1).T).reshape((-1, 1))
        prob_x /= (np.sum(prob_x, 1).T).reshape((-1, 1))
        # Update prob_z and prob_x by upper level MD
        next_prob_z = np.array(list(map(self.p2.project, prob_z, g2)))
        next_prob_x = np.array(list(map(self.p1.project, prob_x, g1)))
        # Fitting next_theta_z and next_theta_x
        next_theta_z = get_next_theta(act, next_prob_z)
        next_theta_z = next_theta_z.reshape([self.n_dims, self.n_bfs])
        next_theta_x = get_next_theta(act, next_prob_x)
        next_theta_x = next_theta_x.reshape([self.n_dims, self.n_bfs])

        self.theta_z = next_theta_z
        self.theta_x = next_theta_x
        self.theta = (self.cnt**(self.alpha - 1) *
                      self.theta_x + self.r*self.theta_z) /\
                     (self.cnt**(self.alpha - 1)+self.r)
        self.cnt += 1
