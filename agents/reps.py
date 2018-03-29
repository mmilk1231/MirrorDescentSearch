import numpy as np
from scipy.optimize import minimize
from agents.gmds import GMDS
from agents.util import reps_dual


class REPS(GMDS):
    def __init__(self, dmps, n_updates, n_reps, n_dims, n_bfs,
                 n_times, r_gain, r_normalize):
        super().__init__(dmps, n_updates, n_reps, n_dims, n_bfs,
                         n_times, r_gain, r_normalize)
        self.min_c = self.c * 0.5
        self.kl_epsilon = 2.0

    def update(self, r, action):
        r = np.array(r)
        action = np.array(action)
        s = np.sum(r, 1)

        res = minimize(reps_dual,
                       x0=10.0,
                       bounds=((self.min_c, None),),
                       args=(self.kl_epsilon, s))
        self.c = res.x[0]
        if self.r_normalize:
            max_s = np.max(s, 0)
            min_s = np.min(s, 0)
            exp_s = np.exp(-self.c*(s-(min_s*np.ones(self.n_reps)))
                           / ((max_s-min_s)*np.ones(self.n_reps)))
        else:
            exp_s = np.exp(-1/self.c*s)

        p = exp_s/(np.sum(exp_s, 0)*np.ones(self.n_reps))
        p_eps = np.zeros((self.n_reps, self.n_dims, self.n_bfs))
        for k in range(self.n_reps):
            for d in range(self.n_dims):
                p_eps[k][d] = p[k]*(action[k, d]-self.theta[d])
        dtheta = np.sum(p_eps, 0)
        self.theta = self.theta + dtheta
