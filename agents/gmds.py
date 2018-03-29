import numpy as np
from util import Saver


class GMDS(Saver):
    saved_names = ('theta',)

    def __init__(self, dmps, n_updates, n_reps, n_dims, n_bfs, n_times,
                 r_gain, r_normalize):
        self.dmps = dmps
        self.n_updates = n_updates
        self.n_reps = n_reps
        self.n_dims = n_dims
        self.n_bfs = n_bfs
        self.n_times = n_times
        self.r_normalize = r_normalize

        self.c = r_gain
        self.std_eps = 0  # Operate out of the class
        self.theta = np.zeros([self.n_dims, self.n_bfs])
        self.epsilon = np.zeros([self.n_reps, self.n_dims, self.n_bfs])
        self.action = np.zeros([self.n_reps, self.n_dims, self.n_bfs])
        self.reward = np.zeros((self.n_reps, self.n_times))

    def update(self, r, action):
        r = np.array(r)
        action = np.array(action)
        s = np.sum(r, 1)
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
        self.theta = self.theta+dtheta

    def act(self, obs, t):
        return self.theta

    def act_and_train(self, obs, reward, t, k):
        xi, dxi, ddxi, gdof = obs
        self.reward[k, t] = reward
        if t == 0:
            self.epsilon[k, :, :] = np.random.randn(self.n_dims,
                                                    self.n_bfs)*self.std_eps
        self.action[k, :] = self.theta+self.epsilon[k, :]
        if(k == self.n_reps-1) and (t == self.n_times-1):
            self.update(self.reward, self.action)
        return self.action[k, :]
