import numpy as np
from util import Saver
from agents.util import (get_next_theta, gauss_pdf, Projection)


class AMDS(Saver):
    saved_names = ('theta', )

    def __init__(self, dmps, n_updates, n_reps, n_dims, n_bfs, n_times,
                 r_gain, r_normalize):
        self.dmps = dmps
        self.n_updates = n_updates
        self.n_reps = n_reps
        self.n_dims = n_dims
        self.n_params = n_bfs * n_dims
        self.n_bfs = n_bfs
        self.n_times = n_times
        self.r_normalize = r_normalize

        self.theta = np.zeros([self.n_dims, self.n_bfs])
        self.theta_z = np.zeros([self.n_dims, self.n_bfs])
        self.theta_x = np.zeros([self.n_dims, self.n_bfs])
        self.epsilon = np.zeros([self.n_reps, self.n_dims, self.n_bfs])
        self.action = np.zeros([self.n_reps, self.n_dims, self.n_bfs])
        self.reward = np.zeros((self.n_reps, self.n_times))

        self.r = 3
        self.alpha = 2
        if self.r < self.alpha:
            raise Exception('r needs to be larger than alpha (default 2)')

        epsilon = 0.3
        self.p1 = Projection(epsilon)
        self.p2 = Projection(0)
        self.c1 = r_gain*self.p1.epsilon / (1 + self.n_reps*self.p1.epsilon)
        self.c2 = r_gain

        self.cnt = 0

    def update(self, r, action):
        r = np.array(r)
        act = np.array(action)
        s = np.sum(r, 1)
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

    def act(self, obs, t):
        return self.theta

    def act_and_train(self, obs, reward, t, k):
        xi, dxi, ddxi, gdof = obs
        self.reward[k, t] = reward
        if t == 0:
            self.epsilon[k] = np.random.randn(self.n_dims,
                                              self.n_bfs) * self.std_eps
        self.action[k, :] = self.theta + self.epsilon[k, :]
        if (k == self.n_reps - 1) and (t == self.n_times - 1):
            self.update(self.reward, self.action)
        return self.action[k, :]
