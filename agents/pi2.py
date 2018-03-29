import numpy as np
from util import Saver


class PI2(Saver):
    saved_names = ('theta',)

    def __init__(self, dmps, n_updates, n_reps, n_dims, n_bfs,
                 n_times, r_gain, r_normalize):
        self.dmps = dmps
        self.n_updates = n_updates
        self.n_reps = n_reps
        self.n_dims = n_dims
        self.n_bfs = n_bfs
        self.n_times = n_times
        self.time_index = range(self.n_times)
        self.r_normalize = r_normalize

        self.c = r_gain
        self.std_eps = 0  # Operate out of the class: TODO What is best representation...?
        self.G = np.zeros(
            (self.n_reps, self.n_dims, self.n_times, self.n_bfs))
        self.psi = np.zeros((self.n_reps, self.n_times, self.n_bfs))
        self.theta = np.zeros([self.n_dims, self.n_bfs])
        self.epsilon = np.zeros(
            [self.n_reps, self.n_dims, self.n_times, self.n_bfs])
        self.action = np.zeros(
            [self.n_reps, self.n_dims, self.n_times, self.n_bfs])
        self.reward = np.zeros((self.n_reps, self.n_times))

    def update(self, r, action):
        g = np.array(self.G)
        r = np.array(r)
        action = np.array(action)
        s = np.rot90(np.cumsum(np.rot90(r, k=2), 1), k=2)
        if self.r_normalize:
            max_s = np.max(s, 0)
            min_s = np.min(s, 0)
            exp_s = np.exp(-self.c*(s-(min_s*np.ones((self.n_reps,
                                                      self.n_times))))
                           / ((max_s-min_s)*np.ones((self.n_reps,
                                                     self.n_times))))
        else:
            exp_s = np.exp(-1/self.c*s)

        p = exp_s/(np.sum(exp_s, 0)*np.ones((self.n_reps, self.n_times)))
        p_m_eps = np.zeros(
            (self.n_reps, self.n_dims, self.n_times, self.n_bfs))
        for k in range(self.n_reps):
            for d in range(self.n_dims):
                g_t_eps = np.sum(g[k][d]*(action[k, d]-self.theta[d]), 1)
                g_t_g = np.sum(g[k][d]**2, 1)
                tmp = (p[k]*g_t_eps/(g_t_g+1.e-10)).reshape(1, -1)
                p_m_eps[k][d] = g[k][d] * \
                    (tmp.T*np.ones((self.n_times, self.n_bfs)))

        dtheta = np.squeeze(np.sum(p_m_eps, 0))
        weight = np.arange(self.n_times, 0, -1).T
        weight = (weight*np.ones((self.n_bfs, self.n_times))).T * \
            np.array(self.psi[0])
        weight = weight / (np.ones((self.n_times, self.n_bfs)) *
                           np.sum(weight, 0))
        weight = np.tile(np.reshape(weight, (1, self.n_times, self.n_bfs)),
                         (self.n_dims, 1, 1))

        weighted_dtheta = np.sum(dtheta*weight, 1)
        weighted_dtheta = np.squeeze(weighted_dtheta)

        self.theta = self.theta+weighted_dtheta

    def act(self, obs, t):
        return self.theta

    def act_and_train(self, obs, reward, t, k):
        xi, dxi, ddxi, gdof = obs
        self.G[k, :, t] = gdof
        self.psi[k, t] = self.dmps[0].psi
        self.reward[k, t] = reward
        self.epsilon[k, :, t, :] = np.random.randn(
            self.n_dims, self.n_bfs)*self.std_eps
        self.action[k, :, t] = self.theta+self.epsilon[k, :, t]
        if(k == self.n_reps-1) and (t == self.n_times-1):
            self.update(self.reward, self.action)
        return self.action[k, :, t]


class PI2IndependTime(PI2):
    def __update(self, r, epsilon, theta):
        next_theta = [[0 for _ in range(self.n_bfs)]
                      for _ in range(self.n_dims)]
        # dtheta = [[0 for _ in range(self.n_bfs)] for _ in range(self.n_dims)]
        c = 10
        n_times = len(self.time_index)
        s = np.rot90(
            np.rot90((np.cumsum(np.rot90(np.rot90(np.array(r).T)), 0))))
        max_s = np.max(s, 1)
        min_s = np.min(s, 1)

        exp_s = np.exp(-c*(s-(min_s*np.ones((self.n_reps, n_times))).T)
                       / ((max_s-min_s)*np.ones((self.n_reps, n_times))).T)

        p = exp_s/(np.sum(exp_s, 1)*np.ones((self.n_reps, n_times))).T
        p = np.tile(p[0, :], (n_times, 1))

        for d in range(self.n_dims):
            for k in range(self.n_reps):
                # dtheta[d] = dtheta[d]+P[0,k]*epsilon[k][d][0]
                next_theta[d] = next_theta[d] + \
                    p[0, k]*(np.add(theta[0, d, 0, :], epsilon[k, d, 0, :]))

        # theta[0, :, 0, :] = theta[0, :, 0, :] + dtheta[:]
        theta[0, :, 0, :] = next_theta[:]

        return theta

    def _act(self, reps, theta, std_eps):
        epsilon = np.zeros([reps, self.n_dims, self.n_times, self.n_bfs])
        # Generate exploration noise
        for t in range(self.n_times):
            if t == 0:
                epsilon[:, :, t, :] = np.random.randn(
                    reps, self.n_dims, self.n_bfs)*std_eps
            else:
                epsilon[:, :, t, :] = epsilon[:, :, 0, :]
