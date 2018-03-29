import numpy as np
from util import Saver
from agents.util import (get_next_theta, gmm_rand, gauss_pdf, gmm_pdf,
                         DebugAnimation, Projection)


class MDS(Saver):
    saved_names = ('theta', )

    def __init__(self, n_updates, n_reps, n_dims, n_bfs, n_times, r_gain,
                 r_normalize):
        self.n_updates = n_updates
        self.n_reps = n_reps
        self.n_dims = n_dims
        self.n_params = n_bfs * n_dims
        self.n_bfs = n_bfs
        self.n_times = n_times
        self.r_normalize = r_normalize

        self.theta = np.zeros([self.n_dims, self.n_bfs])
        self.epsilon = np.zeros([self.n_reps, self.n_dims, self.n_bfs])
        self.action = np.zeros([self.n_reps, self.n_dims, self.n_bfs])
        self.reward = np.zeros((self.n_reps, self.n_times))

        self.p = Projection(0)
        self.c = r_gain

    def update(self, r, action):
        r = np.array(r)
        act = np.array(action)
        s = np.sum(r, 1)
        if self.r_normalize:
            g = self.c * ((s - min(s)) / (max(s) - min(s)))
        else:
            g = (1 / self.c) * s
        g = (g * np.ones((self.n_params, 1)))
        act = act.reshape([self.n_reps, self.n_params]).T
        theta_x = self.theta.reshape(self.n_params).T
        std = self.std_eps * np.ones_like(theta_x)

        # Discretize prob_x
        prob_act = gauss_pdf(act, theta_x, std)
        # Normalize prob_x
        prob_act /= (np.sum(prob_act, 1).T).reshape((-1, 1))
        # Update prob_x by upper level MD
        next_prob_act = np.array(list(map(self.p.project, prob_act, g)))
        # Fitting next_theta_x
        next_theta_x = get_next_theta(act, next_prob_act)
        next_theta_x = next_theta_x.reshape([self.n_dims, self.n_bfs])
        self.theta = next_theta_x

    def act(self, obs, t):
        return self.theta

    def act_and_train(self, obs, reward, t, k):
        self.reward[k, t] = reward
        if t == 0:
            self.epsilon[k] = np.random.randn(self.n_dims,
                                              self.n_bfs) * self.std_eps
            self.action[k] = self.theta + self.epsilon[k]
        if (k == self.n_reps - 1) and (t == self.n_times - 1):
            self.update(self.reward, self.action)
        return self.action[k, :]


class MDSKDE(MDS):
    saved_attributes = ('theta', )

    def __init__(self, n_updates, n_reps, n_dims, n_bfs, n_times,
                 r_gain, r_normalize):
        super().__init__(n_updates, n_reps, n_dims, n_bfs, n_times, r_gain,
                         r_normalize)
        self.theta = np.zeros([self.n_reps, self.n_dims, self.n_bfs])
        self.theta_mean = np.zeros([self.n_dims, self.n_bfs])
        self.prob = np.ones((self.n_reps, self.n_params))
        self.prob /= (np.sum(self.prob.T, 1)).reshape((1, -1))
        self.da = DebugAnimation()

    def update(self, r, action):
        r = np.array(r)
        act = np.array(action)
        s = np.sum(r, 1)

        if self.r_normalize:
            s = (s - min(s)) / (max(s) - min(s))

        # Set s as g
        g = self.c * s
        g = (g * np.ones((self.n_params, 1)))
        act = act.reshape([self.n_reps, self.n_params]).T
        theta_x = self.theta.reshape(self.n_reps, self.n_params).T
        prob_x = self.prob.reshape((self.n_reps, self.n_params)).T
        std = self.std_eps * np.ones_like(theta_x)

        # Discretize prob_act
        prob_act = gmm_pdf(act, prob_x, theta_x, std=std)
        # Normalize prob_act
        prob_act /= (np.sum(prob_act, 1).T).reshape((-1, 1))

        # Update prob_x by upper level MD
        next_prob_act = np.array(list(map(self.p.project, prob_act, g)))

        # --- Methods for debug ---
        # action = self.action.reshape(self.n_reps, self.n_params).T
        # self.da.plot_animation([theta_x[0]], [prob_x[0]], [prob_act[0]],
        #                        [action[0]], [g[0]], [next_prob_act[0]],
        #                        [std[0]], 1)
        # self.da.plot_animation(theta_x, prob_x, prob_act, action, g,
        #                        next_prob_act, std, self.n_params)
        # --- Methods for debug ---

        # Fitting next_theta_x
        next_theta_x = act  # get_next_theta(act, next_prob_act)
        theta_mean = get_next_theta(act, next_prob_act)

        next_theta_x = (next_theta_x.T).reshape(
            [self.n_reps, self.n_dims, self.n_bfs])
        next_prob_act = (next_prob_act.T).reshape(
            [self.n_reps, self.n_dims, self.n_bfs])
        theta_mean = (theta_mean.T).reshape([self.n_dims, self.n_bfs])

        self.theta_mean = theta_mean
        self.theta = next_theta_x
        self.prob = next_prob_act

    def act_and_train(self, obs, reward, t, k):
        self.reward[k, t] = reward
        prob = self.prob.reshape((self.n_reps, self.n_params)).T
        theta = self.theta.reshape(self.n_reps, self.n_params).T
        std = self.std_eps * np.ones_like(theta)
        if t == 0:
            # self.epsilon[k] = np.random.randn(self.n_dims, self.n_bfs) * (
            #     self.std_eps)
            # self.action[k] = self.theta_mean + self.epsilon[k]
            self.action[k] = gmm_rand(prob, theta, std,
                                      1).reshape([self.n_dims, self.n_bfs])

        if (k == self.n_reps - 1) and (t == self.n_times - 1):
            self.update(self.reward, self.action)
        return self.action[k, :]
