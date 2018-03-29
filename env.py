import numpy as np
import dmp
import warnings
import matplotlib
import matplotlib.pyplot as plt
from util import get_end_effector
from abc import ABCMeta, abstractmethod

if matplotlib.get_backend() == 'Qt5Agg' or \
   matplotlib.get_backend() == 'Qt4Agg':
    from painter import Painter
else:
    from simple_painter import Painter


class Env(metaclass=ABCMeta):
    def __init__(self, env):
        pass

    @abstractmethod
    def generate_animation(self, x, y):
        pass

    @abstractmethod
    def render(self, x, y, t):
        pass

    @abstractmethod
    def step(self, action, t):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def set_plot_data(self, t):
        pass


# TODO
class Quadratic(Env):
    def __init__(self, n_dims):
        raise NotImplementedError
        self.n_dims = n_dims
        self.cost_func_order = 2
        self.goal = 1000 * np.ones(n_dims)
        self.xi = []

    def reset(self):
        obs = np.zeros(self.n_dims)
        return obs

    def step(self, action, t):
        obs = action
        self.xi.append(obs)
        reward = np.dot((obs - self.goal)**self.cost_func_order,
                        np.ones(self.n_dims)) / self.cost_func_order
        reward = np.sum(reward)
        # reward =  np.dot((xi-self.goal)**3,np.ones(self.n_dims))/3
        # reward =  np.dot((xi-self.goal)**75,np.ones(self.n_dims))/75
        return obs, reward

    def render(self, x, y, t):
        raise NotImplementedError

    def generate_animation(self, x, y):
        raise NotImplementedError

    def rollouts(self, reps, mean, epsilon):
        data = np.zeros([reps, self.n_dims])
        r = np.zeros([reps, 1])
        for k in range(reps):
            data[k, :] = np.add(mean[k, :, 0, 0], epsilon[k, :, 0, 0])
            r[k, 0] = self.getCost(data[k, :], 0, 0, 0, 0, 0, 0)
        return r

    def set_plot_data(self, t):
        # TODO
        x = y = [0]
        return x, y


class EnvDMP(Env):
    def __init__(self, dmps, n_dims, n_bfs, n_times, duration, dt):
        self.dmps = dmps
        self.n_times = n_times
        self.n_dims = n_dims
        self.n_bfs = n_bfs
        self.ini_theta = np.zeros((self.n_dims, self.n_bfs))
        self.dt = dt
        self.duration = duration
        self.xi = []
        self.prev_action = np.zeros([n_dims, n_bfs])

        self.gxi = self.goal()

        try:
            self.painter = Painter()
        except NameError as e:
            warnings.warn('As your matplotlib backend is {0}, '
                          'pyglet and matplotlib will conflict. '
                          'So, pyglet wasn\'t loaded. '
                          'If you want to use pyglet and motlotlib, '
                          'backend should change Qt5Agg or Qt4Agg. '
                          'Actual error: {1}'.format(matplotlib.get_backend(),
                                                     e))

    def reset_state(self, dmps):
        n_dims = len(dmps)
        for i in range(n_dims):
            dmps[i].reset(0)
            dmps[i].set_goal(self.gxi[i])

    def reset(self):
        self.reset_state(self.dmps)
        obs = self.get_dmp(self.prev_action)
        self.xi = []
        return obs

    def step(self, action, t):
        obs = self.get_dmp(action)
        reward = self._step(action, obs, t)
        self.xi.append(obs[0])
        self.prev_action = action
        return obs, reward

    def get_dmp(self, action):
        xi = np.zeros((self.n_dims, 1))
        dxi = np.zeros((self.n_dims, 1))
        ddxi = np.zeros((self.n_dims, 1))
        g = np.zeros((self.n_dims, self.n_bfs))

        for d in range(self.n_dims):
            xi[d], dxi[d], ddxi[d], g[d] = self.dmps[d].run(action[d])
        return [xi, dxi, ddxi, np.array(g)]

    @abstractmethod
    def goal(self):
        pass


class Arm(EnvDMP):
    def __init__(self, dmps, n_dims, n_bfs, n, duration, dt):
        super().__init__(dmps, n_dims, n_bfs, n, duration, dt)
        plt.ion()
        self.viapoint = (0.5, 0.5)

    def goal(self):
        dof = self.n_dims
        gxi = [np.pi * (1.0 - (dof - 1.0) / dof) for i in range(dof)]
        gxi[0] = gxi[0] / 2.0
        return gxi

    def generate_animation(self, x, y):
        ims = []
        for i in range(len(x)):
            im = plt.scatter(x[i][0:self.n_dims], y[i][0:self.n_dims], c='b')
            im2 = plt.plot(x[i], y[i], 'g')
            im3 = plt.text(-0.8, -0.8, str(i * 10) + "ms", fontsize=17)
            im4 = plt.scatter([0.5], [0.5], c='r')
            ims.append([im] + im2 + [im3] + [im4])
        return ims

    def render(self, x, y, t):
        if self.painter is not None:
            self.viapoint = self.painter.draw(x, y, t)
        else:
            warnings.warn('As matplotlib backend is not appropriate, '
                          'You can\'t use render option.')

    def _step(self, action, obs, t):
        xi, dxi, ddxi, gdof = obs
        ee_x, ee_y = get_end_effector(xi, self.n_dims)
        cost = ((self.viapoint[0] - ee_x)**2 +
                (self.viapoint[1] - ee_y)**2) * (10**8)
        r_numerator = r_denominator = 0

        for d in range(self.n_dims):
            g_t_g = np.sum(np.array(gdof[d])**2)
            g_t_theta_eps = np.array(gdof[d]) * np.array(action[d])
            m_theta_eps = gdof[d] * g_t_theta_eps / (g_t_g + 1.e-10)
            r_numerator += (self.n_dims + 1 - (d + 1)) * (
                0.1 * np.sum(ddxi[d]**2) + 0.5 * np.sum(m_theta_eps**2))
            r_denominator += self.n_dims + 1 - (d + 1)
        reward = (self.dt*t <= self.duration)*(r_numerator/r_denominator) + \
                 (self.dt*t == 0.3)*cost
        return reward

    def set_plot_data(self, t):
        x, y = np.zeros(self.n_dims + 1), np.zeros(self.n_dims + 1)
        for d in range(self.n_dims + 1):
            x[d] = np.sum(
                [np.cos(np.sum(self.xi[t][:i + 1]))
                 for i in range(d)]) * 1.0 / self.n_dims
            y[d] = np.sum(
                [np.sin(np.sum(self.xi[t][:i + 1]))
                 for i in range(d)]) * 1.0 / self.n_dims
        return x, y


class Point(EnvDMP):
    def __init__(self, dmps, n_dims, n_bfs, n, duration, dt):
        super().__init__(dmps, n_dims, n_bfs, n, duration, dt)
        plt.ion()
        self.viapoint = (0.5, 0.2)

    def goal(self):
        return np.ones(self.n_dims)

    def generate_animation(self, x, y):
        ims = []
        for i in range(len(x)):
            im = plt.scatter(x[i], y[i])
            im2 = plt.text(-0.8, -0.8, str(i * 10) + "ms", fontsize=17)
            im3 = plt.scatter([self.viapoint[0]], [self.viapoint[1]], c='r')
            ims.append([im] + [im2] + [im3])
        return ims

    def render(self, x, y, t):
        # TODO
        raise NotImplementedError

    def _step(self, action, obs, t):
        xi, dxi, ddxi, gdof = obs
        cost = ((self.viapoint[0] - xi[0][0])**2 +
                (self.viapoint[1] - xi[1][0])**2) * (10**10)
        r = 0
        for d in range(self.n_dims):
            g_t_g = np.sum(np.array(gdof[d])**2)
            g_t_theta_eps = np.array(gdof[d]) * np.array(action[d])
            m_theta_eps = gdof[d] * g_t_theta_eps / (g_t_g + 1.e-10)
            r += 0.5 * 1000 * np.sum(ddxi[d]**2) + 0.5 * np.sum(m_theta_eps**2)

        reward = (self.dt*t <= self.duration)*r + \
                 (self.dt*t == 0.25)*cost

        return reward

    def set_plot_data(self, t):
        x, y = np.zeros(self.n_dims), np.zeros(self.n_dims)
        x[0] = self.xi[t][0]
        y[0] = self.xi[t][1]
        return x, y


class OneDof:
    def __init__(self, RL):
        # TODO
        raise NotImplementedError
        self.dt = RL.dt
        self.n_dims = RL.n_dims
        self.duration = RL.duration

    def init(self):
        # Set each joint angle[rad] at goal
        self.gxi = self.goal()

        # Set dmp
        for i in range(self.RL.n_dims):
            self.RL.dmps[i] = dmp.DMP()
            self.RL.dmps[i].init(self.RL.n_bfs, 1)
            self.RL.dmps[i].reset()
            self.RL.dmps[i].set_goal(self.gxi[i])
            self.RL.theta[i] = self.RL.dmps[i].minimize_jerk(self.RL.theta[i])

    def goal(self):
        gxi = [1]
        return gxi

    def make_animation(self, Xdata, Ydata):
        ims = []
        for i in range(len(Xdata)):
            im = plt.scatter(Xdata[i], Ydata[i])
            im2 = plt.plot(Xdata[i], Ydata[i], 'g')
            im3 = plt.text(-0.8, -0.8, str(i * 10) + "ms", fontsize=17)
            im4 = plt.plot([0.0, np.cos(1.0)], [0.0, np.sin(1.0)], 'r')
            ims.append([im] + im2 + [im3] + im4)
        return ims

    def get_cost(self, gxi, obs, gDof, _epsilon, theta, t):
        dt = self.dt
        dof = self.n_dims
        duration = self.duration
        xi, dxi, ddxi = obs[t]
        tCost = 10000 * (dxi**2 + 10 * (gxi[0] - xi[0])**2)
        gTg = []
        gTeps = []
        Meps = []
        gTg = np.sum(np.array(gDof[0]) * np.array(gDof[0]))
        gTeps = np.array(gDof[0]) * np.array(_epsilon[0][t])
        Meps = gDof[0] * gTeps / gTg

        f = np.squeeze(ddxi)
        # np.squeeze(ddXi-np.sum(gDof[0]*(theta[0]+_epsilon[0][t])))
        tmp = (0.5 * (f**2) + 5000 * np.sum(
            (theta[0] + Meps) * (theta[0] + Meps)))

        return [(dt * t < duration) * tmp + (dt * t > duration) * tCost]

    def setPlotData(self, xi, t):
        x, y = np.zeros(self.n_dims + 1), np.zeros(self.n_dims + 1)
        x[1] = np.cos(np.sum(xi[t][0])) * 1.0 / self.n_dims
        y[1] = np.sin(np.sum(xi[t][0])) * 1.0 / self.n_dims
        # x=Time[s], y=Position[rad]
        #        x, y = np.zeros(self.n_dims), np.zeros(self.n_dims)
        #        x[0]=float(t)/float(self.n)
        #        y[0]=xi[t][0]
        return x, y
