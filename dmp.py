import numpy as np


# Radial Basis Function
class RBF:
    def __init__(self, n_bfs, duration,
                 second_order, alpha_x, alpha_z):
        self.n_bfs = n_bfs
        self.second_order = second_order
        t = np.linspace(0, duration, n_bfs+1)
        if self.second_order:
            # 2nd canonical system
            # d^2x/dt^2 + alpha*dx/dt + 0.25*alpha^2*x = 0,
            # x(0) = 1, dx/dt(0) = 0
            # => x(t) = (1+alpha/2*t)*exp(-alpha/2*t)
            center = (1 + alpha_z / 2.0 * t) *\
                     np.exp(-alpha_z / 2.0 * t)
        else:
            # 1st canonical system
            # dx/dt + alpha*x = 0, x(0) = 1
            # => x(t) = exp(-alpha*t)
            center = np.exp(-alpha_x * t)

        center_diff = (np.diff(center) * 0.55)**2
        self.center = center[0:len(center)-1]
        self.width = 1 / center_diff

    def kernel(self, x):
        return np.exp(-0.5*self.width *
                      ((x * np.ones(self.n_bfs) - self.center)**2))


# Weighted Linear Regression
class WLR:
    def weight(self, state, target, kernel):
        sks = state**2 * np.ones_like(kernel) * kernel
        stk = state*target * np.ones_like(kernel) * kernel
        weight = np.sum(stk, 0) / (np.sum(sks, 0) + 1.e-10)
        return weight


# Dynamic Movement Primitives
class DMP:
    def __init__(self, n_bfs, duration, tau, dt, second_order):
        self.n_bfs = n_bfs
        self.duration = duration
        self.tau = duration/tau
        self.dt = dt
        self.second_order = second_order

        # Initialize the time constants
        self.alpha_z = 25
        self.beta_z = self.alpha_z / 4.0
        self.alpha_g = self.alpha_z / 2.0
        self.alpha_x = self.alpha_z / 3.0
        self.alpha_v = self.alpha_z
        self.beta_v = self.beta_z
        # Initialize the state variables
        self.z = 0
        self.y = 0
        self.x = 0
        self.v = 0
        # Initialize the current goal state
        self.g = 0
        # Initialize the terminate goal state
        self.goal = 0
        # Initialize the current start state of the primitive
        self.y0 = 0
        # Initialize radial basis function class
        self.rbf = RBF(n_bfs, duration, self.second_order,
                       self.alpha_x, self.alpha_z)
        # Initialize weighted linear regression class
        self.wlr = WLR()

    def set_goal(self, goal, update=True):
        self.goal = goal
        if not self.second_order:
            self.g = self.goal
        if update:
            self.x = 1
            self.y0 = self.y

    def reset(self, start=0):
        y = start
        self.z = 0
        self.y = y
        self.x = 0
        self.v = 0
        self.goal = y
        self.g = y
        self.y0 = y
        self.s = 1

    def update_vx(self, v, x, cc=0):
        if self.second_order:
            vd = (self.alpha_v * (self.beta_v * (-x) - v) + cc) * self.tau
            xd = v * self.tau
        else:
            vd = 0
            xd = self.alpha_x * (-x) * self.tau
        v += vd * self.dt
        x += xd * self.dt
        return v, x

    def update_zy(self, theta, z, y, g, diff_goal,
                  basis, ct=0):
        zd = ((self.alpha_z * (self.beta_z * (g - y) - z))
              + np.sum(basis*theta) * diff_goal + ct) * self.tau
        yd = z * self.tau
        ydd = zd * self.tau
        z += zd * self.dt
        y += yd * self.dt
        return z, y, yd, ydd

    def update_g(self, g, goal):
        gd = self.alpha_g * (goal - g)
        g += gd * self.dt
        return g

    def run(self, theta, ct=0, cc=0):
        psi = self.rbf.kernel(self.x)

        if self.second_order:
            basis = self.v * psi / np.sum(psi + 1.e-10)
        else:
            basis = self.x * psi / np.sum(psi + 1.e-10)

        self.v, self.x = self.update_vx(self.v, self.x, cc)
        self.z, self.y, yd, ydd = self.update_zy(theta, self.z,
                                                 self.y, self.g,
                                                 self.goal-self.y0, basis)
        self.g = self.update_g(self.g, self.goal)
        self.psi = psi
        return self.y, yd, ydd, basis

    def minimize_jerk_step(self, x, xd, xdd, tau):
        if tau < self.dt:
            return x, xd, xdd
        dist = self.goal - x
        a1 = 0
        a0 = xdd * tau**2
        v1 = 0
        v0 = xd * tau

        t1 = self.dt
        t2 = self.dt**2
        t3 = self.dt**3
        t4 = self.dt**4
        t5 = self.dt**5

        c1 = (6.0 * dist + (a1 - a0) / 2.0 - 3.0 * (v0 - v1)) / tau**5
        c2 = (-15.0 * dist +
              (3.0 * a0 - 2.0 * a1) / 2.0 + 8.0 * v0 + 7.0 * v1) / tau**4
        c3 = (10.0 * dist + (a1 - 3.0 * a0) / 2.0 - 6.0 * v0 - 4 * v1) / tau**3
        c4 = xdd / 2.0
        c5 = xd
        c6 = x

        x = c1 * t5 + c2 * t4 + c3 * t3 + c4 * t2 + c5 * t1 + c6
        xd = 5.0 * c1 * t4 + 4.0 * c2 * t3 + 3.0 * c3 * t2 + 2.0 * c4 * t1 + c5
        xdd = 20.0 * c1 * t3 + 12.0 * c2 * t2 + 6.0 * c3 * t1 + 2.0 * c4

        return x, xd, xdd

    def minimize_jerk(self):
        self.reset()
        self.set_goal(self.goal)
        timestep = int(self.tau / self.dt)
        ts = np.zeros((timestep, 1))
        tds = np.zeros((timestep, 1))
        tdds = np.zeros((timestep, 1))
        t = td = tdd = 0
        for i in range(timestep):
            t, td, tdd = self.minimize_jerk_step(t, td, tdd,
                                                 self.tau - i * self.dt)
            ts[i], tds[i], tdds[i] = t, td, tdd
        theta = self.fitting(ts, tds, tdds)
        return theta

    def fitting(self, t, td, tdd):
        y0 = t[0]
        goal = t[-1]

        xs = np.zeros((t.size, 1))
        vs = np.zeros((t.size, 1))
        gs = np.zeros((t.size, 1))
        x = 1
        v = 0
        if self.second_order:
            g = goal
        else:
            g = y0
        for i in range(t.size):
            xs[i] = x
            vs[i] = v
            gs[i] = g
            v, x = self.update_vx(v, x)
            g = self.update_g(g, goal)

        f_target = (tdd/self.tau**2 -
                    self.alpha_z*(self.beta_z*(gs-t)-td/self.tau))
        psi = np.array([self.rbf.kernel(x) for x in xs])

        if self.second_order:
            theta = self.wlr.weight(vs, f_target, psi)
            basis = np.array([vs * p / np.sum(p) for p in psi])
        else:
            theta = self.wlr.weight(xs, f_target, psi)
            basis = np.array([xs * p / np.sum(p) for p in psi])

        y, yd, ydd = self.prediction(t, theta, y0, gs, basis)
        return theta

    def prediction(self, t, theta, y0, gs, basis):
        ys = np.zeros_like(t)
        yds = np.zeros_like(t)
        ydds = np.zeros_like(t)
        ydd = 0
        yd = z = 0
        y = y0
        for i in range(t.size):
            ydds[i] = ydd
            yds[i] = yd
            ys[i] = y
            z, y, yd, ydd = self.update_zy(theta,
                                           z, y, gs[i], 1, basis[i])
        return ys, yds, ydds
