import unittest
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Projection:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def project(self, x, g):
        y = (x+self.epsilon)*np.exp(-g)
        sort_y = sorted(y)
        sum_y = np.sum(y)
        d = len(x)
        j = 0
        while (1+self.epsilon*(d-j))*sort_y[j]-self.epsilon*sum_y <= 0:
            sum_y -= sort_y[j]
            j += 1
        return np.maximum(0, -self.epsilon + y/sum_y*(1+self.epsilon*(d-j)))


def reps_dual(eta, epsilon, costs):
    min_cost = np.min(costs)
    exp_denom = -(costs - min_cost)
    g = epsilon*eta + eta * \
        np.log((1.0 / len(costs)) * np.sum(np.exp(exp_denom / eta))) + \
        (-min_cost)
    return g


def gauss_func(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def gauss_rand(mean, std, num):
    return mean + np.random.randn(num) * std


def gmm_rand(weight, mean, std, num):
    dims = weight.shape[0]
    rand = np.zeros((dims, num))
    for i, w in enumerate(weight):
        w = w / np.sum(w)
        for j in range(num):
            rand_index = np.random.choice(list(range(len(w))), p=w)
            gauss_m = mean[i, rand_index]
            gauss_s = std[i, rand_index]
            rand[i, j] = gauss_rand(gauss_m, gauss_s, 1)
    return rand


def gauss_pdf(sample, mean, std):
    return np.array([
        norm.pdf(x=sa, loc=m, scale=st)
        for sa, m, st in zip(sample, mean, std)
    ])


def gmm_pdf_1dim(sample, weight, mean, std=1):
    """ For 1 dimensional parameter
    """
    mixed_weight = []
    for m, w in zip(mean, weight):
        mixed_weight.append(w * norm.pdf(x=sample, loc=m, scale=std))
    return np.sum(mixed_weight, 0)


def gmm_pdf(sample, weight, mean, std=1):
    weight = [gmm_pdf_1dim(s, w, m) for m, w, s in zip(mean, weight, sample)]
    return np.array(weight)


def estimate(x, y, mean, sigma, mode):
    if mode == -1:
        return mean, sigma
    elif mode == 0:
        p_opt, p_cov = curve_fit(
            gauss_func,
            x,
            y,
            p0=[1, mean, sigma],
            maxfev=100000,
            check_finite=False)
        return p_opt[1], p_opt[2]
    elif mode == 1:
        p_opt, p_cov = curve_fit(
            lambda x, a, x0: gauss_func(x, a, x0, sigma),
            x,
            y,
            p0=[1, mean],
            maxfev=100000,
            check_finite=False)
        return p_opt[1], sigma
    else:
        raise NotImplementedError


def get_next_theta(action, next_prob):
    next_theta = []
    for x, y in zip(action, next_prob):
        mean = np.sum(x * y)
        sigma = np.sum(y * (x - mean)**2)
        try:
            mean, sigma = estimate(x, y, mean, sigma, -1)
        except RuntimeError as e:
            print(e)
        except ValueError as e:
            print(e)
        next_theta.append(mean)
    return np.array(next_theta)


def plot_gmm(x,
             mean,
             std,
             weight,
             sample_x,
             sample_y,
             n_plots,
             disp_comp=True,
             disp_mixed=False):
    ims = []
    for i, (m, st, w, sa_x, sa_y) in enumerate(
            zip(mean, std, weight, sample_x, sample_y)):
        plt.subplot(n_plots, 1, i + 1)
        ims.append(plt.scatter(sa_x, sa_y, color="orangered"))
        yy = []
        for j, (mm, ss, ww) in enumerate(zip(m, st, w)):
            y = gauss_func(x, ww, mm, ss)
            yy.append(y)
            if disp_comp:
                ims.extend(plt.plot(x, y, color="skyblue"))
        if disp_mixed:
            ims.extend(plt.plot(x, np.sum(yy, 0), color="darkblue"))
    return ims


def plot_discrete(sample, prob, n_plots, color):
    ims = []
    for i, (s, p) in enumerate(zip(sample, prob)):
        plt.subplot(n_plots, 1, i + 1)
        ims.extend(plt.bar(s, p, color=color))
    return ims


class Debug:
    outdir = None
    n_updates = None

    @staticmethod
    def set_outdir(dir):
        Debug.outdir = dir

    @staticmethod
    def set_n_updates(num):
        Debug.n_updates = num


class DebugAnimation:
    import time

    def __init__(self):
        self.num = 0
        self.ims = []
        self.fig = plt.figure()

    def plot_and_save_figure(self, theta_x, prob_x, prob_act, action, g,
                             next_prob_act, std, n_params):
        x = np.linspace(-10, 10, 1000)
        plt.figure()
        plot_gmm(x, theta_x, std, prob_x, action, g, n_params)
        plot_discrete(action, prob_act, n_params, "forestgreen")

        plot_discrete(action, next_prob_act, n_params, "orange")
        plt.savefig('{}/{}.png'.format(Debug.outdir, self.num))

        self.num += 1

    def plot_animation(self, theta_x, prob_x, prob_act, action, g,
                       next_prob_act, std, n_params):
        x = np.linspace(-10, 10, 100)
        im1 = plot_gmm(
            x,
            theta_x,
            std,
            prob_x,
            action,
            g,
            n_params,
            disp_comp=False,
            disp_mixed=True)
        im2 = plot_discrete(action, prob_act, n_params, "forestgreen")
        im3 = plot_discrete(action, next_prob_act, n_params, "orange")
        self.num += 1
        self.ims.append(im1 + im2 + im3)
        if self.num == Debug.n_updates:
            self.save_animation()

    def save_animation(self):
        ani = animation.ArtistAnimation(self.fig, self.ims, interval=100)
        ani.save("{}/output.mp4".format(Debug.outdir), writer='ffmpeg')


class TestUtil(unittest.TestCase):
    def test_gmm_rand(self):
        n_reps = 2
        n_params = 3
        weight = np.array([[1, 1, 1], [2, 2, 2]])
        mean = np.array([[1, 2, 3], [3, 2, 1]])
        num = 100

        weight = weight.reshape((n_reps, n_params)).T
        mean = mean.reshape(n_reps, n_params).T
        std = np.ones_like(mean)

        sample = gmm_rand(weight, mean, std, num)

        x = np.linspace(-5, 5, 50)
        plt.figure()
        plot_gmm(x, mean, std, weight, sample, n_params)

        theoretical_exp = np.sum(weight * mean, 1)
        actual_exp = np.sum(sample, 1) / num
        theoretical_exp = theoretical_exp / np.sum(theoretical_exp)
        actual_exp = actual_exp / np.sum(actual_exp)
        print(theoretical_exp, actual_exp)

    def test_gmm_pdf(self):
        n_reps = 2
        n_params = 3
        weight = np.array([[1, 1, 1], [2, 2, 2]])
        mean = np.array([[1, 2, 3], [3, 2, 1]])

        weight = weight.reshape((n_reps, n_params)).T
        mean = mean.reshape(n_reps, n_params).T
        std = np.ones_like(mean)

        sample = gmm_rand(weight, mean, std, n_reps)

        # test = gmm_pdf_1dim(sample[0], weight[0], mean[0], std[0])
        prob = gmm_pdf(sample, weight, mean, std)
        x = np.linspace(-10, 10, 100)

        plt.figure()
        plot_gmm(x, mean, std, weight, sample, n_params, disp_mixed=True)
        plot_discrete(sample, prob, n_params, True)
        plt.show()


if __name__ == "__main__":
    unittest.main()
