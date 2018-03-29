import os
from abc import abstractproperty
import numpy as np
import matplotlib.animation as anim
import matplotlib.pyplot as plt


def plot_cost(reward_list, rollouts, color="blue", label=None):
    x = rollouts
    y = reward_list
    # plt.xscale("log")
    plt.yscale('log')
    plt.xlabel('Number of Roll-Outs')
    plt.ylabel('Cost')
    if label is None:
        plt.plot(x, y, color=color)
    else:
        plt.plot(x, y, label=label, color=color)
        plt.legend()


def plot_animation(path, xs, ys, env):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True, linestyle='-', color='0.75')
    line, = ax.plot([], [], lw=2)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

    # Generate animation
    ims = env.generate_animation(xs, ys)
    ani = anim.ArtistAnimation(fig, ims, interval=10, repeat_delay=1000)
    ani.save(path + '/trajectory.mp4', fps=15)
    plt.show()


def get_end_effector(xi, n_dims):
    if n_dims == 1:
        ee_x = np.cos(xi[0])
        ee_y = np.sin(xi[0])
    else:
        ee_x = np.sum([np.cos(np.sum(xi[:i + 1]))
                       for i in range(n_dims)]) * (1.0 / n_dims)
        ee_y = np.sum([np.sin(np.sum(xi[:i + 1]))
                       for i in range(n_dims)]) * (1.0 / n_dims)
    return ee_x, ee_y


class Saver:
    @abstractproperty
    def saved_names(self):
        pass

    def save(self, dirname):
        os.makedirs(dirname, mode=0o777, exist_ok=True)
        for name in self.saved_names:
            np.save(os.path.join(dirname, '{}'.format(name)),
                    getattr(self, name))

    def load(self, dirname):
        for name in self.saved_names:
            setattr(self, name,
                    np.load(os.path.join(dirname, '{}.npy'.format(name))))
