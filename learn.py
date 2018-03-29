import numpy as np
import sys
import os
import unittest
import glob
import json

from dmp import DMP
from agents.pi2 import PI2
from util import plot_animation, plot_cost

from agents.gmds import GMDS
from agents.mds import MDS, MDSKDE
from agents.amds import AMDS
from agents.reps import REPS
from agents.areps import AREPS
from agents.util import Debug
from env import Arm, Point, Quadratic

import matplotlib.pyplot as plt
from chainerrl.experiments.prepare_output_dir import prepare_output_dir


def run(args):
    if args.seed is not None:
        np.random.seed(args.seed)

    # Set default parameter depend on environment
    if args.n_dims is None:
        if args.env == 'Point' or args.env == 'Arm':
            args.n_dims = 2
        elif args.env == 'OneDof' or args.env == 'Quadratic':
            args.n_dims = 1
        else:
            raise NotImplementedError

    if args.n_bfs is None:
        if args.env == 'Point' or args.env == 'Arm' or args.env == 'OneDof':
            args.n_bfs = 10
        elif args.env == 'Quadratic':
            args.n_bfs = 1
        else:
            raise NotImplementedError

    if args.dt is None:
        if args.env == 'Point' or args.env == 'Arm' or args.env == 'OneDof':
            args.dt = 0.01
        elif args.env == 'Quadratic':
            args.dt = 1.0
        else:
            raise NotImplementedError

    if args.duration is None:
        if args.env == 'Point' or args.env == 'Arm' or args.env == 'OneDof':
            args.duration = 0.5
        elif args.env == 'Quadratic':
            args.duration = 2.0
        else:
            raise NotImplementedError

    # Set default parameter depend on agent
    if args.r_gain is None:
        if args.r_normalize:
            if args.agent == 'PI2':
                args.r_gain = 10
            elif args.agent == 'MDS':
                args.r_gain = 10  # 1.0/100000.0
            elif args.agent == 'GMDS':
                args.r_gain = 10
            elif args.agent == 'AMDS':
                args.r_gain = 10  # 1.0/100000.0
            elif args.agent == 'MDSKDE':
                args.r_gain = 10  # 1.0/100000.0
            elif args.agent == 'REPS':
                args.r_gain = 10
            elif args.agent == 'AREPS':
                args.r_gain = 10
            else:
                raise NotImplementedError
        else:
            if args.agent == 'PI2':
                args.r_gain = 10**8
            elif args.agent == 'MDS':
                args.r_gain = 10**8
            elif args.agent == 'GMDS':
                args.r_gain = 10**8
            elif args.agent == 'AMDS':
                args.r_gain = 10**8
            elif args.agent == 'MDSKDE':
                args.r_gain = 10**8
            elif args.agent == 'REPS':
                args.r_gain = 10**8
            elif args.agent == 'AREPS':
                args.r_gain = 10**8
            else:
                raise NotImplementedError
    if args.std is None:
        if args.agent == 'PI2':
            args.std = 10
        elif args.agent == 'MDS':
            args.std = 10
        elif args.agent == 'GMDS':
            args.std = 10
        elif args.agent == 'AMDS':
            args.std = 10
        elif args.agent == 'MDSKDE':
            args.std = 1
        elif args.agent == 'REPS':
            args.std = 10
        elif args.agent == 'AREPS':
            args.std = 10
        else:
            raise NotImplementedError

    if args.plotdir is not None:
        if not args.plot:
            args.plot = []
        for d in args.plotdir:
            directory = glob.glob('{}/*'.format(os.path.normpath(d)))
            args.plot.extend(directory)

    # Check parameter
    if args.env == 'Point':
        assert 2 == args.n_dims, \
            'Number Error: n_dims of Point task arrow only 2 dimension'
    elif args.env == 'OneDof':
        assert 1 == args.n_dims, \
            'Number Error: n_dims of OneDof task arrow only 1 dimension'
    if args.env == 'Quadratic':
        assert 1 == args.n_bfs, \
            'Number Error: args.n_bfs of Quadratic task arrow only 1 dimension'
        assert 2 == args.duration, \
            'Number Error: duration of Quadratic task arrow only 2 [s]'
        assert 1 == args.dt, \
            'Number Error: dt of Quadratic task arrow only 1 [s]'

    args.outdir = prepare_output_dir(args, args.outdir, argv=sys.argv)
    Debug.set_outdir(args.outdir)
    Debug.set_n_updates(args.n_updates)

    # not use DMP
    if args.env == 'Quadratic':
        dmps = None
    # use DMP
    else:
        dmps = [[] for _ in range(args.n_dims)]

    n_times = int(args.duration / args.dt)

    # set environment
    if args.env == 'Arm' or args.env == 'Point':
        env = eval(args.env)(dmps, args.n_dims, args.n_bfs, n_times,
                             args.duration, args.dt)
    # elif args.env == 'Quadratic':
    #     env = eval(args.env)(args.n_dims)
    else:
        # TODO
        raise NotImplementedError

    # set agent
    if args.agent == 'PI2':
        agent = PI2(dmps, args.n_updates, args.n_reps, args.n_dims, args.n_bfs,
                    n_times, args.r_gain, args.r_normalize)
    elif args.agent == 'GMDS':
        agent = GMDS(dmps, args.n_updates, args.n_reps, args.n_dims,
                     args.n_bfs, n_times, args.r_gain,
                     args.r_normalize)
    elif args.agent == 'MDS':
        agent = MDS(args.n_updates, args.n_reps, args.n_dims, args.n_bfs,
                    n_times, args.r_gain, args.r_normalize)
    elif args.agent == 'AMDS':
        agent = AMDS(dmps, args.n_updates, args.n_reps, args.n_dims,
                     args.n_bfs, n_times, args.r_gain,
                     args.r_normalize)
    elif args.agent == 'MDSKDE':
        agent = MDSKDE(args.n_updates, args.n_reps, args.n_dims, args.n_bfs,
                       n_times, args.r_gain, args.r_normalize)
    elif args.agent == 'REPS':
        agent = REPS(dmps, args.n_updates, args.n_reps, args.n_dims,
                     args.n_bfs, n_times, args.r_gain,
                     args.r_normalize)
    elif args.agent == 'AREPS':
        agent = AREPS(dmps, args.n_updates, args.n_reps, args.n_dims,
                      args.n_bfs, n_times, args.r_gain,
                      args.r_normalize)
    else:
        raise NotImplementedError

    # Set dmp
    if args.env == 'Point' or args.env == 'Arm':
        for i in range(args.n_dims):
            dmps[i] = DMP(args.n_bfs, args.duration, 0.5, args.dt, True)
            dmps[i].reset()
            dmps[i].set_goal(env.gxi[i])
            agent.theta[i] = dmps[i].minimize_jerk()

    if args.plot is None:
        learn(args, agent, env, n_times)
    else:
        plot(args)


def learn(args, agent, env, n_times):
    reward_list = []
    # Update
    for step in range(args.n_updates):
        # Simulated annealing
        if args.annealing:
            agent.std_eps = args.std * \
                np.max((0.1, (args.n_updates - step) / args.n_updates))
        else:
            agent.std_eps = args.std
        # Roll-outs to update parameter
        for k in range(args.n_reps):
            accumulated_reward = 0
            reward = 0
            obs = env.reset()
            # Real-time
            for t in range(n_times):
                action = agent.act_and_train(obs, reward, t, k)
                obs, reward = env.step(action, t)
                accumulated_reward += reward

        # A noise-less roll-out to evaluate parameter
        accumulated_reward = 0
        obs = env.reset()
        # Time
        for t in range(n_times):
            action = agent.act(obs, t)
            obs, reward = env.step(action, t)
            accumulated_reward += reward
            ee_x, ee_y = env.set_plot_data(t - 1)
            if args.render:
                env.render(ee_x, ee_y, t - 1)

        print("step " + str(step + 1) + "\t" + str(accumulated_reward))
        reward_list.append(accumulated_reward)
        agent.save(args.outdir)
    ee_xs = []
    ee_ys = []
    for t in range(len(env.xi)):
        ee_x, ee_y = env.set_plot_data(t)
        ee_xs.append(ee_x)
        ee_ys.append(ee_y)

    rollouts = [args.n_reps * (n + 1) for n in range(len(reward_list))]
    np.savetxt(
        '{}/cost.csv'.format(args.outdir),
        reward_list,
        fmt="%.0f",
        delimiter=',')
    np.savetxt(
        '{}/rollout.csv'.format(args.outdir),
        rollouts,
        fmt="%.0f",
        delimiter=',')
    np.savetxt(
        '{}/x.csv'.format(args.outdir), ee_xs, fmt="%.5f", delimiter=',')
    np.savetxt(
        '{}/y.csv'.format(args.outdir), ee_ys, fmt="%.5f", delimiter=',')
    if not args.no_fig:
        plt.figure()
        plot_cost(reward_list, rollouts)
        plt.savefig('{}/cost.pdf'.format(args.outdir))
        plt.savefig('{}/cost.png'.format(args.outdir))

        if args.env != 'Quadratic':
            plot_animation(args.outdir, ee_xs, ee_ys, env)


def plot(args):
    plt.figure()
    for f in args.plot:
        f = os.path.normpath(f)
        args_info = json.load(open('{}/args.txt'.format(f), 'r'))
        agent = args_info['agent']
        if agent == 'PI2':
            plot_color = "orange"
        elif agent == 'GMDS':
            plot_color = "green"
        elif agent == 'MDS':
            plot_color = "darkblue"
        elif agent == 'AMDS':
            plot_color = "crimson"
        elif agent == 'MDSKDE':
            plot_color = "purple"
        elif agent == 'REPS':
            plot_color = "blue"
        elif agent == 'AREPS':
            plot_color = "pink"
        else:
            raise NotImplementedError
        reward_list = np.loadtxt('{}/cost.csv'.format(f), delimiter=',')
        rollouts = np.loadtxt('{}/rollout.csv'.format(f), delimiter=',')
        ee_xs = np.loadtxt('{}/x.csv'.format(f), delimiter=',')
        ee_ys = np.loadtxt('{}/y.csv'.format(f), delimiter=',')
        plot_cost(reward_list, rollouts, plot_color, '{}'.format(agent))
    plt.savefig('{}/cost.pdf'.format(args.outdir))
    plt.savefig('{}/cost.png'.format(args.outdir))


class TestPI2(unittest.TestCase):
    def test_update(self):
        n_dims = 2
        dmps = [[] for _ in range(n_dims)]
        agent = PI2(
            dmps=dmps,
            n_updates=10,
            n_reps=3,
            n_dims=n_dims,
            n_bfs=4,
            n=5,
            times=1.0,
            r_gain=10**8,
            r_normalize=False)

        r = np.array([[0, 10, 20, 10, 20], [50, 10, 10, 10, 10],
                      [10, 40, 30, 10, 10]])
        action = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9],
                           [10, 11, 12, 13, 14]])
        agent.update(r, action)


class TestGMDS(unittest.TestCase):
    def test_update(self):
        n_dims = 2
        dmps = [[] for _ in range(n_dims)]
        agent = GMDS(
            dmps=dmps,
            n_updates=10,
            n_reps=3,
            n_dims=n_dims,
            n_bfs=4,
            n=5,
            times=1.0,
            r_gain=10**8,
            r_normalize=False)

        r = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]])
        action = [[[0, 1, 2, 3], [4, 5, 6, 7]], [[0, 1, 2, 3], [4, 5, 6, 7]],
                  [[0, 1, 2, 3], [4, 5, 6, 7]]]
        agent.update(r, action)


class TestMDS(unittest.TestCase):
    def test_update(self):
        n_dims = 2
        n_bfs = 4
        n_reps = 3
        dmps = [[] for _ in range(n_dims)]
        agent = MDS(
            dmps=dmps,
            n_updates=10,
            n_reps=n_reps,
            n_dims=n_dims,
            n_bfs=n_bfs,
            n=5,
            times=1.0,
            r_gain=10**(-8),
            r_normalize=False)

        # theta = np.zeros((n_reps, n_dims, n_bfs))
        # epsilon = np.random.randn(n_reps, n_dims, n_bfs)
        # action = theta+epsilon
        # r = np.array([[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14]])
        r = np.array([[10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]])
        action = np.array([[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11],
                                                          [12, 13, 14, 15]],
                           [[16, 17, 18, 19], [20, 21, 22, 23]]])
        agent.update(r, action)


class TestMDSKDE(unittest.TestCase):
    def test_update(self):
        n_dims = 2
        n_bfs = 4
        n_reps = 3
        dmps = [[] for _ in range(n_dims)]
        agent = MDSKDE(
            dmps=dmps,
            n_updates=10,
            n_reps=n_reps,
            n_dims=n_dims,
            n_bfs=n_bfs,
            n=5,
            times=1.0,
            r_gain=10**(-8),
            r_normalize=False)
        # theta = np.zeros((n_reps, n_dims, n_bfs))
        # epsilon = np.random.randn(n_reps, n_dims, n_bfs)
        # action = theta + epsilon
        # r = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]])
        r = np.array([[10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]])
        action = np.array([[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11],
                                                          [12, 13, 14, 15]],
                           [[16, 17, 18, 19], [20, 21, 22, 23]]])
        agent.update(r, action)

    def test_act_and_train(self):
        n_updates = 10
        n_dims = 2
        n_bfs = 4
        n_reps = 3
        duration = 0.5
        dt = 0.01
        dmps = [[] for _ in range(n_dims)]

        n_times = int(duration / dt)

        env = Point(dmps, n_dims, n_bfs, n_times, duration, dt)

        agent = MDSKDE(
            dmps=dmps,
            n_updates=n_updates,
            n_reps=n_reps,
            n_dims=n_dims,
            n_bfs=n_bfs,
            n=5,
            times=1.0,
            r_gain=10**(-8),
            r_normalize=False)

        for i in range(n_dims):
            dmps[i] = DMP(n_bfs, duration, 0.5, dt, True)
            dmps[i].reset()
            dmps[i].set_goal(env.gxi[i])
            agent.theta[i] = dmps[i].minimize_jerk()

        reward = 0
        obs = env.reset()
        t = 0
        k = 0
        agent.act_and_train(obs, reward, t, k)


if __name__ == "__main__":
    unittest.main()
