import argparse
import matplotlib
import os

if os.getenv('DISPLAY'):
    # matplotlib.use('TkAgg')
    matplotlib.use('qt5agg')
else:
    matplotlib.use('PS')

from learn import run


def main():
    parser = argparse.ArgumentParser()

    # Parameter setting
    parser.add_argument('--outdir', type=str, default='mds_out',
                        help='output directory')
    parser.add_argument('--agent', type=str, default='PI2',
                        help='PI2, REPS, GMDS, MDS or AMDS')
    parser.add_argument('--r-normalize', action='store_true',
                        help='normalize or not accumulated reward')
    parser.add_argument('--r-gain', type=float,
                        help='the gain of accumulated reward')
    parser.add_argument('--env', type=str,
                        default='Arm', help='Arm or Point')
    parser.add_argument('--n-updates', type=int, default=50,
                        help='the number of updates')
    parser.add_argument('--annealing', action='store_true',
                        help='disable simulated annealing')
    parser.add_argument('--std', type=float, default=10.0,
                        help='the standard deviation of exploration noise')
    parser.add_argument('--n-dims', type=int, help='the number of dimensions')
    parser.add_argument('--n-bfs', type=int,
                        help='the number of basis functions')
    parser.add_argument('--n-reps', type=int, default=10,
                        help='the number of rollouts')
    parser.add_argument('--duration', type=float,
                        help='execution time [s], if task has dynamics')
    parser.add_argument('--dt', type=float,
                        help='actual time [s] per 1 time step')
    parser.add_argument('--render', action='store_true',
                        help='realtime rendering or not')
    parser.add_argument('--plot', action='append',
                        help='set data directories to plot')
    parser.add_argument('--plotdir', action='append',
                        help='set data PARENTS directory to plot')
    parser.add_argument('--no-fig', action='store_true')
    parser.add_argument('--seed', type=int, help='fix random seed for debug')

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
