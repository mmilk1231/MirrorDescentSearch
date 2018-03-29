import argparse
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os


def arm_plots(xlink, ylink):
    for x, y in zip(xlink, ylink):
        prev_xx = None
        prev_yy = None
        for xx, yy in zip(x, y):
            if prev_xx is not None:
                plt.plot((prev_xx, xx), (prev_yy, yy), c="grey", alpha=0.3)
            prev_xx = xx
            prev_yy = yy


def trajectory_plots(x, y, viapoint, xlabel, ylabel,
                     labels):
    n_plots = len(x)
    styles = ["solid", "dashed", "dashdot", "dotted"]
    plt.scatter(viapoint[0], viapoint[1], c="white",
                edgecolors="black", marker="*", s=100)
    for i in range(n_plots):
        color = cm.Set1(i)  # list(cm.viridis(i/n_plots))
        linestyle = styles[i % 4]
        plt.plot(x[i], y[i], label=labels[i], color=color, linestyle=linestyle)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([np.min(x), np.max(x)])


def mean_std_plots(x, y_mean, y_std, xlabel, ylabel,
                   labels):
    n_plots = len(x)
    styles = ["solid", "dashed", "dashdot", "dotted"]
    for i in range(n_plots):
        color = cm.Set1(i)  # list(cm.viridis(i/n_plots))
        linestyle = styles[i % 4]
        mean_std_plot(x[i], y_mean[i], y_std[i], labels[i],
                      color, linestyle)
    plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([np.min(x), np.max(x)])


def mean_std_plot(x, y_mean, y_std, label, color, linestyle="solid"):
    plt.plot(x, y_mean, label=label, color=color, linestyle=linestyle)
    plt.fill_between(x, y_mean+y_std, y_mean-y_std,
                     color=color, alpha=0.3, interpolate=True)


def loadcsv(path):
    return np.loadtxt(path, delimiter=",")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plotdir", action="append")
    parser.add_argument("--plot", action="append")
    parser.add_argument("--label", action="append")
    parser.add_argument("--env", type=str, default="Point")
    parser.add_argument("--name", type=str)
    parser.add_argument("--n-plots", type=int)
    parser.add_argument("--n-stats", type=int)
    parser.add_argument("--n-dims", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=50)
    args = parser.parse_args()

    if args.name is None:
        args.name = args.env
    if args.env == "Arm":
        args.n_dims += 1

    if args.plotdir is not None:
        if not args.plot:
            args.plot = []
        for d in args.plotdir:
            directory = glob.glob('{}/*'.format(os.path.normpath(d)))
            args.plot.extend(directory)

    rollout = []
    cost = []
    x = []
    y = []
    for pdir in args.plot:
        rollout.append(loadcsv("{}/rollout.csv".format(pdir)))
        cost.append(loadcsv("{}/cost.csv".format(pdir)))
        x.append(loadcsv("{}/x.csv".format(pdir)))
        y.append(loadcsv("{}/y.csv".format(pdir)))

    rollout = np.reshape(rollout, (args.n_plots, args.n_stats, -1))
    cost = np.reshape(cost, (args.n_plots, args.n_stats, -1))
    x = np.reshape(x, (args.n_plots, args.n_stats, -1, args.n_dims))
    y = np.reshape(y, (args.n_plots, args.n_stats, -1, args.n_dims))
    if args.env == "Point":
        xee = x[:, :, :, 0]
        yee = y[:, :, :, 0]
    elif args.env == "Arm":
        xee = x[:, :, :, -1]
        yee = y[:, :, :, -1]

    c_mean = np.mean(cost, 1)
    c_std = np.std(cost, 1)

    mean_std = []
    for i, _ in enumerate(c_std[:, -1]):
        # print(c_mean[i, -1], c_std[i, -1])
        mean = "{:.1e}".format(c_mean[i, -1]).replace("e+0", "\\times10^")
        std = "{:.1e}".format(c_std[i, -1]).replace("e+0", "\\times10^")
        mean_std.append([mean, "\pm", std])

    np.savetxt("{}_mean-std.txt".format(args.name),
               np.array(mean_std), fmt="%s")

    plt.rc('font', family='sans-serif')
    plt.rc('text', usetex=True)
    plt.rcParams["font.size"] = 15
    plt.tight_layout()
    plt.figure()
    mean_std_plots(rollout[:, 0], c_mean, c_std,
                   "The Number of Roll-outs", "Cost",
                   args.label)
    if args.env == "Point":
        plt.legend(loc='lower left')
    elif args.env == "Arm":
        plt.legend(loc='upper right')
    plt.savefig("{}_cost.pdf".format(args.name))

    xee_mean = np.mean(xee, 1)
    yee_mean = np.mean(yee, 1)
    x_mean = np.mean(x, 1)
    y_mean = np.mean(y, 1)

    plt.figure()
    if args.env == "Point":
        viapoint = (0.5, 0.2)
        trajectory_plots(xee_mean, yee_mean, viapoint,
                         "X-Axis", "Y-Axis",
                         args.label)
        plt.legend(loc='upper left')
        plt.xlim(xmin=0, xmax=1)
        plt.savefig("{}_env.pdf".format(args.name))
    elif args.env == "Arm":
        viapoint = (0.5, 0.5)
        arm_plots(x_mean[0], y_mean[0])
        trajectory_plots(xee_mean, yee_mean, viapoint,
                         "X-Axis", "Y-Axis",
                         args.label)
        plt.legend(loc='upper right')
        plt.xlim(xmin=0, xmax=1)
        plt.ylim(ymin=-0.1)
        plt.savefig("{}_env.pdf".format(args.name))


if __name__ == "__main__":
    main()
