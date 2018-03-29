import matplotlib.pyplot as plt


class Painter:
    def __init__(self):
        self.viapoint = (0.5, 0.5)

    def draw(self, x, y, t):
        n_dims = len(x)-1
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.scatter(x[0:n_dims], y[0:n_dims], c='b')
        plt.plot(x, y, 'g')
        plt.text(-0.8, -0.8, str(t*10)+"ms", fontsize=17)
        plt.scatter([0.5], [0.5], c='r')
        # plt.draw()
        plt.pause(.0001)
        plt.clf()

        return self.viapoint
