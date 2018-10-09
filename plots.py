from matplotlib import pyplot as plt
import numpy as np
import ad


def plot_function(y, x, start_val, end_val, description):
    plot_x = np.linspace(start_val, end_val, 1001)
    plot_y = [y.eval({x: v}) for v in plot_x]
    plot_yd = [y.d({x: v}) for v in plot_x]
    plt.plot(plot_x, plot_y)
    plt.plot(plot_x, plot_yd)
    plt.title(description)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")
    plt.legend([r"$f(x)$", r"$f'(x)$"])
    plt.show()


if __name__ == '__main__':
    x1 = ad.Variable('x1')
    y = x1.sin()
    plot_function(y, x1, 0, 10, r"$f(x) = \sin(x)$")

    x2 = ad.Variable('x2')
    y = (5 / x2).exp() - 5
    plot_function(y, x2, 1, 3, r"$f(x) = \exp(5 / x_2) - 5$")

    x3 = ad.Variable('x3')
    y = x3.sin().exp()
    plot_function(y, x3, 0, 10, r"$f(x) = \exp(\sin(x))$")
