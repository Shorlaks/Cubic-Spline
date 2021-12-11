from cubic_spline import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    with open(path, "r") as file:
        lines = file.readlines()
    x = np.fromstring(lines[0], sep=",")
    y = np.fromstring(lines[1], sep=",")
    return x, y


if __name__ == "__main__":
    # Example 1 (data from file)
    x, y = load_data("data.txt")
    cs = CubicSpline(x, y, sort=False)
    """
    Find an interpolation for each point along x axis.
    Specify the range for the interpolation between the points.
    example: (x1=1, xn=n, span=0.1) - interpolation for (1.0, 1.1, 1.2, ..., n.0) 
    """
    xx, yy = cs.interpolate_all(span=0.1)
    plt.plot(xx, yy, color='black')
    plt.show()

    """
    Find an interpolation for a single point.
    """
    import random
    n = random.uniform(x[0], x[-1])
    yy = cs.interpolate(n)
