import numpy as np


class CubicSpline():
    def __init__(self, x, y, sort=False):
        self.x = x
        self.y = y
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.calc_polynomials()
    
    def calc_polynomials(self):
        """
        points:       (xi, yi), i=1...n+1
        cubic spline: (xi, yi) : Si(x) = ai + bi(x-xi)^1 + ci(x-xi)^2 + di(x-xi)^3, i=1...n
        deriv #1:     (xi, yi) : Si'(x) = bi + 2ci(x-xi) + 3di(x-xi)^2
        deriv #2:     (xi, yi) : Si''(x) = 2ci + 6di(x-xi)
        conditions:
                    1) Si(xi) = yi           :n
                    2) Si(xi+1) = y+1        :n
                    3) Si'(x) = Si+1'(x)     :n-1
                    4) Si''(x) = Si+1''(x)   :n-1
        boundary:
                    1) S0''(x1) = 0          :1
                    2) Sn''(xi) = 0          :1
        ---------------------------------------------               
        Total:                               :4n
        """
        x_diff = np.diff(self.x)
        h = np.diff(self.y) / x_diff
        h_diff = np.diff(h)
        x_diff_sum = x_diff[:-1] + x_diff[1:]

        tri = (3 * h_diff) / x_diff_sum
        tri = np.insert(tri, 0, 0.0)
        tri = np.concatenate((tri, np.array([0.0])))

        di = tri[1:] - tri[:-1] / 6 * x_diff
        ci = tri[0:-1] / 2
        bi = h - x_diff * (2 * tri[:-1] + tri[1:]) / 6
        ai = self.y[0:-1]

        self.A, self.B, self.C, self.D = ai, bi, ci, di

    def interpolate(self, n):
        i = np.argmax(self.x >= n)
        h = n - self.x[i]
        y = self.A[i] + self.B[i]*h + self.C[i]*h**2 + self.D[i]*h**3
        return y

    def interpolate_all(self, span=0.01):
        x = None
        y = None
        for i in range(len(self.x) - 1):
            r = self.x[i], self.x[i+1]
            x_r = np.arange(r[0], r[1], span)
            h = x_r-r[0]
            y_r = self.A[i] + self.B[i]*h + self.C[i]*h**2 + self.D[i]*h**3
            if x is None:
                x = np.copy(x_r)
                y = np.copy(y_r)
            else:
                x = np.append(x, x_r)
                y = np.append(y, y_r)
        return x, y