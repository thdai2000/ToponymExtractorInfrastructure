import numpy as np
from scipy.special import comb

def get_bezier_ctrl_pts(points, degree=3):
    X = [p[0] for p in points]
    Y = [p[1] for p in points]
    return get_bezier_parameters(X, Y, degree)

def get_bezier_parameters(X, Y, degree=3):
    def bpoly(n, t, k):
        """ Bernstein polynomial when a = 0 and b = 1. """
        return t ** k * (1 - t) ** (n - k) * comb(n, k)

    def bmatrix(T):
        """ Bernstein matrix for Bézier curves. """
        return np.matrix([[bpoly(degree, t, k) for k in range(degree + 1)] for t in T])

    def least_square_fit(points, M):
        M_ = np.linalg.pinv(M)
        return M_ * points

    n_points = len(X)
    if n_points == 1:
        # Single point, all control points are the same
        raise Exception("Cannot create a Bézier curve with a single point!")
    elif n_points == 2:
        # Linear Bézier curve
        X = [X[0], X[0] + 0.25 * (X[1] - X[0]), X[0] + 0.75 * (X[1] - X[0]), X[1]]
        Y = [Y[0], Y[0] + 0.25 * (Y[1] - Y[0]), Y[0] + 0.75 * (Y[1] - Y[0]), Y[1]]

    elif n_points == 3:
        # Cubic Bézier curve with interpolation
        X = [X[0], X[0] + 0.5 * (X[1] - X[0]), X[1], X[1] + 0.5 * (X[2] - X[1]), X[2]]
        Y = [Y[0], Y[0] + 0.5 * (Y[1] - Y[0]), Y[1], Y[1] + 0.5 * (Y[2] - Y[1]), Y[2]]

    T = np.linspace(0, 1, len(X))
    M = bmatrix(T)
    points = np.array(list(zip(X, Y)))

    final = least_square_fit(points, M).tolist()
    final[0] = [X[0], Y[0]]
    final[len(final)-1] = [X[len(X)-1], Y[len(Y)-1]]
    return final