import math
import logging
import numpy as np


def bohachevsky(x1, x2):
    """
    Bohachevsky function

    >>> bohachevsky(0, 0)
    0.0
    """
    term1 = x1 ** 2
    term2 = 2 * x2 ** 2
    term3 = -0.3 * math.cos(3 * math.pi * x1)
    term4 = -0.4 * math.cos(4 * math.pi * x2)

    return term1 + term2 + term3 + term4 + 0.7


def branin(x1, x2, a=1, b=5.1/(4*math.pi**2), c=5/math.pi, r=6, s=10, t=1/(8*math.pi)):
    """
    Branin function

    >>> 55 < branin(0, 0) < 56
    True
    """
    term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
    term2 = s * (1 - t) * math.cos(x1)

    return term1 + term2 + s


def camel(x1, x2):
    """
    Camel function

    >>> camel(0, 0)
    0.0
    """
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * x2 ** 2

    return term1 + term2 + term3


def forester(x):
    """
    Forester function

    >>> forester(1/3)
    0.0
    """
    fact1 = (6 * x - 2) ** 2
    fact2 = math.sin(12 * x - 4)

    return fact1 * fact2


def goldstein_price(x1, x2):
    """
    Goldstein-Price function

    >>> goldstein_price(0.0, -1.0)
    3.0
    """
    fact1a = (x1 + x2 + 1) ** 2
    fact1b = 19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2
    fact1 = 1 + fact1a * fact1b

    fact2a = (2 * x1 - 3 * x2) ** 2
    fact2b = 18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2
    fact2 = 30 + fact2a * fact2b

    return fact1 * fact2


optimal_values = {
    'bohachevsky': 0.0,
    'branin': 0.397887,
    'camel': -1.0316,
    'forester': forester(0.757248757841856),
    'goldstein_price': 3.0,
}

default_values = {
    'bohachevsky': [100, 100],
    'branin': [-5, 0],
    'camel': [-3, -2],
    'forester': [1],
    'goldstein_price': [-2, 2],
}

bounds_values = {
    'bohachevsky': [[-100, 100], [-100, 100]],
    'branin': [[-5, 10], [0, 15]],
    'camel': [[-3, 3], [-2, 2]],
    'forester': [[0, 1]],
    'goldstein_price': [[-2, 2], [-2, 2]],
}


def log_gap(function_name):
    """
    given a function f from the ones defined above, returns a function calculating
    logarithmic gap between of f and its optimal value
    :param function_name: the name of the function from ones defined above
    :return: function, calculating logarithmic gap

    >>> bohachevsky(1, 0)
    1.6

    >>> goldstein_price_gap(0.0, -1.0)
    -inf

    """
    def return_function(*args):
        if isinstance(args[0], np.ndarray) and len(args) == 1:
            # all the arguments are given in the first argument
            args = args[0].tolist()
        return np.log(eval(function_name)(*args) - optimal_values[function_name])
    return return_function


bohachevsky_gap = log_gap('bohachevsky')
branin_gap = log_gap('branin')
camel_gap = log_gap('camel')
forester_gap = log_gap('forester')
goldstein_price_gap = log_gap('goldstein_price')
