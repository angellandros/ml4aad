import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from smac.facade.func_facade import fmin_smac

from ass1.task1 import clean_smac_shit
from ass1.task2 import bohachevsky_gap, branin_gap, camel_gap, forester_gap, goldstein_price_gap
from ass1.task2 import bounds_values, default_values


def main(func_name):
    func = eval(func_name)
    func_name = func_name[:-4]
    smac_x, cost, smac = fmin_smac(func=func,
                              x0=default_values[func_name],  # default values
                              bounds=bounds_values[func_name],  # bounds of each x
                              maxfun=20,  # maximal number of function evaluations
                              rng=1234  # random seed
                              )

    runhistory = smac.get_runhistory()

    # extract x value and corresponding y value
    x_smac = []
    y_smac = []
    z_smac = []
    for entry in runhistory.data:  # iterate over data because it is an OrderedDict
        config_id = entry.config_id  # look up config id
        config = runhistory.ids_config[config_id]  # look up config
        x_ = config["x1"]
        y_ = config["x2"]
        z_ = runhistory.get_cost(config)  # get cost
        x_smac.append(x_)
        y_smac.append(y_)
        z_smac.append(z_)
    x_smac = np.array(x_smac)
    y_smac = np.array(y_smac)
    z_smac = np.array(z_smac)

    # 3d plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = np.linspace(*bounds_values[func_name][0], 100)
    y = np.linspace(*bounds_values[func_name][1], 100)
    x, y = np.meshgrid(x, y)
    zs = np.array([func(a, b) for a, b in zip(np.ravel(x), np.ravel(y))])
    z = zs.reshape(x.shape)

    ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0)
    ax.scatter(x_smac, y_smac, z_smac, color='red')
    plt.show()


if __name__ == '__main__':
    try:
        for func in ['bohachevsky_gap', 'branin_gap', 'camel_gap', 'goldstein_price_gap']:
            print(func)
            main(func)
            del func
    except Exception as e:
        logging.exception(e)
    finally:
        clean_smac_shit()
