import math
import numpy as np
import matplotlib.pyplot as plt


def test_func(x):
    # x is vector; here of length 1
    x = x[0]
    return math.cos(x) * x**2 + x


def run_smac(max_fun=30):
    from smac.facade.func_facade import fmin_smac

    x, cost, smac = fmin_smac(func=test_func,
                              x0=[-0],  # default values
                              bounds=[(-5, 5)],  # bounds of each x
                              maxfun=max_fun,  # maximal number of function evaluations
                              rng=1234  # random seed
                              )

    runhistory = smac.get_runhistory()

    # extract x value and corresponding y value
    x_smac = []
    y_smac = []
    for entry in runhistory.data:  # iterate over data because it is an OrderedDict
        config_id = entry.config_id  # look up config id
        config = runhistory.ids_config[config_id]  # look up config
        y_ = runhistory.get_cost(config)  # get cost
        x_ = config["x1"]  # there is only one entry in our example
        x_smac.append(x_)
        y_smac.append(y_)
    x_smac = np.array(x_smac)
    y_smac = np.array(y_smac)

    return smac, x_smac, y_smac


def plot_state(smac, model, x_points, y_points, x_smac, y_smac, step=None):
    """
      plot function with all evaluated points,
      EI acquisition function
      Predictions with uncertainties
    """
    from smac.optimizer.acquisition import EI

    # cost all points for x
    step = step or len(x_smac)
    x_smac_ = np.array([[x] for x in x_smac[:step]])
    y_smac_ = np.array([[y] for y in y_smac[:step]])
    # as an alternative, we could extract the points from the runhistory again
    # but these points will be scaled to a unit-hypercube
    # X, Y = smac.solver.rh2EPM.transform(runhistory)

    model.train(x_smac_, y_smac_)

    acq_func = EI(model=model)
    acq_func.update(model=model, eta=np.min(y_smac))

    x_points_ = np.array([[x] for x in x_points])
    acq_values = acq_func._compute(X=x_points_)[:, 0]

    # plot acquisition function
    y_mean, y_var = model.predict(x_points_)
    y_mean = y_mean[:, 0]
    y_std = np.sqrt(y_var)[:, 0]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(x_points, acq_values)
    plt.title("Aquisition Function")

    plt.savefig('fig%da.pdf' % step)

    # plot uncertainties
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(x_points, y_mean)
    ax1.fill_between(x_points, y_mean - y_std,
                     y_mean + y_std, alpha=0.5)
    ax1.plot(x_smac[:step], y_smac[:step], 'bo')
    ax1.plot(x_smac[:step], y_smac[:step], 'ro')
    ax1.plot(x_points, y_points, '--')
    plt.title("Uncertainty Predictions")

    plt.savefig('fig%db.pdf' % step)


if __name__ == '__main__':
    from smac.epm.rf_with_instances import RandomForestWithInstances

    x_points = np.linspace(start=-5, stop=5, num=100)
    y_points = list(map(test_func, map(lambda x: [x], x_points)))

    smac, x_smac, y_smac = run_smac()

    types, bounds = np.array([0]), np.array([[0.0, 1.0]])
    model = RandomForestWithInstances(types=types,
                                      bounds=bounds,
                                      instance_features=None,
                                      seed=12345,
                                      pca_components=12345,
                                      ratio_features=1,
                                      num_trees=1000,
                                      min_samples_split=1,
                                      min_samples_leaf=1,
                                      max_depth=100000,
                                      do_bootstrapping=False,
                                      n_points_per_tree=-1,
                                      eps_purity=0
                                      )

    plot_state(smac, model, x_points, y_points, x_smac, y_smac, 5)

    import os
    import shutil
    for f in os.listdir('.'):
        if f.startswith('smac3-output_'):
            shutil.rmtree(f)
