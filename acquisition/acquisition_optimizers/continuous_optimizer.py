import time
from functools import partial

import cma
import numpy as np
import scipy.optimize as spo
import torch

from acquisition.acquisition_functions import expected_improvement
from acquisition.acquisition_marginalization import acquisition_expectation


def continuous_acquisition_expectation(
    x_continuous,
    discrete_part,
    inference_samples,
    partition_samples,
    n_vertices,
    acquisition_func,
    reference,
    batch=False,
):
    if batch:
        eval_x = torch.from_numpy(
            np.concatenate((np.tile(discrete_part, (len(x_continuous), 1)), x_continuous), axis=1)
        ).float()
        results = acquisition_expectation(
            eval_x, inference_samples, partition_samples, n_vertices, acquisition_func, reference
        )
        return np.array(results)
    else:
        eval_x = torch.from_numpy(np.concatenate((discrete_part, x_continuous))).float()
        print(
            acquisition_expectation(
                eval_x, inference_samples, partition_samples, n_vertices, acquisition_func, reference
            )[0].numpy()
        )
        return acquisition_expectation(
            eval_x, inference_samples, partition_samples, n_vertices, acquisition_func, reference
        )[0].numpy()


def cma_es_optimizer(
    objective,
    x_init,
    max_acquisition,
    inference_samples,
    partition_samples,
    n_vertices,
    acquisition_func=expected_improvement,
    reference=None,
):
    cont_bounds = [
        objective.problem.lower_bounds[objective.num_discrete :],
        objective.problem.upper_bounds[objective.num_discrete :],
    ]
    start_time = time.time()
    es = cma.CMAEvolutionStrategy(
        x0=x_init[objective.num_discrete :],
        sigma0=0.1,
        inopts={"bounds": cont_bounds, "popsize": 50},
    )
    iter = 1
    total_time_in_acq = 0
    while not es.stop():
        iter += 1
        xs = es.ask()
        X = torch.tensor(xs).float()
        # evaluate the acquisition function (optimizer assumes we're minimizing)
        temp_time = time.time()
        Y = -1 * continuous_acquisition_expectation(
            xs,
            x_init[: objective.num_discrete].numpy(),
            inference_samples,
            partition_samples,
            n_vertices,
            acquisition_func,
            reference,
            batch=True,
        )
        total_time_in_acq += time.time() - temp_time
        es.tell(xs, Y.ravel())  # return the result to the optimizer
        if iter > 10:
            break
    best_x = torch.from_numpy(es.best.x).float()
    if -1 * es.best.f > max_acquisition:
        return torch.cat((x_init[: objective.num_discrete], best_x), dim=0), -1 * es.best.f
    else:
        return x_init, max_acquisition
