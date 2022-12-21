import os
import sys
import time

import numpy as np
import psutil
import torch
import torch.multiprocessing as mp

from acquisition.acquisition_functions import expected_improvement
from acquisition.acquisition_marginalization import prediction_statistic
from acquisition.acquisition_optimizers.continuous_optimizer import cma_es_optimizer
from acquisition.acquisition_optimizers.graph_utils import neighbors
from acquisition.acquisition_optimizers.greedy_ascent import greedy_ascent
from acquisition.acquisition_optimizers.starting_points import optim_inits

MAX_N_ASCENT = float("inf")
N_CPU = os.cpu_count()
N_AVAILABLE_CORE = min(10, N_CPU)
N_SA_RUN = 10


def next_evaluation(
    objective,
    x_opt,
    input_data,
    inference_samples,
    partition_samples,
    edge_mat_samples,
    n_vertices,
    acquisition_func=expected_improvement,
    reference=None,
    parallel=None,
):
    id_digit = np.ceil(np.log(np.prod(n_vertices)) / np.log(10))
    id_unit = torch.from_numpy(np.cumprod(np.concatenate([np.ones(1), n_vertices[:-1]])).astype(np.int64))
    fmt_str = "\t %5.2f (id:%" + str(id_digit) + "d) ==> %5.2f (id:%" + str(id_digit) + "d)"

    start_time = time.time()
    print(
        "(%s) Acquisition function optimization initial points selection began"
        % (time.strftime("%H:%M:%S", time.localtime(start_time)))
    )

    x_inits, acq_inits = optim_inits(
        objective,
        x_opt,
        inference_samples,
        partition_samples,
        edge_mat_samples,
        n_vertices,
        acquisition_func,
        reference,
    )
    n_inits = x_inits.size(0)
    assert n_inits % 2 == 0

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        "(%s) Acquisition function optimization initial points selection ended - %s"
        % (time.strftime("%H:%M:%S", time.localtime(end_time)), time.strftime("%H:%M:%S", time.localtime(elapsed_time)))
    )

    start_time = time.time()
    print(
        "(%s) Acquisition function optimization with %2d inits"
        % (time.strftime("%H:%M:%S", time.localtime(start_time)), x_inits.size(0))
    )

    ga_args_list = [
        (
            x_inits[i],
            inference_samples,
            partition_samples,
            edge_mat_samples,
            n_vertices,
            acquisition_func,
            MAX_N_ASCENT,
            reference,
        )
        for i in range(n_inits)
    ]
    ga_start_time = time.time()
    sys.stdout.write("    Greedy Ascent  began at %s " % time.strftime("%H:%M:%S", time.localtime(ga_start_time)))
    if parallel:
        with mp.Pool(processes=min(n_inits, N_CPU // 3)) as pool:
            ga_result = []
            process_started = [False] * n_inits
            process_running = [False] * n_inits
            process_index = 0
            while process_started.count(False) > 0:
                cpu_usage = psutil.cpu_percent(0.25)
                run_more = (100.0 - cpu_usage) * float(psutil.cpu_count()) > 100.0 * N_AVAILABLE_CORE
                if run_more:
                    ga_result.append(pool.apply_async(greedy_ascent, args=ga_args_list[process_index]))
                    process_started[process_index] = True
                    process_running[process_index] = True
                    process_index += 1
            while [not res.ready() for res in ga_result].count(True) > 0:
                time.sleep(1)
            ga_return_values = [res.get() for res in ga_result]
    else:
        ga_return_values = [greedy_ascent(*(ga_args_list[i])) for i in range(n_inits)]

    ga_args_list = [
        (
            objective,
            ga_return_values[i][0],
            ga_return_values[i][1],
            inference_samples,
            partition_samples,
            n_vertices,
            expected_improvement,
            reference,
        )
        for i in range(len(ga_return_values))
    ]

    ga_return_values = [cma_es_optimizer(*(ga_args_list[i])) for i in range(len(ga_args_list))]
    ga_opt_vrt, ga_opt_acq = zip(*ga_return_values)
    sys.stdout.write("and took %s\n" % time.strftime("%H:%M:%S", time.localtime(time.time() - ga_start_time)))

    opt_vrt = list(ga_opt_vrt[:])
    opt_acq = list(ga_opt_acq[:])

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        "(%s) Acquisition function optimization ended %s"
        % (time.strftime("%H:%M:%S", time.localtime(end_time)), time.strftime("%H:%M:%S", time.localtime(elapsed_time)))
    )

    # argsort sorts in ascending order so it is negated to have descending order
    acq_sort_inds = np.argsort(-np.array(opt_acq))
    suggestion = None
    for i in range(len(opt_vrt)):
        ind = acq_sort_inds[i]
        if not torch.all(opt_vrt[ind] == input_data, dim=1).any():
            suggestion = opt_vrt[ind]
            break
    if suggestion is None:
        print("random suggestion!!!")
        for i in range(len(opt_vrt)):
            ind = acq_sort_inds[i]
            nbds = neighbors(
                opt_vrt[ind][: inference_samples[0].model.kernel.num_discrete],
                partition_samples,
                edge_mat_samples,
                n_vertices,
                uniquely=True,
            )
            nbds = torch.cat(
                (
                    nbds,
                    opt_vrt[ind][inference_samples[0].model.kernel.num_discrete :].unsqueeze(0).repeat(nbds.size(0), 1),
                ),
                dim=1,
            )
            # nbds = neighbors(opt_vrt[ind], partition_samples, edge_mat_samples, n_vertices, uniquely=True)
            for j in range(nbds.size(0)):
                if not torch.all(nbds[j] == input_data, dim=1).any():
                    suggestion = nbds[j]
                    break
            if suggestion is not None:
                break
    if suggestion is None:
        suggestion = torch.cat(
            tuple([torch.randint(low=0, high=int(n_v), size=(1, 1)) for n_v in n_vertices]), dim=1
        ).long()
    if torch.all(suggestion == input_data, dim=1).any():
        suggestion = objective.generate_random_points(1)
    mean, std, var = prediction_statistic(suggestion, inference_samples, partition_samples, n_vertices)
    return suggestion, mean, std, var
