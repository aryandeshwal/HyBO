import numpy as np
import torch

from acquisition.acquisition_functions import expected_improvement
from acquisition.acquisition_marginalization import acquisition_expectation
from acquisition.acquisition_optimizers.graph_utils import neighbors

N_RANDOM_VERTICES = 200  # 20000
N_GREEDY_ASCENT_INIT = 20
N_SPRAY = 10


def optim_inits(
    objective,
    x_opt,
    inference_samples,
    partition_samples,
    edge_mat_samples,
    n_vertices,
    acquisition_func=expected_improvement,
    reference=None,
):
    """
    :param x_opt: 1D Tensor
    :param inference_samples:
    :param partition_samples:
    :param edge_mat_samples:
    :param n_vertices:
    :param acquisition_func:
    :param reference:
    :return:
    """
    # for x, y in zip(objective.problem.lower_bounds, objective.problem.upper_bounds):
    #     print(x, y)
    # print("n_vertices", n_vertices)
    # print(partition_samples)
    # print(edge_mat_samples)
    # print(len(edge_mat_samples))
    # print(edge_mat_samples[0])
    # for i in range(len(edge_mat_samples)):
    #    print(len(edge_mat_samples[i]))

    # rnd_nbd = torch.cat(tuple([torch.randint(low=0, high=int(n_v), size=(N_RANDOM_VERTICES, 1)) for n_v in n_vertices]), dim=1).long()
    rnd_nbd = objective.generate_random_points(N_RANDOM_VERTICES)
    min_nbd = neighbors(
        x_opt[: objective.num_discrete], partition_samples, edge_mat_samples, n_vertices, uniquely=False
    )
    # print(min_nbd.size(0))
    # print(min_nbd)
    # print(x_opt[objective.num_discrete:].unsqueeze(0).repeat(min_nbd.size(0), 1)[:10])
    min_nbd = torch.cat((min_nbd, x_opt[objective.num_discrete :].unsqueeze(0).repeat(min_nbd.size(0), 1)), dim=1)
    # print(min_nbd[:6])
    shuffled_ind = list(range(min_nbd.size(0)))
    np.random.shuffle(shuffled_ind)
    x_init_candidates = torch.cat(tuple([min_nbd[shuffled_ind[:N_SPRAY]], rnd_nbd]), dim=0)
    acquisition_values = acquisition_expectation(
        x_init_candidates, inference_samples, partition_samples, n_vertices, acquisition_func, reference
    )
    # print("acquisition_values")
    # print(acquisition_values[:30])

    nonnan_ind = ~torch.isnan(acquisition_values).squeeze(1)
    x_init_candidates = x_init_candidates[nonnan_ind]
    acquisition_values = acquisition_values[nonnan_ind]

    acquisition_sorted, acquisition_sort_ind = torch.sort(acquisition_values.squeeze(1), descending=True)
    x_init_candidates = x_init_candidates[acquisition_sort_ind]

    return x_init_candidates[:N_GREEDY_ASCENT_INIT], acquisition_sorted[:N_GREEDY_ASCENT_INIT]
