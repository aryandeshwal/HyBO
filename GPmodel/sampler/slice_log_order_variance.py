import numpy as np
import torch

from GPmodel.inference.inference import Inference
from GPmodel.sampler.priors import log_prior_edgeweight
from GPmodel.sampler.tool_partition import group_input
from GPmodel.sampler.tool_slice_sampling import univariate_slice_sampling


def slice_log_order_variance(
    model,
    input_data,
    output_data,
    n_vertices,
    log_order_variance,
    sorted_partition,
    fourier_freq_list,
    fourier_basis_list,
    ind,
):
    """
    Slice sampling the edgeweight(exp('log_beta')) at 'ind' in 'log_beta' vector
    Note that model.kernel members (fourier_freq_list, fourier_basis_list) are updated.
    :param model:
    :param input_data:
    :param output_data:
    :param n_vertices: 1d np.array
    :param log_beta:
    :param sorted_partition: Partition of {0, ..., K-1}, list of subsets(list)
    :param fourier_freq_list:
    :param fourier_basis_list:
    :param ind:
    :return:
    """
    grouped_input_data = group_input(
        input_data=input_data[:, : model.kernel.num_discrete], sorted_partition=sorted_partition, n_vertices=n_vertices
    )  # need to understand this?
    grouped_input_data = torch.cat((grouped_input_data, input_data[:, model.kernel.num_discrete :]), dim=1)
    inference = Inference(train_data=(grouped_input_data, output_data), model=model)

    def logp(log_order_variance_i):
        """
        Note that model.kernel members (fourier_freq_list, fourier_basis_list) are updated.
        :param log_beta_i: numeric(float)
        :return: numeric(float)
        """
        log_prior = log_prior_edgeweight(log_order_variance_i)
        if np.isinf(log_prior):
            return log_prior
        model.kernel.log_order_variances[ind] = log_order_variance_i
        log_likelihood = float(-inference.negative_log_likelihood(hyper=model.param_to_vec()))
        return log_prior + log_likelihood

    x0 = float(log_order_variance[ind])
    x1 = univariate_slice_sampling(logp, x0)
    log_order_variance[ind] = x1
    model.kernel.log_order_variances[ind] = x1
    return log_order_variance


if __name__ == "__main__":
    pass
