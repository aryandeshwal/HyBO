import numpy as np
import torch

from GPmodel.inference.inference import Inference
from GPmodel.sampler.priors import log_prior_edgeweight
from GPmodel.sampler.tool_partition import group_input
from GPmodel.sampler.tool_slice_sampling import univariate_slice_sampling

# no prior as such but best to keep the values within 0 and a maximum (100)


def slice_lengthscale(
    model,
    input_data,
    output_data,
    n_vertices,
    lengthscale_sample,
    sorted_partition,
    fourier_freq_list,
    fourier_basis_list,
    ind,
):
    """
    Slice sampling  at 'ind' in 'log_lengthscale' vector
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

    def logp(ls):
        """
        Note that model.kernel members (fourier_freq_list, fourier_basis_list) are updated.
        :param log_beta_i: numeric(float)
        :return: numeric(float)
        """
        if ls < 0 or ls > 2:
            return -np.inf  # float('-inf')
        inference.model.kernel.lengthscales[ind] = ls
        log_likelihood = float(-inference.negative_log_likelihood(hyper=model.param_to_vec()))
        return log_likelihood

    x0 = float(lengthscale_sample[ind])
    x1 = univariate_slice_sampling(logp, x0)
    lengthscale_sample[ind] = x1
    model.kernel.lengthscales[ind] = x1
    return lengthscale_sample


if __name__ == "__main__":
    pass
