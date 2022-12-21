import copy
import sys
import time

import numpy as np

from config import PROGRESS_BAR_LEN
from GPmodel.sampler.sample_edgeweight import slice_edgeweight
from GPmodel.sampler.sample_hyper import slice_hyper
from GPmodel.sampler.slice_lengthscale import slice_lengthscale
from GPmodel.sampler.slice_log_order_variance import slice_log_order_variance
from GPmodel.sampler.tool_partition import direct_product


def posterior_sampling(
    model,
    input_data,
    output_data,
    n_vertices,
    adj_mat_list,
    log_order_variance,
    log_beta,
    lengthscale,
    sorted_partition,
    n_sample,
    n_burn=0,
    n_thin=1,
):
    """

    :param model:
    :param input_data:
    :param output_data:
    :param n_vertices:
    :param adj_mat_list:
    :param log_beta:
    :param sorted_partition:
    :param n_sample:
    :param n_burn:
    :param n_thin:
    :return:
    """

    hyper_samples = []
    log_beta_samples = []
    log_order_variance_samples = []
    lengthscale_samples = []
    partition_samples = []
    freq_samples = []
    basis_samples = []
    edge_mat_samples = []

    partition_sample = sorted_partition
    log_beta_sample = log_beta
    log_order_variance_sample = log_order_variance
    lengthscale_sample = lengthscale
    fourier_freq_list = model.kernel.fourier_freq_list
    fourier_basis_list = model.kernel.fourier_basis_list
    edge_mat_list = [direct_product(adj_mat_list, subset) for subset in sorted_partition]  # kronecker operation

    n_sample_total = n_sample * n_thin + n_burn
    n_digit = int(np.ceil(np.log(n_sample_total) / np.log(10)))
    for s in range(0, n_sample_total):
        # print("*************[ slicing hyper parameter ]*************")
        slice_hyper(model, input_data, output_data, n_vertices, sorted_partition=partition_sample)
        # print("model.mean:", model.mean.const_mean)
        # print("model likelihood:", model.likelihood.log_noise_var)
        # print("model kernel:", model.kernel.log_amp)
        # In 'Batched High-dimensional Bayesian Optimization via Structural Kernel Learning',
        # similar additive structure is updated for every 50 iterations(evaluations)
        # This may be due to too much variability if decomposition is learned every iterations.
        # Thus, when multiple points are sampled, sweeping all inds for each sample may be not a good practice
        # For example if there are 50 variables and 10 samples are needed,
        # then after shuffling indices, and 50/10 thinning can be used.

        shuffled_beta_ind = list(range(len(adj_mat_list)))
        np.random.shuffle(shuffled_beta_ind)
        for beta_ind in shuffled_beta_ind:
            # In each sampler, model.kernel fourier_freq_list, fourier_basis_list are updated.
            log_beta_sample = slice_edgeweight(
                model,
                input_data,
                output_data,
                n_vertices,
                log_beta=log_beta_sample,
                sorted_partition=partition_sample,
                fourier_freq_list=fourier_freq_list,
                fourier_basis_list=fourier_basis_list,
                ind=beta_ind,
            )

        shuffled_ls_ind = list(range(model.kernel.num_continuous))
        np.random.shuffle(shuffled_ls_ind)
        for ls_ind in shuffled_ls_ind:
            lengthscale_sample = slice_lengthscale(
                model,
                input_data,
                output_data,
                n_vertices,
                lengthscale_sample=lengthscale_sample,
                sorted_partition=partition_sample,
                fourier_freq_list=fourier_freq_list,
                fourier_basis_list=fourier_basis_list,
                ind=ls_ind,
            )
        shuffled_lov_ind = list(range(model.kernel.num_continuous + model.kernel.num_discrete))
        np.random.shuffle(shuffled_lov_ind)
        for lov_ind in shuffled_lov_ind:
            log_order_variance_sample = slice_log_order_variance(
                model,
                input_data,
                output_data,
                n_vertices,
                log_order_variance=log_order_variance_sample,
                sorted_partition=partition_sample,
                fourier_freq_list=fourier_freq_list,
                fourier_basis_list=fourier_basis_list,
                ind=lov_ind,
            )
        if s >= n_burn and (s - n_burn + 1) % n_thin == 0:
            hyper_samples.append(model.param_to_vec())
            log_beta_samples.append(log_beta_sample.clone())
            lengthscale_samples.append(lengthscale_sample.clone())
            log_order_variance_samples.append(log_order_variance_sample.clone())
            partition_samples.append(copy.deepcopy(partition_sample))
            freq_samples.append([elm.clone() for elm in fourier_freq_list])
            basis_samples.append([elm.clone() for elm in fourier_basis_list])
            edge_mat_samples.append([elm.clone() for elm in edge_mat_list])
        progress_mark_len = int((s + 1.0) / n_sample_total * PROGRESS_BAR_LEN)
        fmt_str = (
            "(%s)   %3d%% (%"
            + str(n_digit)
            + "d of %d) |"
            + "#" * progress_mark_len
            + "-" * (PROGRESS_BAR_LEN - progress_mark_len)
            + "|"
        )
        progress_str = fmt_str % (
            time.strftime("%H:%M:%S", time.gmtime()),
            int((s + 1.0) / n_sample_total * 100),
            s + 1,
            n_sample_total,
        )
        sys.stdout.write(("\b" * len(progress_str)) + progress_str)
        sys.stdout.flush()
    return (
        hyper_samples,
        log_order_variance_samples,
        log_beta_samples,
        lengthscale_samples,
        partition_samples,
        freq_samples,
        basis_samples,
        edge_mat_samples,
    )
