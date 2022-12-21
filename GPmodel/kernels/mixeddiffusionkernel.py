import math

import numpy as np
import torch

from GPmodel.kernels.mixedkernel import MixedKernel


class MixedDiffusionKernel(MixedKernel):
    def __init__(
        self,
        log_order_variances,
        grouped_log_beta,
        fourier_freq_list,
        fourier_basis_list,
        lengthscales,
        num_discrete,
        num_continuous,
    ):
        super(MixedDiffusionKernel, self).__init__(
            log_order_variances,
            grouped_log_beta,
            fourier_freq_list,
            fourier_basis_list,
            lengthscales,
            num_discrete,
            num_continuous,
        )

    def forward(self, x1, x2=None, diagonal=False):
        """
        :param x1, x2: each row is a vector with vertex numbers starting from 0 for each
        :return:
        """
        if diagonal:
            assert x2 is None
        stabilizer = 0
        if x2 is None:
            x2 = x1
            if diagonal:
                stabilizer = 1e-6 * x1.new_ones(x1.size(0), 1, dtype=torch.float32)
            else:
                stabilizer = torch.diag(1e-6 * x1.new_ones(x1.size(0), dtype=torch.float32))

        base_kernels = []
        for i in range(len(self.fourier_freq_list)):
            beta = torch.exp(self.grouped_log_beta[i])
            fourier_freq = self.fourier_freq_list[i]
            fourier_basis = self.fourier_basis_list[i]
            cat_i = fourier_freq.size(0)
            discrete_kernel = ((1 - torch.exp(-beta * cat_i)) / (1 + (cat_i - 1) * torch.exp(-beta * cat_i))) ** (
                (x1[:, i].unsqueeze(1)[:, np.newaxis] != x2[:, i].unsqueeze(1)).sum(axis=-1)
            )
            if diagonal:
                base_kernels.append(torch.diagonal(discrete_kernel).unsqueeze(1))
            else:
                base_kernels.append(discrete_kernel)

        lengthscales = torch.exp(self.lengthscales) ** 2
        temp_x_1 = x1[:, self.num_discrete :] / lengthscales
        temp_x_2 = x2[:, self.num_discrete :] / lengthscales

        for i in range(self.num_continuous):
            normalized_dists = torch.cdist(temp_x_1[:, i].unsqueeze(1), temp_x_2[:, i].unsqueeze(1))
            gaussian_kernel = torch.exp(-0.5 * (normalized_dists) ** 2)
            if not diagonal:
                base_kernels.append(gaussian_kernel)
            else:
                base_kernels.append(torch.diagonal(gaussian_kernel).unsqueeze(1))
        base_kernels = torch.stack(base_kernels)
        if diagonal:
            base_kernels = base_kernels.squeeze(-1)

        num_dimensions = self.num_discrete + self.num_continuous
        if not diagonal:
            e_n = torch.empty([num_dimensions + 1, base_kernels.size(1), base_kernels.size(2)])
            e_n[0, :, :] = 1.0
            interaction_orders = torch.arange(1, num_dimensions + 1).reshape([-1, 1, 1, 1]).float()
            kernel_dim = -3
            shape = [1 for _ in range(3)]
        else:
            e_n = torch.empty([num_dimensions + 1, base_kernels.size(1)])
            e_n[0, :] = 1.0
            interaction_orders = torch.arange(1, num_dimensions + 1).reshape([-1, 1, 1]).float()
            kernel_dim = -2
            shape = [1 for _ in range(2)]

        s_k = base_kernels.unsqueeze(kernel_dim - 1).pow(interaction_orders).sum(dim=kernel_dim)
        m1 = torch.tensor([-1.0])
        shape[kernel_dim] = -1

        for deg in range(1, num_dimensions + 1):  # deg goes from 1 to R (it's 1-indexed!)
            ks = torch.arange(1, deg + 1, dtype=torch.float).reshape(*shape)  # use for pow
            kslong = torch.arange(1, deg + 1, dtype=torch.long)  # use for indexing
            # note that s_k is 0-indexed, so we must subtract 1 from kslong
            sum_ = (
                m1.pow(ks - 1) * e_n.index_select(kernel_dim, deg - kslong) * s_k.index_select(kernel_dim, kslong - 1)
            ).sum(dim=kernel_dim) / deg
            if kernel_dim == -3:
                e_n[deg, :, :] = sum_
            else:
                e_n[deg, :] = sum_

        order_variances = torch.exp(self.log_order_variances)
        if kernel_dim == -3:
            kernel_mat = (
                torch.exp(self.log_amp)
                * (
                    (order_variances.unsqueeze(-1).unsqueeze(-1) * e_n.narrow(kernel_dim, 1, num_dimensions)).sum(
                        dim=kernel_dim
                    )
                )
                + stabilizer
            )
            return torch.exp(self.log_amp) * (
                (order_variances.unsqueeze(-1).unsqueeze(-1) * e_n.narrow(kernel_dim, 1, num_dimensions)).sum(
                    dim=kernel_dim
                )
                + stabilizer
            )
        else:
            return torch.exp(self.log_amp) * (
                (order_variances.unsqueeze(-1) * e_n.narrow(kernel_dim, 1, num_dimensions)).sum(dim=kernel_dim)
                + stabilizer
            )

    # def grad(self, x1, x2=None):
    #    if x2 is None:
    #        x2 = x1
    #    diffs = (x1[:, self.num_discrete:] - x2[:, self.num_discrete:])/self.lengthscales
    #    return diffs.t()


if __name__ == "__main__":
    pass
