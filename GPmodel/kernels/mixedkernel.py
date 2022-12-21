import torch

from GPmodel.modules.gp_modules import GPModule


class MixedKernel(GPModule):
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
        super(MixedKernel, self).__init__()
        self.log_amp = torch.FloatTensor(1)
        self.log_order_variances = log_order_variances  # torch.ones(size=(num_discrete + num_continuous, )) # one for each combination of interaction
        self.grouped_log_beta = grouped_log_beta
        self.fourier_freq_list = fourier_freq_list
        self.fourier_basis_list = fourier_basis_list
        self.lengthscales = lengthscales
        self.num_discrete = num_discrete
        self.num_continuous = num_continuous
        assert (
            self.log_order_variances.size(0) == self.num_continuous + self.num_discrete
        ), "order variances are not properly initialized"
        assert self.lengthscales.size(0) == self.num_continuous, "lengthscales is not properly initialized"
        assert self.grouped_log_beta.size(0) == self.num_discrete, "beta is not properly initialized"

    def n_params(self):
        return 1

    def param_to_vec(self):
        return self.log_amp.clone()

    def vec_to_param(self, vec):
        assert vec.numel() == 1  # self.num_discrete + self.num_continuous
        self.log_amp = vec[:1].clone()

    def forward(self, input1, input2=None):
        raise NotImplementedError
