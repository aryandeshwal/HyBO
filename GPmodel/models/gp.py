import torch

from GPmodel.modules.gp_modules import GPModule


class GP(GPModule):
    def __init__(self, **kwargs):
        super(GP, self).__init__()

    def init_param(self, output_data):
        raise NotImplementedError

    def n_params(self):
        cnt = 0
        print("heree")
        for param in self.parameters():
            cnt += param.numel()
        return cnt

    def param_to_vec(self):
        flat_param_list = []
        for m in self.children():
            # print("************* [children] ************")
            # print(m, m.param_to_vec())
            flat_param_list.append(m.param_to_vec())
        return torch.cat(flat_param_list)

    def vec_to_param(self, vec):
        # print("vec", vec)
        ind = 0
        for m in self.children():
            jump = m.n_params()
            # print("jump: ", jump)
            m.vec_to_param(vec[ind : ind + jump])
            ind += jump
