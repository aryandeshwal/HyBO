from __future__ import division, print_function

import cocoex  # experimentation and post-processing modules
import cocopp

from experiments.test_functions.experiment_configuration import sample_mixed_init_points

# prepare mixed integer suite
suite_name = "bbob-mixint"
output_folder = "cocex-optimize-fmin"
suite = cocoex.Suite(suite_name, "", "")
# observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
# minimal_print = cocoex.utilities.MiniPrint()
import numpy as np
import torch


class MixedIntegerCOCO(object):
    """
    Mixed Integer Black box optimization using cocoex library
    """

    def __init__(self, random_seed=None, problem_id=None):
        self.random_seed = random_seed
        self.problem_id = problem_id
        self.problem = suite.get_problem(self.problem_id)
        self.num_discrete = self.problem.number_of_integer_variables
        self.num_continuous = self.problem.dimension - self.num_discrete
        print(f"num_discrete: {self.num_discrete}, num_continuous: {self.num_continuous}")
        self.n_vertices = []
        for i in range(self.num_discrete):
            self.n_vertices.append(int(self.problem.upper_bounds[i] - self.problem.lower_bounds[i] + 1))
        self.n_vertices = np.array(self.n_vertices)
        self.suggested_init = sample_mixed_init_points(
            self.problem.lower_bounds,
            self.problem.upper_bounds,
            self.num_discrete,
            n_points=20,
            random_seed=random_seed,
        ).float()
        self.suggested_init[-1] = torch.tensor(self.problem.initial_solution).float()
        self.adjacency_mat = []
        self.fourier_freq = []
        self.fourier_basis = []
        self.random_seed_info = str(random_seed).zfill(4)
        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            # print(n_v)
            adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.linalg.eigh(laplacian)
            self.fourier_freq.append(eigval)
            self.fourier_basis.append(eigvec)

    def evaluate(self, x):
        if x.dim() == 2:
            x = x.squeeze(0)
        evaluation = self.problem(x)
        return torch.tensor(evaluation).float()

    def generate_random_points(self, n_points, random_seed=None):
        return sample_mixed_init_points(
            self.problem.lower_bounds,
            self.problem.upper_bounds,
            self.num_discrete,
            n_points=n_points,
            random_seed=self.random_seed if random_seed is None else random_seed,
        ).float()


if __name__ == "__main__":
    mixobj = MixedIntegerCOCO(44, "bbob-mixint_f001_i01_d20")
    print(mixobj.evaluate(mixobj.suggested_init[0]))
    print(mixobj.problem.evaluations)
    print(mixobj.evaluate(mixobj.suggested_init[5]))
    print(mixobj.problem.evaluations)
