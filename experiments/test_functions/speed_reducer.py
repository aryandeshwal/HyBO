from __future__ import division, print_function

import numpy as np
import torch

from experiments.test_functions.experiment_configuration import sample_speed_reducer_points


class Problem(object):
    def __init__(self, dimension, lower_bounds, upper_bounds):
        self.dimension = dimension
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        pass

    def __call__(self, x_unorder):
        x = []
        x.append(2.6 + (x_unorder[1] + 1) / 2)
        x.append(0.7 + ((x_unorder[2] + 1) * 0.1) / 2)
        x.append(x_unorder[0] + 17)
        x.append(7.3 + (x_unorder[3] + 1) / 2)
        x.append(7.3 + (x_unorder[4] + 1) / 2)
        x.append(2.9 + (x_unorder[5] + 1) / 2)
        x.append(5 + ((x_unorder[6] + 1) * 0.5) / 2)
        func_value = (
            0.7854 * x[0] * (x[1] ** 2) * (3.333 * (x[2] ** 2) + 14.9334 * (x[2]) - 43.0934)
            - 1.508 * x[0] * (x[5] ** 2 + x[6] ** 2)
            + 7.4777 * (x[5] ** 3 + x[6] ** 3)
            + 0.7854 * (x[3] * (x[5] ** 2) + x[4] * (x[6] ** 2))
        )
        constraint_cost = 0
        if 27 / (x[0] * (x[1] ** 2) * x[2]) >= 1:
            constraint_cost += 1000
        if 397.5 / (x[0] * (x[1] ** 2) * (x[2] ** 2)) >= 1:
            constraint_cost += 1000
        if 1.93 * (x[3] ** 3) / (x[1] * x[2] * (x[5] ** 4)) >= 1:
            constraint_cost += 1000
        if 1.93 * (x[4] ** 3) / (x[1] * x[2] * (x[6] ** 4)) >= 1:
            constraint_cost += 1000
        if np.sqrt(((745 * x[3] / (x[1] * x[2])) ** 2) + 16.9 * 1e6) / (110 * (x[5] ** 3)) >= 1:
            constraint_cost += 1000
        if np.sqrt(((745 * x[4] / (x[1] * x[2])) ** 2) + 157.5 * 1e6) / (85 * (x[6] ** 3)) >= 1:
            constraint_cost += 1000
        if x[1] * x[2] / 40 >= 1:
            constraint_cost += 1000
        if 5 * x[1] / x[0] >= 1:
            constraint_cost += 1000
        if x[0] / (12 * x[1]) >= 1:
            constraint_cost += 1000
        if (1.5 * x[5] + 1.9) / x[3] >= 1:
            constraint_cost += 1000
        if (1.1 * x[6] + 1.9) / x[4] >= 1:
            constraint_cost += 1000
        return func_value  # /100


class SpeedReducer(object):
    def __init__(self, random_seed=None, problem_id=None):
        self.random_seed = random_seed
        self.problem_id = problem_id
        lower_bounds = []
        upper_bounds = []
        # discrete variables
        lower_bounds.append(0)
        upper_bounds.append(11)

        for i in range(6):
            lower_bounds.append(-1)
            upper_bounds.append(1)
        assert len(lower_bounds) == 7
        assert len(upper_bounds) == 7
        print(lower_bounds)
        print(upper_bounds)
        self.num_discrete = 1
        self.num_continuous = 6

        self.problem = Problem(dimension=7, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
        self.problem.dimension = 7
        print(f"num_discrete: {self.num_discrete}, num_continuous: {self.num_continuous}")
        self.n_vertices = [12]
        self.n_vertices = np.array(self.n_vertices)
        self.suggested_init = self.generate_random_points(n_points=10, random_seed=random_seed).float()
        self.adjacency_mat = []
        self.fourier_freq = []
        self.fourier_basis = []
        self.random_seed_info = str(random_seed).zfill(4)
        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.linalg.eigh(laplacian)
            self.fourier_freq.append(eigval)
            self.fourier_basis.append(eigvec)

    def evaluate(self, x_unorder):
        if x_unorder.dim() == 2:
            x_unorder = x_unorder.squeeze(0)
        x = x_unorder.numpy()
        evaluation = self.problem(x)
        print(evaluation)
        return torch.tensor(evaluation).float()

    def sample_points(self, n_points, random_seed=None):
        if random_seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(random_seed)
        init_points = []
        for _ in range(n_points):
            random_point = []
            random_point.append(torch.randint(0, 12, (1,)))
            for i in range(6):
                random_point.append(torch.FloatTensor(1).uniform_(-1, 1))
            init_points.append(random_point)
        return torch.tensor(init_points).float()

    def generate_random_points(self, n_points, random_seed=None):
        return self.sample_points(
            n_points, random_seed=self.random_seed if random_seed is None else random_seed
        ).float()


if __name__ == "__main__":
    mixobj = SpeedReducer(44, "analog_zhi")
    print(mixobj.suggested_init)
    print(mixobj.suggested_init[0].type())
    print(mixobj.evaluate(mixobj.suggested_init[0]))
    print(mixobj.evaluate(mixobj.suggested_init[5]))
