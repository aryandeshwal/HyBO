from __future__ import division, print_function

import numpy as np
import torch


class Problem(object):
    def __init__(self, dimension, lower_bounds, upper_bounds):
        self.dimension = dimension
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        pass

    def __call__(self, x):
        x[0] += 1
        x[1] += 1
        x[2] = 10 + ((x[2] + 1) * (200 - 10)) / 2
        x[3] = 10 + ((x[3] + 1) * (240 - 10)) / 2
        func_value = (
            0.6224 * x[0] * x[2] * x[3]
            + 1.7781 * x[1] * (x[2] ** 2)
            + 3.1661 * (x[0] ** 2) * x[3]
            + 19.84 * (x[0] ** 2) * x[2]
        )
        constraint_value = 0
        constraint_1 = x[0] - 0.0193 * x[2]
        constraint_2 = x[1] - 0.00954 * x[2]
        constraint_3 = np.pi * (x[2] ** 2) * x[3] + (4 / 3) * np.pi * (x[2] ** 3) - 1296000
        constraint_cost = (constraint_1 < 0) * 100 + (constraint_2 < 0) * 100 + (constraint_3 < 0) * 100
        print(f"func_value : {func_value/1e6}")
        return func_value / 1e6  # + constraint_cost


class Pressure_Vessel_Design(object):
    def __init__(self, random_seed=None, problem_id=None):
        self.random_seed = random_seed
        self.problem_id = problem_id
        lower_bounds = []
        upper_bounds = []
        # discrete variables
        lower_bounds.append(0)  #
        upper_bounds.append(99)
        lower_bounds.append(0)  #
        upper_bounds.append(99)

        for i in range(2):
            lower_bounds.append(-1)  # weld_thickness (h)
            upper_bounds.append(1)

        assert len(lower_bounds) == 4
        assert len(upper_bounds) == 4
        print(lower_bounds)
        print(upper_bounds)
        self.problem = Problem(dimension=4, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
        self.problem.dimension = 4
        self.num_discrete = 2
        self.num_continuous = 2
        print(f"num_discrete: {self.num_discrete}, num_continuous: {self.num_continuous}")
        self.n_vertices = [100, 100]
        self.n_vertices = np.array(self.n_vertices)
        self.suggested_init = self.generate_random_points(n_points=10, random_seed=random_seed)
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

    def evaluate(self, x_unorder):
        if x_unorder.dim() == 2:
            x_unorder = x_unorder.squeeze(0)
        x = x_unorder.numpy().copy()
        print(f"evaluating {x}....")
        evaluation = self.problem(x)
        return torch.tensor(evaluation).float()

    def sample_points(self, n_points, random_seed=None):
        if random_seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(random_seed)
        init_points = []
        for _ in range(n_points):
            random_point = []
            random_point.append(torch.randint(0, 100, (1,)))
            random_point.append(torch.randint(0, 100, (1,)))

            for i in range(2):
                random_point.append(torch.FloatTensor(1).uniform_(-1, 1))
            init_points.append(random_point)
        return torch.tensor(init_points).float()

    def generate_random_points(self, n_points, random_seed=None):
        return self.sample_points(
            n_points, random_seed=self.random_seed if random_seed is None else random_seed
        ).float()
