from __future__ import division, print_function

import numpy as np
import torch


class Problem(object):
    def __init__(self, dimension, lower_bounds, upper_bounds):
        self.dimension = dimension
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        pass

    def c(self, s, t, M, D, L, tau):
        print(
            (M * np.exp(-(s**2) / 4 * D * t)) / np.sqrt(4 * np.pi * D * t)
            + ((t > tau) * M * np.exp(-((s - L) ** 2) / (4 * D * (t - tau)))) / np.sqrt(4 * np.pi * D * (t - tau))
        )
        val = (M * np.exp(-(s**2) / 4 * D * t)) / np.sqrt(4 * np.pi * D * t)
        if t > tau:
            val += ((t > tau) * M * np.exp(-((s - L) ** 2) / (4 * D * (t - tau)))) / np.sqrt(4 * np.pi * D * (t - tau))
        return val

    def __call__(self, x):
        tau = 30.01 + x[0] / 1000
        M = 7 + ((x[1] + 1) * 6) / 2
        D = 0.02 + ((x[2] + 1) * 0.10) / 2
        L = 0.01 + ((x[3] + 1) * 2.99) / 2
        print(f"M:{M}, D:{D}, L:{L}, tau:{tau}")
        val = 0.0
        for s in [0, 1, 2.5]:
            for t in [15, 30, 45, 60]:
                print(f"s:{s}, t:{t}")
                val += (self.c(s, t, 10, 0.07, 1.505, 30.1525) - self.c(s, t, M, D, L, tau)) ** 2
        print(f"val:{val}")
        return val


class EM_func(object):
    """
    Environmental model function
    """

    def __init__(self, random_seed=None, problem_id=None):
        self.random_seed = random_seed
        self.problem_id = problem_id
        self.lower_bounds = []
        self.upper_bounds = []
        # discrete variables
        self.lower_bounds.append(0)
        self.upper_bounds.append(284)
        for i in range(3):
            self.lower_bounds.append(-1)
            self.upper_bounds.append(1)
        assert len(self.lower_bounds) == 4
        assert len(self.upper_bounds) == 4
        # print(lower_bounds)
        # print(upper_bounds)
        self.num_discrete = 1
        self.num_continuous = 3

        self.problem = Problem(dimension=4, lower_bounds=self.lower_bounds, upper_bounds=self.upper_bounds)
        self.problem.dimension = 4
        print(f"num_discrete: {self.num_discrete}, num_continuous: {self.num_continuous}")
        self.n_vertices = []
        for i in range(self.num_discrete):
            self.n_vertices.append(self.upper_bounds[i] - self.lower_bounds[i] + 1)
        self.n_vertices = np.array(self.n_vertices)
        self.suggested_init = self.generate_random_points(n_points=10, random_seed=random_seed).float()
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
        print(evaluation)
        return torch.tensor(evaluation).float()

    def sample_points(self, n_points, random_seed=None):
        if random_seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(random_seed)
        init_points = []
        for _ in range(n_points):
            random_point = []
            for i in range(self.num_discrete):
                random_point.append(torch.randint(self.lower_bounds[i], self.upper_bounds[i] + 1, (1,)))
            for i in range(self.num_discrete, self.num_discrete + self.num_continuous):
                random_point.append(torch.FloatTensor(1).uniform_(-1, 1))
            init_points.append(random_point)
        return torch.tensor(init_points).float()

    def generate_random_points(self, n_points, random_seed=None):
        return self.sample_points(
            n_points, random_seed=self.random_seed if random_seed is None else random_seed
        ).float()
