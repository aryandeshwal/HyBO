from __future__ import division, print_function

import numpy as np
import torch
from hyperopt import hp

from experiments.test_functions.robot_push_14d import push_function


class Problem(object):
    def __init__(self, dimension, lower_bounds, upper_bounds):
        self.dimension = dimension
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        pass

    def __call__(self, x):
        argv = []
        argv.append(x[0] - 5)
        argv.append(x[1] - 5)
        argv.append(x[4] - 10)
        argv.append(x[5] - 10)
        argv.append(x[8] + 2)
        argv.append(((x[10] + 1) * 2 * np.pi) / 2)
        argv.append(x[2] - 5)
        argv.append(x[3] - 5)
        argv.append(x[6] - 10)
        argv.append(x[7] - 10)
        argv.append(x[9] + 2)
        argv.append(((x[11] + 1) * 2 * np.pi) / 2)
        argv.append(((x[12] + 1) * 10) / 2)
        argv.append(((x[13] + 1) * 10) / 2)
        f = push_function.PushReward()
        reward = f(argv)
        print(f"reward: {reward}")
        return -1 * reward


class Push_robot_14d(object):
    def __init__(self, random_seed=None, problem_id=None):
        self.random_seed = random_seed
        self.problem_id = problem_id
        self.lower_bounds = []
        self.upper_bounds = []
        # discrete variables
        for i in range(4):
            self.lower_bounds.append(0)
            self.upper_bounds.append(10)
        for i in range(4):
            self.lower_bounds.append(0)
            self.upper_bounds.append(20)
        for i in range(2):
            self.lower_bounds.append(0)
            self.upper_bounds.append(28)
        for i in range(4):
            self.lower_bounds.append(-1)
            self.upper_bounds.append(1)
        assert len(self.lower_bounds) == 14
        assert len(self.upper_bounds) == 14
        # print(lower_bounds)
        # print(upper_bounds)
        self.num_discrete = 10
        self.num_continuous = 4

        self.problem = Problem(dimension=14, lower_bounds=self.lower_bounds, upper_bounds=self.upper_bounds)
        self.problem.dimension = 14
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
                random_point.append(torch.randint(self.lower_bounds[i], self.upper_bounds[i], (1,)))
            for i in range(self.num_discrete, self.num_discrete + self.num_continuous):
                random_point.append(torch.FloatTensor(1).uniform_(-1, 1))
            init_points.append(random_point)
        return torch.tensor(init_points).float()

    def generate_random_points(self, n_points, random_seed=None):
        return self.sample_points(
            n_points, random_seed=self.random_seed if random_seed is None else random_seed
        ).float()
