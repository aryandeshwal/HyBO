from __future__ import division, print_function

import numpy as np
import torch


class Problem(object):
    def __init__(self, dimension, lower_bounds, upper_bounds):
        self.dimension = dimension
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        pass

    def __call__(self, x_unorder):
        w = x_unorder[0]  # w
        m = x_unorder[1]  # m
        h = 0.0625 + ((x_unorder[2] + 1) * (2 - 0.0625)) / 2  # h
        l = ((x_unorder[3] + 1) * (10 - 0)) / 2  # l
        t = 2 + ((x_unorder[4] + 1) * (20 - 2)) / 2  # t
        b = 0.0625 + ((x_unorder[5] + 1) * (2 - 0.0625)) / 2  # b
        print(f"w: {w}, m: {m}, h: {h}, l: {l}, t: {t}, b: {b}")
        if w == 0:
            A = np.sqrt(2) * h * l
            J = np.sqrt(2) * h * l * (((h + t) ** 2) / 4 + (l**2) / 12)
            R = np.sqrt(l**2 + (h + t) ** 2) / 2
            costheta = l / (2 * R)
        else:
            A = np.sqrt(2) * h * (t + l)
            J = np.sqrt(2) * h * l * (((h + t) ** 2) / 4 + (l**2) / 12) + np.sqrt(2) * h * t * (
                ((h + l) ** 2) / 4 + (t**2) / 12
            )
            R = max(np.sqrt(l**2 + (h + t) ** 2) / 2, np.sqrt(t**2 + (h + l) ** 2) / 2)
            costheta = l / (2 * R)
        if m == 0:
            C_1, C_2, E, sd, G = 0.1047, 0.0481, 3e7, 3e4, 12e6
        elif m == 1:
            C_1, C_2, E, sd, G = 0.0489, 0.0224, 14e6, 8e3, 6e6
        elif m == 2:
            C_1, C_2, E, sd, G = 0.5235, 0.2405, 1e7, 5e3, 4e6
        else:
            C_1, C_2, E, sd, G = 0.5584, 0.2566, 16e6, 8e3, 6e6
        L = 14
        F = 6000
        delta_max = 0.25
        sigma = 6 * F * L / ((t**2) * b)
        delta = 4 * F * (L**3) / (E * (t**3) * b)
        pc = 4.013 * t * (b**3) * (np.sqrt(E * G)) ** (1 - (t * np.sqrt(E) / 4 * L * np.sqrt(G))) / (6 * (L**2))
        tau1 = F / A
        tau2 = F * (L + 0.05 * l) * R / J
        tau = np.sqrt(tau1**2 + tau2**2 + 2 * tau1 * tau2 * costheta)
        func_value = (1 + C_1) * (w * t + l) * (h**2) + C_2 * t * b * (14 + l)
        constraint_cost = (
            (0.577 * sd < tau) * 10 + (sd < sigma) * 10 + (b < h) * 10 + (pc < F) * 10 + (delta_max < delta) * 10
        )
        constraint_cost = 0
        return func_value + constraint_cost


class Weld_Design(object):
    def __init__(self, random_seed=None, problem_id=None):
        self.random_seed = random_seed
        self.problem_id = problem_id
        lower_bounds = []
        upper_bounds = []
        # discrete variables
        lower_bounds.append(0)  # weld_config (w)
        upper_bounds.append(1)
        lower_bounds.append(0)  # bulk_material (b)
        upper_bounds.append(3)

        for i in range(4):
            lower_bounds.append(-1)  # weld_thickness (h)
            upper_bounds.append(1)

        assert len(lower_bounds) == 6
        assert len(upper_bounds) == 6
        print(lower_bounds)
        print(upper_bounds)
        self.problem = Problem(dimension=6, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
        self.problem.dimension = 6
        self.num_discrete = 2
        self.num_continuous = 4
        print(f"num_discrete: {self.num_discrete}, num_continuous: {self.num_continuous}")
        self.n_vertices = [2, 4]
        # for i in range(2):
        #     self.n_vertices.append(10) # m values
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
            random_point.append(torch.randint(0, 2, (1,)))
            random_point.append(torch.randint(0, 4, (1,)))

            for i in range(4):
                random_point.append(torch.FloatTensor(1).uniform_(-1, 1))
            init_points.append(random_point)
        return torch.tensor(init_points).float()

    def generate_random_points(self, n_points, random_seed=None):
        return self.sample_points(
            n_points, random_seed=self.random_seed if random_seed is None else random_seed
        ).float()


if __name__ == "__main__":
    mixobj = Weld_Design(44, "weld_design")
    print(mixobj.suggested_init)
    print(mixobj.evaluate(mixobj.suggested_init[0]))
    print(mixobj.evaluate(mixobj.suggested_init[5]))
