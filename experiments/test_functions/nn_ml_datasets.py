from __future__ import division, print_function

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neural_network import MLPClassifier

import experiments.test_functions.ml_datasets as mld


class Problem(object):
    def __init__(self, dataset, dimension, num_discrete, num_continuous, lower_bounds, upper_bounds):
        self.dataset = dataset
        self.dimension = dimension
        self.num_discrete = num_discrete
        self.num_continuous = num_continuous
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.train_X, self.train_y, self.test_X, self.test_y = mld.gen_train_test_data(self.dataset)

    def __call__(self, x):
        hidden_layer_sizes = list(range(40, 310, 20))
        activation = ["identity", "logistic", "tanh", "relu"]
        batch_size = list(range(40, 210, 20))
        learning_rate = ["constant", "invscaling", "adaptive"]
        early_stopping = [False, True]
        learning_rate_init = [0.001, 1]
        momentum = [0.5, 1]
        alpha = [0.0001, 1]
        x[5] = 0.001 + ((x[5] + 1) * (1 - 0.001)) / 2
        x[6] = 0.5 + ((x[6] + 1) * (1 - 0.5)) / 2
        x[7] = 0.0001 + ((x[7] + 1) * (1 - 0.0001)) / 2
        print(f"x after conversion {x}")
        kwag_dict = {
            "hidden_layer_sizes": hidden_layer_sizes[int(x[0])],
            "activation": activation[int(x[1])],
            "batch_size": batch_size[int(x[2])],
            "learning_rate": learning_rate[int(x[3])],
            "early_stopping": early_stopping[int(x[4])],
            "learning_rate_init": x[5],
            "momentum": x[6],
            "alpha": x[7],
            "solver": "sgd",
        }
        print(kwag_dict)
        model = MLPClassifier(**kwag_dict, random_state=2)
        model.fit(self.train_X, self.train_y)
        pred_y = model.predict(self.test_X)
        # maximize accuracy
        auc = accuracy_score(self.test_y, pred_y)
        print(f"auc: {auc}")
        return -auc


class NN_ML_Datasets(object):
    def __init__(self, random_seed=None, problem_id=None):
        self.random_seed = random_seed
        self.problem_id = problem_id
        self.lower_bounds = []
        self.upper_bounds = []
        # Discrete bounds
        self.lower_bounds.append(0)
        self.upper_bounds.append(13)
        self.lower_bounds.append(0)
        self.upper_bounds.append(3)

        self.lower_bounds.append(0)
        self.upper_bounds.append(8)

        self.lower_bounds.append(0)
        self.upper_bounds.append(2)
        self.lower_bounds.append(0)
        self.upper_bounds.append(1)

        # Continuous bounds
        for i in range(3):
            self.lower_bounds.append(-1)
            self.upper_bounds.append(1)
        assert len(self.lower_bounds) == 8
        assert len(self.upper_bounds) == 8
        print(self.lower_bounds)
        print(self.upper_bounds)
        self.num_discrete = 5
        self.num_continuous = 3
        self.dimension = 8
        assert self.dimension == self.num_discrete + self.num_continuous
        self.problem = Problem(
            dataset=self.problem_id,
            dimension=self.dimension,
            num_discrete=self.num_discrete,
            num_continuous=self.num_continuous,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )
        print(f"num_discrete: {self.num_discrete}, num_continuous: {self.num_continuous}")
        self.n_vertices = []
        for i in range(self.num_discrete):
            self.n_vertices.append(self.upper_bounds[i] - self.lower_bounds[i] + 1)
        self.n_vertices = np.array(self.n_vertices)
        self.suggested_init = self.sample_points(n_points=10, random_seed=random_seed).float()
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
        # assert x.numel() == len(self.n_vertices)
        # x = x.numpy()
        if x_unorder.dim() == 2:
            x_unorder = x_unorder.squeeze(0)
        x = x_unorder.numpy().copy()
        print(f"evaluating {x}....")
        evaluation = self.problem(x)
        # print(evaluation)
        # print(type(evaluation))
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
