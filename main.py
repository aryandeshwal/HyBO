import argparse
import os
import sys
import time

import torch

from acquisition.acquisition_functions import expected_improvement
from acquisition.acquisition_marginalization import inference_sampling
from acquisition.acquisition_optimization import next_evaluation
from config import experiment_directory
from experiments.random_seed_config import generate_random_seed_coco
from experiments.test_functions.em_func import EM_func
from experiments.test_functions.mixed_integer import MixedIntegerCOCO

# from experiments.test_functions.push_robot_14d import Push_robot_14d
from experiments.test_functions.nn_ml_datasets import NN_ML_Datasets
from experiments.test_functions.pressure_vessel_design import Pressure_Vessel_Design
from experiments.test_functions.speed_reducer import SpeedReducer
from experiments.test_functions.weld_design import Weld_Design
from GPmodel.inference.inference import Inference
from GPmodel.kernels.mixeddiffusionkernel import MixedDiffusionKernel
from GPmodel.models.gp_regression import GPRegression
from GPmodel.sampler.sample_mixed_posterior import posterior_sampling
from GPmodel.sampler.tool_partition import group_input
from utils import displaying_and_logging, load_model_data, model_data_filenames


def HyBO(objective=None, n_eval=200, path=None, parallel=False, store_data=True, problem_id=None, **kwargs):
    """
    :param objective:
    :param n_eval:
    :param path:
    :param parallel:
    :param kwargs:
    :return:
    """
    acquisition_func = expected_improvement

    n_vertices = adj_mat_list = None
    eval_inputs = eval_outputs = log_beta = sorted_partition = lengthscales = None
    time_list = elapse_list = pred_mean_list = pred_std_list = pred_var_list = None

    if objective is not None:
        exp_dir = experiment_directory()
        objective_id_list = [objective.__class__.__name__]
        if hasattr(objective, "random_seed_info"):
            objective_id_list.append(objective.random_seed_info)
        if hasattr(objective, "data_type"):
            objective_id_list.append(objective.data_type)
        objective_id_list.append("HyBO")
        if problem_id is not None:
            objective_id_list.append(problem_id)
        objective_name = "_".join(objective_id_list)
        model_filename, data_cfg_filaname, logfile_dir = model_data_filenames(
            exp_dir=exp_dir, objective_name=objective_name
        )

        n_vertices = objective.n_vertices
        adj_mat_list = objective.adjacency_mat
        grouped_log_beta = torch.ones(len(objective.fourier_freq))
        log_order_variances = torch.zeros((objective.num_discrete + objective.num_continuous))
        fourier_freq_list = objective.fourier_freq
        fourier_basis_list = objective.fourier_basis
        suggested_init = objective.suggested_init  # suggested_init should be 2d tensor
        n_init = suggested_init.size(0)
        num_discrete = objective.num_discrete
        num_continuous = objective.num_continuous
        lengthscales = torch.zeros((num_continuous))
        print("******************* initializing kernel ****************")
        kernel = MixedDiffusionKernel(
            log_order_variances=log_order_variances,
            grouped_log_beta=grouped_log_beta,
            fourier_freq_list=fourier_freq_list,
            fourier_basis_list=fourier_basis_list,
            lengthscales=lengthscales,
            num_discrete=num_discrete,
            num_continuous=num_continuous,
        )
        surrogate_model = GPRegression(kernel=kernel)
        eval_inputs = suggested_init
        eval_outputs = torch.zeros(eval_inputs.size(0), 1, device=eval_inputs.device)
        for i in range(eval_inputs.size(0)):
            eval_outputs[i] = objective.evaluate(eval_inputs[i])
        assert not torch.isnan(eval_outputs).any()
        log_beta = eval_outputs.new_zeros(num_discrete)
        log_order_variance = torch.zeros((num_discrete + num_continuous))
        sorted_partition = [[m] for m in range(num_discrete)]
        lengthscale = torch.zeros((num_continuous))

        time_list = [time.time()] * n_init
        elapse_list = [0] * n_init
        pred_mean_list = [0] * n_init
        pred_std_list = [0] * n_init
        pred_var_list = [0] * n_init

        surrogate_model.init_param(eval_outputs)
        print("(%s) Burn-in" % time.strftime("%H:%M:%S", time.localtime()))
        sample_posterior = posterior_sampling(
            surrogate_model,
            eval_inputs,
            eval_outputs,
            n_vertices,
            adj_mat_list,
            log_order_variance,
            log_beta,
            lengthscale,
            sorted_partition,
            n_sample=1,
            n_burn=1,
            n_thin=1,
        )
        log_order_variance = sample_posterior[1][0]
        log_beta = sample_posterior[2][0]
        lengthscale = sample_posterior[3][0]
        sorted_partition = sample_posterior[4][0]
        print("")
    else:
        surrogate_model, cfg_data, logfile_dir = load_model_data(path, exp_dir=experiment_directory())

    for i in range(n_eval):
        start_time = time.time()
        reference = torch.min(eval_outputs, dim=0)[0].item()
        print(f"Iteration {i}")
        print("(%s) Sampling" % time.strftime("%H:%M:%S", time.localtime()))
        sample_posterior = posterior_sampling(
            surrogate_model,
            eval_inputs,
            eval_outputs,
            n_vertices,
            adj_mat_list,
            log_order_variance,
            log_beta,
            lengthscale,
            sorted_partition,
            n_sample=10,
            n_burn=0,
            n_thin=1,
        )
        (
            hyper_samples,
            log_order_variance_samples,
            log_beta_samples,
            lengthscale_samples,
            partition_samples,
            freq_samples,
            basis_samples,
            edge_mat_samples,
        ) = sample_posterior
        log_order_variance = log_order_variance_samples[-1]
        log_beta = log_beta_samples[-1]
        lengthscale = lengthscale_samples[-1]
        sorted_partition = partition_samples[-1]
        print("\n")
        # print(hyper_samples[0])
        # print(log_order_variance)
        # print(log_beta)
        # print(lengthscale)
        # print(sorted_partition)
        # print('')

        x_opt = eval_inputs[torch.argmin(eval_outputs)]
        inference_samples = inference_sampling(
            eval_inputs,
            eval_outputs,
            n_vertices,
            hyper_samples,
            log_order_variance_samples,
            log_beta_samples,
            lengthscale_samples,
            partition_samples,
            freq_samples,
            basis_samples,
            num_discrete,
            num_continuous,
        )
        suggestion = next_evaluation(
            objective,
            x_opt,
            eval_inputs,
            inference_samples,
            partition_samples,
            edge_mat_samples,
            n_vertices,
            acquisition_func,
            reference,
            parallel,
        )
        next_eval, pred_mean, pred_std, pred_var = suggestion

        processing_time = time.time() - start_time
        print("next_eval", next_eval)

        eval_inputs = torch.cat([eval_inputs, next_eval.view(1, -1)], 0)
        eval_outputs = torch.cat([eval_outputs, objective.evaluate(eval_inputs[-1]).view(1, 1)])
        assert not torch.isnan(eval_outputs).any()
        time_list.append(time.time())
        elapse_list.append(processing_time)
        pred_mean_list.append(pred_mean.item())
        pred_std_list.append(pred_std.item())
        pred_var_list.append(pred_var.item())

        displaying_and_logging(
            logfile_dir,
            eval_inputs,
            eval_outputs,
            pred_mean_list,
            pred_std_list,
            pred_var_list,
            time_list,
            elapse_list,
            hyper_samples,
            log_beta_samples,
            lengthscale_samples,
            log_order_variance_samples,
            store_data,
        )
        print(
            "Optimizing %s with regularization %.2E up to %4d visualization random seed : %s"
            % (
                objective.__class__.__name__,
                objective.lamda if hasattr(objective, "lamda") else 0,
                n_eval,
                objective.random_seed_info if hasattr(objective, "random_seed_info") else "none",
            )
        )


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser(description="Hybrid Bayesian optimization using additive diffusion kernels")
    parser_.add_argument("--n_eval", dest="n_eval", type=int, default=220)
    parser_.add_argument("--n_expts", dest="n_expts", type=int, default=1)
    parser_.add_argument("--objective", dest="objective")
    parser_.add_argument("--problem_id", dest="problem_id", type=str, default=None)

    args_ = parser_.parse_args()
    kwag_ = vars(args_)
    objective_ = kwag_["objective"]
    print(kwag_)
    for i in range(kwag_["n_expts"]):
        if objective_ == "coco":
            random_seed_ = sorted(generate_random_seed_coco())[i]
            kwag_["objective"] = MixedIntegerCOCO(random_seed_, problem_id=kwag_["problem_id"])
        elif objective_ == "weld_design":
            random_seed_ = sorted(generate_random_seed_coco())[i]
            kwag_["objective"] = Weld_Design(random_seed_, problem_id=kwag_["problem_id"])
        elif objective_ == "speed_reducer":
            random_seed_ = sorted(generate_random_seed_coco())[i]
            kwag_["objective"] = SpeedReducer(random_seed_, problem_id=kwag_["problem_id"])
        elif objective_ == "pressure_vessel":
            random_seed_ = sorted(generate_random_seed_coco())[i]
            kwag_["objective"] = Pressure_Vessel_Design(random_seed_, problem_id=kwag_["problem_id"])
        # elif objective_ == 'push_robot':
        #   random_seed_ = sorted(generate_random_seed_coco())[i]
        #   kwag_['objective'] = Push_robot_14d(random_seed_, problem_id=kwag_['problem_id'])
        elif objective_ == "em_func":
            random_seed_ = sorted(generate_random_seed_coco())[i]
            kwag_["objective"] = EM_func(random_seed_, problem_id=kwag_["problem_id"])
        elif objective_ == "nn_ml_datasets":
            random_seed_ = sorted(generate_random_seed_coco())[i]
            kwag_["objective"] = NN_ML_Datasets(random_seed_, problem_id=kwag_["problem_id"])
        else:
            raise NotImplementedError
        HyBO(**kwag_)
