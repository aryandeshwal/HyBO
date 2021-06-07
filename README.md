<div align="center">
# Bayesian Optimization over Hybrid Spaces
</div>

This repository contains the source code for the paper "[Bayesian Optimization over Hybrid Spaces]()" which will be presented at upcoming Thirty-eighth [ICML'21](https://icml.cc/Conferences/2021/AcceptedPapersInitial) conference. 

- By default, data is stored in `../EXPERIMENTS`. Directory can be changed in `config.py`
- The command-line arguments are described below:
         --n_eval : The number of evaluations
         --objective : ['coco'] (example on how to create new objective can be seen in experiments/test_functions/mixed_integer.py)
     --problem_id : applicable only for 'coco' and 'nn_ml_datasets' domain
         --path : A path to the directory of the experiment to be continued.(Only need when you want to resume an experiment)

- Example run
```python main.py --objective coco --problem_id bbob-mixint_f001_i01_d10  --n_eval 180```

- There are 7 benchmarks used in the paper. For using synthetic benchmark, please see interactions in [coco](https://github.com/numbbo/coco) suite. For robot pushing benchmark, please see the original description in [Ensemble-Bayesian-Optimization](https://github.com/zi-w/Ensemble-Bayesian-Optimization).

The discrete part of the code is built upon the [source code](https://github.com/QUVA-Lab/COMBO) provided by the COMBO authors. We thank them for their code and have added appropriate licenses. 

