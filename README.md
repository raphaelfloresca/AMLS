# AMLS
This repository contains the code for the computer vision final project of the ELEC0134 Applied Machine Learning Systems module at UCL.

## How to compile
In the `AMLS_19-20_Raphael_Angelo_Floresca_SN16011494` folder, compile `main.py`. The following command line arguments can be specified, otherwise it will run with the following default settings
- `--schedule_type`: specifies the type of learning rate schedule to run. Specify the specific learning rate schedules for the models in sequential order in the following format (e.g. `one_cycle,one_cycle,one_cycle,one_cycle`). Accepts `none`,`step`,`linear`,`poly` and `one_cycle`. Default: `one_cycle,one_cycle,one_cycle,one_cycle`
- `--epochs`: specifies the number of training epochs. Specify the specific epochs for the models in sequential order in the following format (e.g. `10,10,10,10`). Default: `10,10,10,10`
- `--learning_rates`: specifies the learning rates. Specify the specific learning rate for the models in sequential order in the following format (e.g. `0.1,0.2,0.1,0.01`). Default: `0.03,0.03,0.03,0.03`
- `--find_lr`: specifies whether the learning rate finder should be used. Default: `False`
- `--random_state`: specifies a random seed when creating the training, validation and test sets. Default: `None`
- `--model_type`: specifies the models used for each task. Specify the specific models for the tasks in sequential order in the following format (e.g. `mlp,mlp,mlp,mlp`). Accepts `mlp`,`cnn` and `xception` Default: `xception,xception,xception,xception`.
