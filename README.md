Assignments for [Berkeley CS 285: Deep Reinforcement Learning, Decision Making, and Control](http://rail.eecs.berkeley.edu/deeprlcourse/).

## Behavioral cloning

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Walker2d.pkl \
--env_name Walker2d-v4 --exp_name bc_Walker2d --n_iter 1 \
--expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl \
--video_log_freq -1 --num_agent_train_steps_per_iter 298
--eval_batch_size 10000 --ep_len 2000

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name bc_Ant --n_iter 1 \
--expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1 --num_agent_train_steps_per_iter 298
--eval_batch_size 10000 --ep_len 2000


## DAGGER

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
--do_dagger --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1 --num_agent_train_steps_per_iter 298
--eval_batch_size 10000 --ep_len 2000

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Walker2d.pkl \
--env_name Walker2d-v4 --exp_name dagger_Walker2d --n_iter 10 \
--do_dagger --expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl \
--video_log_freq -1 --num_agent_train_steps_per_iter 298
--eval_batch_size 10000 --ep_len 2000

