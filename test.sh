#!/bin/bash
T_TRAIN=100
for SEED in 34164 63882 88106 60166 75316 48784 50067 65846 1837 43786; do
	for ENVIRONMENT in CorridorSmall-v5 FrozenLake-v0; do
		echo "Vanilla DQN (seed=$SEED)"
		python main.py --env_name=$ENVIRONMENT --async_threads=8 --agent_type=Async \
			--history_length=1 --t_learn_start=0 --learning_rate_decay_step=10 \
			--learning_rate=-1 --learning_rate_minimum=-1 --n_action_repeat=1 \
			--network_header_type=mlp --network_output_type=normal --observation_dims='[16]' \
			--t_ep_end=10 --trace_steps=5 --use_gpu=False \
			--entropy_regularization_minimum=0.0 --entropy_regularization=0.0 \
			--max_grad_norm=0.0 --learning_rate_decay=0.96 --momentum=0.9 --ep_start=0.5 \
			--ep_end=0.0 --t_ep_end=50 --t_train_max=$T_TRAIN --double_q=False --random_seed=$SEED \
			"--model_dir=$ENVIRONMENT-DQN-$SEED"

		echo "Double DQN (seed=$SEED)"
		python main.py \
			--env_name=$ENVIRONMENT --async_threads=8 --agent_type=Async \
			--history_length=1 --t_learn_start=0 --learning_rate_decay_step=10 \
			--learning_rate=-1 --learning_rate_minimum=-1 --n_action_repeat=1 \
			--network_header_type=mlp --network_output_type=normal --observation_dims='[16]' \
			--t_ep_end=10 --trace_steps=5 --use_gpu=False \
			--entropy_regularization_minimum=0.0 --entropy_regularization=0.0 \
			--max_grad_norm=0.0 --learning_rate_decay=0.96 --momentum=0.9 --ep_start=0.5 \
			--ep_end=0.0 --t_ep_end=50 --t_train_max=$T_TRAIN --double_q=True --random_seed=$SEED \
			"--model_dir=$ENVIRONMENT-2DQN-$SEED"

		echo "Dueling Double DQN (seed=$SEED)"
		python main.py \
			--env_name=$ENVIRONMENT --async_threads=8 --agent_type=Async \
			--history_length=1 --t_learn_start=0 --learning_rate_decay_step=10 \
			--learning_rate=-1 --learning_rate_minimum=-1 --n_action_repeat=1 \
			--network_header_type=mlp --network_output_type=dueling \
			--observation_dims='[16]' --t_ep_end=10 --trace_steps=5 --use_gpu=False \
			--entropy_regularization_minimum=0.0 --entropy_regularization=0.0 \
			--max_grad_norm=0.0 --learning_rate_decay=0.96 --momentum=0.9 --ep_start=0.5 \
			--ep_end=0.0 --t_ep_end=50 --t_train_max=$T_TRAIN --double_q=True --random_seed=$SEED \
			"--model_dir=$ENVIRONMENT-D2DQN-$SEED"


		echo "Dueling DQN (seed=$SEED)"
		python main.py \
			--env_name=$ENVIRONMENT --async_threads=8 --agent_type=Async \
			--history_length=1 --t_learn_start=0 --learning_rate_decay_step=10 \
			--learning_rate=-1 --learning_rate_minimum=-1 --n_action_repeat=1 \
			--network_header_type=mlp --network_output_type=dueling \
			--observation_dims='[16]' --t_ep_end=10 --trace_steps=5 --use_gpu=False \
			--entropy_regularization_minimum=0.0 --entropy_regularization=0.0 \
			--max_grad_norm=0.0 --learning_rate_decay=0.96 --momentum=0.9 --ep_start=0.5 \
			--ep_end=0.0 --t_ep_end=50 --t_train_max=$T_TRAIN --double_q=False --random_seed=$SEED \
			"--model_dir=$ENVIRONMENT-DDQN-$SEED"

		echo "Async Advantage Actor Critic (seed=$SEED), disjoint"
		python main.py \
			--env_name=$ENVIRONMENT --async_threads=8 --agent_type=Async --history_length=1 \
			--t_learn_start=0 --learning_rate_decay_step=10 --learning_rate=0.0025 \
			--learning_rate_minimum=0.0025 --n_action_repeat=1 --network_header_type=mlp \
			--network_output_type=actor_critic --observation_dims='[16]' --t_ep_end=10 \
			--trace_steps=5 --use_gpu=False --entropy_regularization_minimum=0.0 \
			--entropy_regularization=0.01 --entropy_regularization_decay_step=0.1 \
			--max_grad_norm=0.0 --learning_rate_decay=0.96 --momentum=0.9 --t_train_max=$T_TRAIN \
			--disjoint_a3c=True --random_seed=$SEED "--model_dir=$ENVIRONMENT-A3C-$SEED"

	done
done

