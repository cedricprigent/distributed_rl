import argparse

import reverb
import tempfile
import os
import json
import tensorflow as tf
import time

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network

# from tf_agents.environments import suite_pybullet
from tf_agents.agents.dqn import dqn_agent

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import sequential

from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy

from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers

from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

from tf_agents.specs import tensor_spec
from tf_agents.utils import common

import gym



def main():
    tempdir = tempfile.gettempdir()

    # Hyperparams
    env_name = "CartPole-v0"
    num_iterations = 100000
    lr = 1e-4
    gamma = 0.99
    actor_fc_layer_params = (256, 256)
    critic_fc_layer_params = (256, 256)

    log_interval = 5000
    num_eval_episodes = 20
    eval_interval = 10000
    policy_saved_interval = 5000

    NUM_WORKERS = 1
    NUM_PS = 1

    # Argument Parser
    parser = argparse.ArgumentParser()

    parser.add_argument('id', help="task id")
    parser.add_argument('type', help="task type (chief, ps, worker)")
    args = parser.parse_args()

    # Cluster Configuration
    task = {'type': args.type, 'index': args.id}

    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": {
            "ps": ['localhost:8000'],
            "worker": ['localhost:8100', 'localhost:8101'],
            "chief": ['localhost:8200']
        },
        'task': task
    })

    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()


    if cluster_resolver.task_type in ('worker', 'ps'):
        # If Worker or PS: start a TF Server and wait
        print('Starting TF Server..')
        os.environ["GRPC_FAIL_FAST"] = "use_caller"

        server = tf.distribute.Server(
            cluster_resolver.cluster_spec(),
            job_name=cluster_resolver.task_type,
            task_index=cluster_resolver.task_id,
            protocol=cluster_resolver.rpc_layer or "grpc",
            start=True
        )
        server.join()

    else:
        # Run the Cluster Coordinator
        print('Starting Coordinator..')
        strategy = tf.distribute.experimental.ParameterServerStrategy(
            cluster_resolver=cluster_resolver
        )
        coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)



        # Environment
        train_py_env = suite_gym.load(env_name)
        eval_py_env = suite_gym.load(env_name)

        train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


        # Network
        fc_layer_params = (100, 50)
        action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        def dense_layer(num_units):
            return tf.keras.layers.Dense(
                num_units,
                activation=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=2.0, mode='fan_in', distribution='truncated_normal'
                )
            )

        dense_layer = [dense_layer(num_units) for num_units in fc_layer_params]
        q_values_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03
            ),
            bias_initializer=tf.keras.initializers.Constant(-0.2)
        )
        q_net = sequential.Sequential(dense_layer + [q_values_layer])


        # Agent
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        train_step_counter = tf.Variable(0)

        agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter
        )

        agent.initialize()

        with strategy.scope():
            v1 = tf.Variable(initial_value=0.0)
            v2 = tf.Variable(initial_value=1.0)

        # Worker step function
        @tf.function
        def run_ep():
            v1.assign_add(0.1)
            v2.assign_sub(0.1)
            return v1.read_value() / v2.read_value()

        start = time.time()
        result = 0
        for _ in range(10000):
            result = coordinator.schedule(run_ep)

        coordinator.join()
        print(result.fetch())
        end = time.time()
        print(f'time: {end - start}')


# Instruction for running
# python dist_agent.py <task_id> <task_type>

if __name__ == '__main__':
    main()
