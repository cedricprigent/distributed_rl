import os
import reverb
import tempfile
import portpicker
import multiprocessing as mp
import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.agents.dqn import dqn_agent

# from tf_agents.environments import suite_pybullet
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
    lr = 3e-4
    gamma = 0.99
    actor_fc_layer_params = (256, 256)
    critic_fc_layer_params =(256, 256)

    log_interval = 5000
    num_eval_episodes = 20
    eval_interval = 10000
    policy_saved_interval = 5000

    NUM_WORKERS = 3
    NUM_PS = 2

    # Distribution Strategy
    os.environ["GRPC_FAIL_FAST"] = "use_caller"
    cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)

    variable_partitioner = (
        tf.distribute.experimental.partitioners.MinSizePartitioner(
            min_shard_bytes=(256 << 10),
            max_shards=NUM_PS
        )
    )

    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver,
        variable_partitioner=variable_partitioner
    )


    # ClusterCoordinator
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

    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    def dense_layer(num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # it's output.
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])


    # Agent
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    # Worker step function
    @tf.function
    def run_ep():

        env.reset()
        # def run_step():




def create_in_process_cluster(num_workers, num_ps):
    worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
    ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]
    cluster_dict = {}
    cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
    if num_ps > 0:
        cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

    cluster_spec = tf.train.ClusterSpec(cluster_dict)

    worker_config = tf.compat.v1.ConfigProto()
    if mp.cpu_count() < num_workers + 1:
        worker_config.inter_op_parallelism_threads = num_workers + 1

    # Définition d'un tf.distribute.Server pour chaque Worker et Parameter Server
    # Les serveurs appartenant à un même cluster peuvent communiquer les uns avec les autres
    for i in range(num_workers):
        tf.distribute.Server(
            cluster_spec,
            job_name="worker",
            task_index=i,
            config=worker_config,
            protocol="grpc"
        )

    for i in range(num_ps):
        tf.distribute.Server(
            cluster_spec,
            job_name="ps",
            task_index=i,
            protocol="grpc"
        )

    cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cluster_spec,
        rpc_layer="grpc"
    )

    return cluster_resolver


if __name__ == '__main__':
    main()

