import reverb
import tempfile
import portpicker
import multiprocessing as mp
import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_pybullet
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network

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



def main():
    tempdir = tempfile.gettempdir()

    # Hyperparams
    env_name = "MinitaurBulletEnv-v0"
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
    cluster_resolver = create_in_process_cluster(num_workers, num_ps)

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



    # Environment
    env = suite_pybullet.load(env_name)
    env.reset()



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