import os
import sys

import click
import numpy as np
import json
from mpi4py import MPI

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments

import baselines.herebp.experiment.config as config

from baselines.her.rollout import RolloutWorker
from baselines.her.util import mpi_fork

import os.path as osp
import tempfile
import datetime


def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def train(policy, rollout_worker, evaluator, n_epochs, n_test_rollouts, n_cycles, n_batches, 
          policy_save_interval, save_policies, num_cpu, dump_buffer, w_potential, w_linear,
          w_rotational, rank_method, clip_energy, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    logger.info("Training...")
    best_success_rate = -1
    t = 1
    for epoch in range(n_epochs):
        # train
        rollout_worker.clear_history()
        for cycle in range(n_cycles):
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode, dump_buffer, w_potential, w_linear, w_rotational, rank_method, clip_energy)
            for batch in range(n_batches):
                t = ((epoch*n_cycles*n_batches)+(cycle*n_batches)+batch)*num_cpu
                policy.train(t, dump_buffer)

            policy.update_target_net()

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

            if dump_buffer:
                policy.dump_buffer(epoch)

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            #evaluator.save_policy(best_policy_path)
            #evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            #evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def launch(
    env_name, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return,
    temperature, prioritization, binding, version, dump_buffer, n_cycles, rank_method,
    w_potential, w_linear, w_rotational, clip_energy, override_params={}, save_policies=True):

    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        #whoami = mpi_fork(num_cpu, binding)
        whoami = mpi_fork(num_cpu)
        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)
    """
    if logging: 
        logdir = 'logs/'+str(env_name)+'-temperature'+str(temperature)+\
                 '-prioritization'+str(prioritization)+'-replay_strategy'+str(replay_strategy)+\
                 '-n_epochs'+str(n_epochs)+'-num_cpu'+str(num_cpu)+'-seed'+str(seed)+\
                 '-n_cycles'+str(n_cycles)+'-rank_method'+str(rank_method)+\
                 '-w_potential'+str(w_potential)+'-w_linear'+str(w_linear)+'-w_rotational'+str(w_rotational)+\
                 '-clip_energy'+str(clip_energy)+\
                 '-version'+str(version)
    else:
        logdir = osp.join(tempfile.gettempdir(),
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))

    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)
    """

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['env_name'] = env_name
    params['replay_strategy'] = replay_strategy
    params['temperature'] = temperature
    params['prioritization'] = prioritization
    params['binding'] = binding
    params['max_timesteps'] = n_epochs * params['n_cycles'] *  params['n_batches'] * num_cpu
    params['version'] = version
    params['dump_buffer'] = dump_buffer
    params['n_cycles'] = n_cycles
    params['rank_method'] = rank_method
    params['w_potential'] = w_potential
    params['w_linear'] = w_linear
    params['w_rotational'] = w_rotational
    params['clip_energy'] = clip_energy
    params['n_epochs'] = n_epochs
    params['num_cpu'] = num_cpu

    if params['dump_buffer']:
        params['alpha'] =0

    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    train(
        logdir=logdir, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, save_policies=save_policies,
        num_cpu=num_cpu, dump_buffer=dump_buffer, w_potential=params['w_potential'], 
        w_linear=params['w_linear'], w_rotational=params['w_rotational'], rank_method=rank_method,
        clip_energy=clip_energy)


@click.command()
@click.option('--env_name', type=click.Choice(["FetchReach-v1", "HandManipulateBlockRotateZ-v0", "HandManipulateBlockRotateXYZ-v0",'FetchPickAndPlace-v0', "HandReach-v0", 'HandManipulateBlockFull-v0', \
        'HandManipulateEggFull-v0', 'HandManipulatePenRotate-v0']), default='FetchPickAndPlace-v0', help='the name of the OpenAI Gym \
        environment that you want to train on. We tested EBP on four challenging robotic manipulation tasks, including: \
        FetchPickAndPlace-v0, HandManipulateBlockFull-v0, HandManipulateEggFull-v0, HandManipulatePenRotate-v0')

@click.option('--logdir', type=str, default=None, help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')

@click.option('--n_epochs', type=int, default=50, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'final', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--temperature', type=float, default=1.0, help='temperature value for Enery-Based Prioritization (EBP)')
@click.option('--prioritization', type=click.Choice(['none', 'energy', 'tderror']), default='energy', help='the prioritization strategy to be used. "energy" uses EBP;\
                                                                                                             "none" is vanilla HER; tderror is Prioritized Experience Replay.')
@click.option('--binding', type=click.Choice(['none', 'core']), default='core', help='configure mpi using bind-to none or core.')
#@click.option('--logging', type=bool, default=False, help='whether or not logging')
@click.option('--version', type=int, default=0, help='version')
@click.option('--dump_buffer', type=bool, default=False, help='dump buffer contains achieved goals, energy, tderrors for analysis')
@click.option('--n_cycles', type=int, default=25, help='n_cycles')
@click.option('--rank_method', type=click.Choice(['none', 'min', 'dense', 'average']), default='none', help='energy ranking method')
@click.option('--w_potential', type=float, default=1.0, help='w_potential')
@click.option('--w_linear', type=float, default=1.0, help='w_linear')
@click.option('--w_rotational', type=float, default=1.0, help='w_rotational')
@click.option('--clip_energy', type=float, default=999, help='clip_energy')

def main(**kwargs):
    launch(**kwargs)

if __name__ == '__main__':
    main()
