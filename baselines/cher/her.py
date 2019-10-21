import numpy as np
import random
import baselines.cher.config_curriculum as config_cur
import math
from sklearn.neighbors import NearestNeighbors
from gym.envs.robotics import rotations


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def curriculum(transitions, batch_size_in_transitions, batch_size):
        sel_list = lazier_and_goals_sample_kg(
            transitions['g'], transitions['ag'], transitions['o'],
            batch_size_in_transitions)
        transitions = {
            key: transitions[key][sel_list].copy()
            for key in transitions.keys()
        }
        config_cur.learning_step += 1
        return transitions

    def fa(k, a_set, v_set, sim, row, col):
        if len(a_set) == 0:
            init_a_set = []
            marginal_v = 0
            for i in v_set:
                max_ki = 0
                if k == col[i]:
                    max_ki = sim[i]
                init_a_set.append(max_ki)
                marginal_v += max_ki
            return marginal_v, init_a_set

        new_a_set = []
        marginal_v = 0
        for i in v_set:
            sim_ik = 0
            if k == col[i]:
                sim_ik = sim[i]

            if sim_ik > a_set[i]:
                max_ki = sim_ik
                new_a_set.append(max_ki)
                marginal_v += max_ki - a_set[i]
            else:
                new_a_set.append(a_set[i])
        return marginal_v, new_a_set

    def lazier_and_goals_sample_kg(goals, ac_goals, obs,
                                   batch_size_in_transitions):
        if config_cur.goal_type == "ROTAION":
            goals, ac_goals = goals[..., 3:], ac_goals[..., 3:]

        num_neighbor = 1
        kgraph = NearestNeighbors(
            n_neighbors=num_neighbor, algorithm='kd_tree',
            metric='euclidean').fit(goals).kneighbors_graph(
                mode='distance').tocoo(copy=False)
        row = kgraph.row
        col = kgraph.col
        sim = np.exp(
            -np.divide(np.power(kgraph.data, 2),
                       np.mean(kgraph.data)**2))
        delta = np.mean(kgraph.data)

        sel_idx_set = []
        idx_set = [i for i in range(len(goals))]
        balance = config_cur.fixed_lambda
        if int(balance) == -1:
            balance = math.pow(
                1 + config_cur.learning_rate,
                config_cur.learning_step) * config_cur.lambda_starter
        v_set = [i for i in range(len(goals))]
        max_set = []
        for i in range(0, batch_size_in_transitions):
            sub_size = 3
            sub_set = random.sample(idx_set, sub_size)
            sel_idx = -1
            max_marginal = float("-inf")  #-1 may have an issue
            for j in range(sub_size):
                k_idx = sub_set[j]
                marginal_v, new_a_set = fa(k_idx, max_set, v_set, sim, row,
                                           col)
                euc = np.linalg.norm(goals[sub_set[j]] - ac_goals[sub_set[j]])
                marginal_v = marginal_v - balance * euc
                if marginal_v > max_marginal:
                    sel_idx = k_idx
                    max_marginal = marginal_v
                    max_set = new_a_set

            idx_set.remove(sel_idx)
            sel_idx_set.append(sel_idx)
        return np.array(sel_idx_set)

    # does not use it: from gym https://github.com/openai/gym/blob/master/gym/envs/robotics/hand/manipulate.py#L87
    def _goal_rot_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        assert goal_a.shape[-1] == 7
        d_rot = np.zeros_like(goal_b[..., 0])
        quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]
        # Subtract quaternions and extract angle between them.
        quat_diff = rotations.quat_mul(quat_a,
                                       rotations.quat_conjugate(quat_b))
        angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
        d_rot = angle_diff
        return d_rot

    # does not use it
    def lazier_and_goals_sample(goals, ac_goals, obs,
                                batch_size_in_transitions):
        init = []
        init.append(goals[0])
        sel_idx_set = set([0])
        idx_set = [i for i in range(len(goals))]
        idx_set.remove(0)
        balance = 1.0
        #balance = config_cur.learning_down + config_cur.learning_rate * config_cur.learning_step / config_cur.total_learning_step
        #balance = math.pow(1 + config_cur.learning_rate, config_cur.learning_step)*config_cur.lambda_starter
        balance = math.pow(1 + config_cur.learning_rate,
                           config_cur.learning_step)
        for i in range(1, batch_size_in_transitions):
            max_dist = np.NINF  #-100.
            sel_idx = -1
            sub_size = 3
            sub_set = random.sample(idx_set, sub_size)
            for j in range(sub_size):
                ob = obs[sub_set[j]]
                gripper_pos = ob[0:3]
                object_pos = ob[3:6]
                dist = get_distance(goals[sub_set[j]], init)
                comb_dist = dist / len(init) - balance * np.linalg.norm(
                    goals[sub_set[j]] - ac_goals[sub_set[j]]
                ) - balance * np.linalg.norm(gripper_pos - object_pos)
                #comb_dist = -balance * np.linalg.norm(goals[sub_set[j]]-ac_goals[sub_set[j]])
                if comb_dist > max_dist:
                    max_dist = comb_dist
                    sel_idx = sub_set[j]
            init.append(goals[sel_idx])
            idx_set.remove(sel_idx)
            sel_idx_set.add(sel_idx)
        return np.array(list(sel_idx_set))

    # does not use it
    def get_distance(p, init_set):
        dist = 0.
        for i in range(len(init_set)):
            dist += np.linalg.norm(p - init_set[i])
        return dist

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = config_cur.learning_candidates

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {
            key: episode_batch[key][episode_idxs, t_samples].copy()
            for key in episode_batch.keys()
        }

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        #assert batch_size_in_transitions == 64
        if batch_size_in_transitions != config_cur.learning_selected:
            batch_size_in_transitions = config_cur.learning_selected

        # curriculum learning process
        transitions = curriculum(transitions, batch_size_in_transitions,
                                 batch_size)
        batch_size = batch_size_in_transitions

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {
            k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
            for k in transitions.keys()
        }

        assert (transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions
