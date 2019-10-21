import threading

import numpy as np

from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree

import math

from scipy.stats import rankdata

import json

def quaternion_to_euler_angle(array):
    w = array[0]
    x = array[1]
    y = array[2]
    z = array[3]
    ysqr = y * y
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.atan2(t3, t4)
    result = np.array([X, Y, Z])
    return result

class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        transitions = self.sample_transitions(buffers, batch_size)

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx


class ReplayBufferEnergy:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions, prioritization, env_name):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions

        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}
        self.buffers['e'] = np.empty([self.size, 1]) # energy
        self.buffers['p'] = np.empty([self.size, 1]) # priority/ranking

        self.prioritization = prioritization
        self.env_name = env_name

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        self.current_size_test = 0
        self.n_transitions_stored_test = 0

        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size, rank_method, temperature):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        transitions = self.sample_transitions(buffers, batch_size, rank_method, temperature)

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            if not key == 'p' and not key == 'e':
                assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def store_episode(self, episode_batch, w_potential, w_linear, w_rotational, rank_method, clip_energy):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        buffers = {}
        for key in episode_batch.keys():
            buffers[key] = episode_batch[key]

        if self.prioritization == 'energy':
            if self.env_name in ["FetchReach-v1", 'FetchPickAndPlace-v0', 'FetchSlide-v0', 'FetchPush-v0']:
                height = buffers['ag'][:, :, 2]
                height_0 = np.repeat(height[:,0].reshape(-1,1), height[:,1::].shape[1], axis=1)
                height = height[:,1::] - height_0
                g, m, delta_t = 9.81, 1, 0.04
                potential_energy = g*m*height
                diff = np.diff(buffers['ag'], axis=1)
                velocity = diff / delta_t
                kinetic_energy = 0.5 * m * np.power(velocity, 2)
                kinetic_energy = np.sum(kinetic_energy, axis=2)
                energy_totoal = w_potential*potential_energy + w_linear*kinetic_energy
                energy_diff = np.diff(energy_totoal, axis=1)
                energy_transition = energy_totoal.copy()
                energy_transition[:,1::] = energy_diff.copy()
                energy_transition = np.clip(energy_transition, 0, clip_energy)
                energy_transition_total = np.sum(energy_transition, axis=1)
                episode_batch['e'] = energy_transition_total.reshape(-1,1)
            elif self.env_name in ["HandReach-v0", "HandManipulateBlockRotateZ-v0" , "HandManipulateBlockRotateXYZ-v0", 'HandManipulatePenRotate-v0', \
                                   'HandManipulateEggFull-v0', \
                                   'HandManipulateBlockFull-v0', \
                                   'HandManipulateBlockRotateXYZ-v0']:
                g, m, delta_t, inertia  = 9.81, 1, 0.04, 1
                quaternion = buffers['ag'][:,:,3:].copy()
                angle = np.apply_along_axis(quaternion_to_euler_angle, 2, quaternion)
                diff_angle = np.diff(angle, axis=1)
                angular_velocity = diff_angle / delta_t
                rotational_energy = 0.5 * inertia * np.power(angular_velocity, 2)
                rotational_energy = np.sum(rotational_energy, axis=2)
                buffers['ag'] = buffers['ag'][:,:,:3]
                height = buffers['ag'][:, :, 2]
                height_0 = np.repeat(height[:,0].reshape(-1,1), height[:,1::].shape[1], axis=1)
                height = height[:,1::] - height_0
                potential_energy = g*m*height
                diff = np.diff(buffers['ag'], axis=1)
                velocity = diff / delta_t
                kinetic_energy = 0.5 * m * np.power(velocity, 2)
                kinetic_energy = np.sum(kinetic_energy, axis=2)
                energy_totoal = w_potential*potential_energy + w_linear*kinetic_energy + w_rotational*rotational_energy
                energy_diff = np.diff(energy_totoal, axis=1)
                energy_transition = energy_totoal.copy()
                energy_transition[:,1::] = energy_diff.copy()
                energy_transition = np.clip(energy_transition, 0, clip_energy)
                energy_transition_total = np.sum(energy_transition, axis=1)
                episode_batch['e'] = energy_transition_total.reshape(-1,1)
            else:
                print('Trajectory Energy Function Not Implemented')
                exit()

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys():
                if not key == 'p':
                    self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T

            energy_transition_total = self.buffers['e'][:self.current_size]
            if rank_method == 'none':
                rank_method = 'dense'
            energy_rank = rankdata(energy_transition_total, method=rank_method)
            energy_rank = energy_rank - 1
            energy_rank = energy_rank.reshape(-1, 1)
            self.buffers['p'][:self.current_size] = energy_rank.copy()

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions, alpha, env_name):
        """Create Prioritized Replay buffer.
        """
        super(PrioritizedReplayBuffer, self).__init__(buffer_shapes, size_in_transitions, T, sample_transitions)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        self.size_in_transitions = size_in_transitions
        while it_capacity < size_in_transitions:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

        self.T = T
        self.buffers['td'] = np.zeros([self.size, self.T]) # accumulated td-error
        self.buffers['e'] = np.zeros([self.size, self.T]) # trajectory energy
        self.env_name = env_name

    def store_episode(self, episode_batch, dump_buffer):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """

        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        if dump_buffer:

            buffers = {}
            for key in episode_batch.keys():
                buffers[key] = episode_batch[key]

            if self.env_name in ['FetchPickAndPlace-v0', 'FetchSlide-v0', 'FetchPush-v0']:
                height = buffers['ag'][:, :, 2]
                height_0 = np.repeat(height[:,0].reshape(-1,1), height[:,1::].shape[1], axis=1)
                height = height[:,1::] - height_0
                g, m, delta_t = 9.81, 1, 0.04
                potential_energy = g*m*height
                diff = np.diff(buffers['ag'], axis=1)
                velocity = diff / delta_t
                kinetic_energy = 0.5 * m * np.power(velocity, 2)
                kinetic_energy = np.sum(kinetic_energy, axis=2)
                energy_totoal = potential_energy + kinetic_energy
                energy_diff = np.diff(energy_totoal, axis=1)
                energy_transition = energy_totoal.copy()
                energy_transition[:,1::] = energy_diff.copy()
                episode_batch['e'] = energy_transition
            elif self.env_name in ['HandManipulatePenRotate-v0', \
                                   'HandManipulateEggFull-v0', \
                                   'HandManipulateBlockFull-v0', \
                                   'HandManipulateBlockRotateXYZ-v0']:
                g, m, delta_t, inertia  = 9.81, 1, 0.04, 1
                quaternion = buffers['ag'][:,:,3:].copy()
                angle = np.apply_along_axis(quaternion_to_euler_angle, 2, quaternion)
                diff_angle = np.diff(angle, axis=1)
                angular_velocity = diff_angle / delta_t
                rotational_energy = 0.5 * inertia * np.power(angular_velocity, 2)
                rotational_energy = np.sum(rotational_energy, axis=2)
                buffers['ag'] = buffers['ag'][:,:,:3]
                height = buffers['ag'][:, :, 2]
                height_0 = np.repeat(height[:,0].reshape(-1,1), height[:,1::].shape[1], axis=1)
                height = height[:,1::] - height_0
                potential_energy = g*m*height
                diff = np.diff(buffers['ag'], axis=1)
                velocity = diff / delta_t
                kinetic_energy = 0.5 * m * np.power(velocity, 2)
                kinetic_energy = np.sum(kinetic_energy, axis=2)
                energy_totoal = potential_energy + kinetic_energy + rotational_energy
                energy_diff = np.diff(energy_totoal, axis=1)
                energy_transition = energy_totoal.copy()
                energy_transition[:,1::] = energy_diff.copy()
                episode_batch['e'] = energy_transition


        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys():
                if not key == 'td':
                    if dump_buffer:
                        self.buffers[key][idxs] = episode_batch[key]
                    else:
                        if not key == 'e':
                            self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T

            for idx in idxs:
                episode_idx = idx
                for t in range(episode_idx*self.T, (episode_idx+1)*self.T):
                    assert (episode_idx+1)*self.T-1 < min(self.n_transitions_stored, self.size_in_transitions)
                    self._it_sum[t] = self._max_priority ** self._alpha
                    self._it_min[t] = self._max_priority ** self._alpha

    def dump_buffer(self, epoch):
        for i in range(self.current_size):
            entry = {"e": self.buffers['e'][i].tolist(), \
                     "td": self.buffers['td'][i].tolist(), \
                     "ag": self.buffers['ag'][i].tolist() }
            with open('buffer_epoch_{0}.txt'.format(epoch), 'a') as file:
                 file.write(json.dumps(entry))  # use `json.loads` to do the reverse
                 file.write("\n")

        print("dump buffer")


    def sample(self, batch_size, beta):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """

        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        transitions, weights, idxs = self.sample_transitions(self, buffers, batch_size, beta)

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            if not key == 'td' and not key == 'e':
                assert key in transitions, "key %s missing from transitions" % key

        return (transitions, weights, idxs)


    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities.flatten()):
            assert priority > 0
            assert 0 <= idx < self.n_transitions_stored
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
