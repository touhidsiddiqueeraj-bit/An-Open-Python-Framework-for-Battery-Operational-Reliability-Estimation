import numpy as np
from collections import defaultdict


class RLThresholdPolicy:
    """Learnable dispatch threshold tuned via Q-learning.

    The policy learns an optimal risk threshold τ(t) as a function of
    the battery state (SOH, cycle, recent failure pattern) using a
    tabular Q-learning approach.

    State: discretized (soh_bin, cycle_bin, recent_failures)
    Action: choose τ from a discrete set
    Reward: delivered_energy - penalty × failure_event
    """

    def __init__(self, tau_choices=None, penalty=500.0,
                 lr=0.1, gamma=0.9, epsilon=0.1):
        self.tau_choices = tau_choices or [0.05, 0.10, 0.15, 0.20,
                                           0.30, 0.40, 0.50]
        self.penalty = penalty
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(len(self.tau_choices)))
        self.n_actions = len(self.tau_choices)

    def _discretize(self, soh, cycle, recent_failures):
        soh_bin = min(int(soh * 10), 9)
        cycle_bin = min(cycle // 50, 7)
        fail_bin = min(recent_failures, 3)
        return (soh_bin, cycle_bin, fail_bin)

    def choose_action(self, soh, cycle, recent_failures, eval_mode=False):
        state = self._discretize(soh, cycle, recent_failures)
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, soh, cycle, recent_failures, action,
               reward, next_soh, next_cycle, next_failures,
               done=False):
        state = self._discretize(soh, cycle, recent_failures)
        next_state = self._discretize(next_soh, next_cycle, next_failures)
        best_next = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next * (1 - done)
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.lr * td_error

    def get_tau(self, soh, cycle, recent_failures):
        action = self.choose_action(soh, cycle, recent_failures, eval_mode=True)
        return self.tau_choices[action]

    def train(self, trajectories, n_episodes=100):
        """Train on pre-collected trajectories.

        trajectories: list of (state, action, reward, next_state, done)
        """
        for _ in range(n_episodes):
            for traj in trajectories:
                self.update(*traj)

    @property
    def name(self):
        return "RL-AdaptiveThreshold"
