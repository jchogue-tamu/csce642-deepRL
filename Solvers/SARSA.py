# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).

from collections import defaultdict
import numpy as np
from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


class Sarsa(AbstractSolver):
    def __init__(self, env, eval_env, options):
        assert str(env.observation_space).startswith("Discrete"), (
            str(self) + " cannot handle non-discrete state spaces"
        )
        assert str(env.action_space).startswith("Discrete") or str(
            env.action_space
        ).startswith("Tuple(Discrete"), (
            str(self) + " cannot handle non-discrete action spaces"
        )
        super().__init__(env, eval_env, options)
        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def train_episode(self):
        """
        Run one episode of the SARSA algorithm: On-policy TD control.

        Use:
            self.env: OpenAI environment.
            self.epsilon_greedy_action(state): returns an epsilon greedy action
            self.options.steps: number of steps per episode
            self.options.gamma: Gamma discount factor.
            self.options.alpha: TD learning rate.
            self.Q[state][action]: q value for ('state', 'action')
            self.options.epsilon: Chance the sample a random action. Float betwen 0 and 1.

        """

        # Reset the environment
        state, _ = self.env.reset()

        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################

        # choose an epsilon-greedy action
        action_probs = self.epsilon_greedy_action(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        # Iterate over each step in the episode
        for _ in np.arange(self.options.steps):
            # take a step
            next_state, reward, done, _ = self.step(action)

            # choose A' from S' using (epsilon-greedy) policy derived from Q
            next_action_probs = self.epsilon_greedy_action(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            # update the current state's Q-values
            self.Q[state][action] += self.options.alpha * (reward + self.options.gamma * self.Q[next_state][next_action] - self.Q[state][action])

            # update current state and action to point to the next ones
            state = next_state
            action = next_action

            # bail if this is a terminal state
            if done:
                # print('reached terminal state. breaking from the loop.')
                break

    def __str__(self):
        return "Sarsa"

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.

        Returns:
            A function that takes a state as input and returns a greedy action.
        """

        def policy_fn(state):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            return np.argmax(self.Q[state])

        return policy_fn

    def epsilon_greedy_action(self, state):
        """
        Return an epsilon-greedy action based on the current Q-values and
        epsilon.

        Use:
            self.env.action_space.n: the size of the action space
            np.argmax(self.Q[state]): action with highest q value
        Returns:
            Probability of taking actions as a vector where each entry is the probability of taking that action
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        nA = self.env.action_space.n

        # initialize vector of action probabilities
        #   multiply by epsilon and divide that by the size of the action space
        A = (np.ones(nA, dtype=float) * self.options.epsilon) / nA

        # get the best action value from this state's Q-values. 
        #   argmax will arbitrarily break ties for us
        best_action_val = np.argmax(self.Q[state])

        # compute the greedy action probability ((1 - eps) + (eps / nA))
        A[best_action_val] = (1 - self.options.epsilon) + A[best_action_val]

        return A
        

    def plot(self, stats, smoothing_window=20, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)
