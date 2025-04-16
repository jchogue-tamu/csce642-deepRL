import os
import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.activations import tanh, relu, softmax
from typing import List, Tuple
import keras

gdrive_path='/content/gdrive/MyDrive/Colab Notebooks/DRL/project/models'
gdrive_videos_path='/content/gdrive/MyDrive/Colab Notebooks/DRL/project/videos'
algorithm = 'a2c'

####################
# Helper Functions #
####################
def numpy_to_tensor(x: np.ndarray) -> tf.Tensor:
    """
        Converts a NumPy array to a TensorFlow tensor
    """
    return tf.convert_to_tensor(x, dtype=tf.float32)


class ActorCriticNetwork():
    """
        A combined Actor-Critic model. This model contains separate neural 
        networks for the Actor (ie. policy estimation) and the Critic
        ie. value estimation). Furthermore, it includes a Memory class for
        storing trajectories, as well as a 'train' function to update both 
        networks.
    """

    class Savable():
        def save(self, path: str):
            pass
        def load(self, path: str):
            pass

    class Actor(tf.keras.Model, Savable):
        """
            Neural network that maps states to action probs
        """
        def __init__(self, state_dim: int, n_actions: int, epsilon: float, load_model: bool=False) -> None:
            """
                Init the actor model

                params:
                - state_dim : dimension of the input state
                - n_actions : number of discrete possible actions
                - epsilon : small value added to probs for stability (avoid log(0))
                - load_model : whether to load the model from drive or not
            """
            super().__init__()

            self.state_dim = state_dim
            self.n_actions = n_actions

            if load_model:
                self.model = self.load(f"{gdrive_path}/{algorithm}")
            else:
                self.model = models.Sequential([
                    layers.Dense(64, activation=tanh),
                    layers.Dense(32, activation=tanh),
                    layers.Dense(self.n_actions, activation=softmax)
                ])

        def call(self, X: tf.Tensor) ->  tf.Tensor:
            """
                Performs a forward pass through the network

                params:
                - X : Input state tensor

                returns:
                - Tensor of action probs
            """
            return self.model(X)
        
        def compute_loss(self, log_probs: tf.Tensor, advantage: tf.Tensor, beta: float, entropies: tf.Tensor) -> tf.Tensor:
            """
                Computes the actor loss using the policy gradient with entropy regularization
                
                The computed loss encourages actions that result in larger advantages and promotes
                exploration by maximizing entropy
                
                Advantage is treated as a fixed value to prevent the actor from influencing the critic
                learning

                params:
                - log_probs : Log probabilities of the selected actions
                - advantage : Tensor containing the advantage values (Q(s, a) - V(s))
                - beta : Entropy regularization value
                - entropies : Entropy of the action distributions

                returns:
                - Tensor representing the total actor loss
            """

            return -tf.reduce_mean(log_probs * tf.stop_gradient(advantage)) - beta * tf.reduce_mean(entropies)

        def save(self, path: str):
            """
                Saves the actor model at the given 'path'

                params:
                - path : The path to save the model
            """
            full_path = f"{path}/actor.keras"
            # os.makedirs(os.path.dirname(full_path), exist_ok=True)
            # self.model.save(full_path)

        def load(self, path: str) -> keras.Model:
            """
                Loads the (keras) model from the specified path

                params:
                - path : The path to load the model from

                returns:
                - The loaded keras model
            """
            return keras.models.load_model(f"{path}/actor.keras")
        
        def choose_action(self, state: tf.Tensor) -> tuple[int, tf.Tensor]:
            """
                Chooses an action based on the policy distribution predicted by 
                the actor model

                params:
                - state : The current state of the environment

                returns:
                - action : An integer index representing the selected action
                - probs  : The prob distribution over actions
            """
            # state = np.expand_dims(state, axis=0)
            # prob = self.model.predict_on_batch(state).flatten()
            # action = np.random.choice(self.n_actions, p=prob)

            # convert state to tensor and get action probs
            state_tensor = tf.expand_dims(state, axis=0)
            probs = tf.squeeze(self.call(state_tensor))

            # select an action from the categorical distribution
            action_dist = tf.random.categorical(tf.math.log([probs]), 1)[0, 0]
            action = int(action_dist.numpy())

            return action, probs

    class Critic(tf.keras.Model, Savable):
        """
            Neural network that maps states to state-value estimates
        """
        def __init__(self, state_dim: int, load_model: bool=False) -> None:
            """
                Init the critic model

                params:
                - state_dim : dimension of the input state
                - load_model : whether to load the model from drive or not
            """
            super().__init__()

            if load_model:
                self.model = self.load(f"{gdrive_path}/{algorithm}")
            else:
                self.model = models.Sequential([
                    layers.Dense(64, activation=relu),
                    layers.Dense(32, activation=relu),
                    layers.Dense(1)
                ])

        def call(self, X: tf.Tensor) ->  tf.Tensor:
            """
                Performs a forward pass through the network

                params:
                - X : Input state tensor

                returns:
                - Tensor containing the estimated state value
            """
            return self.model(X)
        
        def compute_loss(self, advantage: tf.Tensor) -> tf.Tensor:
            """
                Computes the critic loss as the MSE of the advantage
                
                Advantage is the diff between the estimated return (Q-val)
                and the predicted value (V(s)) from the critic. 
                
                This function implements standard MSE loss, which is used 
                to train the critic to more accurately estimate V(s).

                params:
                - advantage : Tensor containing the advantage values (Q(s, a) - V(s))

                returns:
                - Tensor representing the MSE loss
            """
            return tf.reduce_mean(tf.square(advantage))

        def save(self, path: str):
            """
                Saves the critic model at the given 'path'

                params:
                - path : The path to save the model
            """
            full_path = f"{path}/critic.keras"
            # os.makedirs(os.path.dirname(full_path), exist_ok=True)
            # self.model.save(full_path)

        def load(self, path: str) -> keras.Model:
            """
                Loads the (keras) model from the specified path

                params:
                - path : The path to load the model from

                returns:
                - The loaded keras model
            """
            return keras.models.load_model(f"{path}/critic.keras")
            

    class Memory():
        """
            Stores the trajectory (states, actions, rewards, dones, entropies)
            collected during one episode for later training.
        """

        def __init__(self) -> None:
            """
                Inits memory buffers for storing transitions.
            """

            self.states = []
            self.actions = []
            self.rewards = []
            self.dones = []
            self.entropies = []

        def add(self, state: np.ndarray, action: int, reward: float, done: bool, entropy: tf.Tensor) -> None:
            """
                Adds a transition to memory buffer

                params:
                - state : The observed state 
                - action : The action taken
                - reward : The reward received
                - done : Whether the episode terminated or not
                - entropy : The entropy of the action distribution
            """

            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
            self.entropies.append(entropy)

        def clear(self) -> None:
            """
                Clears all of the stored transitions
            """

            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.dones.clear()
            self.entropies.clear()

        def _zip(self) -> zip:
            """
                Zip the stored transitions to be iterated over

                returns:
                - zip : zipped iterator for the stored memory
            """

            return zip(self.states, self.actions, self.rewards, self.dones, self.entropies)

        def reversed(self) -> List[Tuple[np.ndarray, int, float, bool, tf.Tensor]]:
            """
                Reverses the stored transitions and returns it

                returns:
                - List of tuples representing reversed memory buffer
            """
            return list(self._zip())[::-1]

        def __len__(self) -> int:
            """
                returns:
                - The number of transitions stored
            """

            return len(self.rewards)

    def __init__(self, state_dim: int, n_actions: int, lr: float = 1e-3, gamma: float = 0.99, beta: float = 1e-2, epsilon: float = 1e-10, max_grad_norm: float = 1.0, load_models: bool=False) -> None:
        """
            Init the Actor-Critic network and optimizers

            params:
            - state_dim : Dimension of the input state space
            - n_actions : Number of possible actions
            - lr : Learning rate
            - gamma : Discount factor
            - beta : Entropy regularization value
            - epsilon : Small value added to probs for stability (to avoid log(0))
            - max_grad_norm : max allowed value for gradients;
                              gradients larger than this are clipped to prevent
                              instability
            - load_models : whether to load the previously saved models or not
        """

        # save the hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm

        # init actor and critic networks
        self.actor = self.Actor(state_dim, n_actions, epsilon, load_model=load_models)
        self.critic = self.Critic(state_dim, load_model=load_models)

        # init Adam optimizers for both networks
        self.actor_optimizer = optimizers.Adam(learning_rate=lr)
        self.critic_optimizer = optimizers.Adam(learning_rate=lr)

        # init memory buffer for storing episode transitions
        self.memory = self.Memory()

    def choose_action(self, state: tf.Tensor) -> tuple[int, tf.Tensor]:
        """
            Chooses an action using the actor model given the current state

            params:
            - state : The current state of the environment

            returns:
            - Tuple of action, probs
        """
        return self.actor.choose_action(state)

    def train(self, q_val: float) -> Tuple[tf.Tensor, tf.Tensor]:
        """
            Trains the actor and critic networks using stored memory

            params:
            - q_val : Final estimated value for the last state

            returns:
            - Tuple of actor and critic loss (for logging/debugging)
        """

        # convert stored memory to tensors
        states = tf.convert_to_tensor(self.memory.states, dtype=tf.float32)
        actions = tf.convert_to_tensor(self.memory.actions, dtype=tf.int32)
        entropies = tf.stack(self.memory.entropies)
        q_vals = np.zeros((len(self.memory), 1))

        # compute discounted returns in reverse
        for i, (_, _, reward, done, _) in enumerate(self.memory.reversed()):
            q_val = reward + self.gamma * q_val * (1.0 - done)
            q_vals[len(self.memory) - 1 - i] = q_val

        q_vals = tf.convert_to_tensor(q_vals, dtype=tf.float32)

        #################
        # Critic update #
        #################
        with tf.GradientTape() as tape:
            # get current value estimates
            values = tf.squeeze(self.critic(states), axis=1)
            # compute advantage
            advantage = tf.squeeze(q_vals, axis=1) - values
            # compute MSE loss
            critic_loss = self.critic.compute_loss(advantage)
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        # apply gradient clipping to improve stability
        grads = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads]
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        ################
        # Actor update #
        ################
        with tf.GradientTape() as tape:
            probs = self.actor(states)
            # one-hot encode the actions
            action_masks = tf.one_hot(actions, depth=probs.shape[1])
            # compute log probabilities of actions for input states (using log likelihood)
            log_probs = tf.reduce_sum(action_masks * tf.math.log(probs + self.epsilon), axis=1)
            # compute policy loss
            actor_loss = self.actor.compute_loss(log_probs, advantage, self.beta, entropies)
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        # apply gradient clipping to improve stability
        grads = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads]
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        return actor_loss, critic_loss

    def evaluate(self, env: gym.Env, max_steps: int) -> float:
        """
            Evaluates the performance of an Actor-Critic agent in the given 
            Gym environment.

            parms:
            - env : The Gym environment to evaluate in
            - max_steps : Max number of steps to run in the episode

            returns:
            - episode_reward : Total accumulated reward during the episode.=
        """
        state, _ = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            action, _ = self.choose_action(state)
            state, reward, done, truncated, _ = env.step(action)
            finished = done or truncated
            episode_reward += reward

            if finished:
                break

        return episode_reward
    
    def save_models(self) -> None:
        """
            Saves the actor and critic models to GDrive

            Depends on global variables 'gdrive_path' and 'algorithm'
        """
        self.actor.save(f"{gdrive_path}/{algorithm}")
        self.critic.save(f"{gdrive_path}/{algorithm}")


def main():
    # init the LunarLander environment
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    # env, folder_name = create_environment('LunarLander-v3')
    # print(f'Training video directory name: {folder_name}\n')
    env_test = gym.make("LunarLander-v3")

    

    # get env config
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    max_steps_per_episode = env.spec.max_episode_steps

    # define hyperparameters for tuning
    n_episodes = 10
    # n_episodes = 1000
    n_steps = 200
    learning_rate = 1e-3
    gamma = 0.99
    beta = 1e-3
    epsilon = 1e-10
    max_grad_norm = 0.1

    load_models = False

    # init actor-critic network
    acn = ActorCriticNetwork(state_dim,
                            n_actions,
                            lr=learning_rate,
                            gamma=gamma,
                            beta=beta,
                            epsilon=epsilon,
                            max_grad_norm=max_grad_norm,
                            load_models=load_models)

    best_reward = float('-inf')
    episode_rewards = []
    test_rewards = []

    for episode in range(n_episodes):
        done = False
        trunc = False
        total_reward = 0
        state, _ = env.reset()
        steps = 0

        while not done and not trunc and steps < max_steps_per_episode:
            actor_loss = 0.0
            critic_loss = 0.0

            # convert state to tensor and get action probs
            state_tensor = numpy_to_tensor(state)
            # select an action from the categorical distribution
            action, probs = acn.choose_action(state_tensor)

            # compute entropy of the distribution
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + acn.epsilon))

            # perform action in env
            next_state, reward, done, trunc, _ = env.step(action)

            total_reward += reward
            steps += 1

            # store transition
            acn.memory.add(state, action, reward, done, entropy)

            # move to next state
            state = next_state

            # only train after fixed number of steps or end of an episode
            if done or trunc or (steps % n_steps == 0):
                next_value = tf.squeeze(acn.critic(tf.expand_dims(numpy_to_tensor(next_state), axis=0))).numpy()
                actor_loss, critic_loss = acn.train(next_value)
                acn.memory.clear()

        episode_rewards.append(total_reward)

        # Evaluate and store rewards
        test_total_reward = acn.evaluate(env_test, max_steps_per_episode)
        test_rewards.append(test_total_reward)

        # Save best models
        # if total_reward > best_reward:
        #     best_reward = total_reward
        #     print(f"New best reward: {best_reward:.2f}! Saving models...")
        #     acn.save_models()
        if test_total_reward > best_reward:
            best_reward = test_total_reward
            print(f"New best reward: {best_reward:.2f}! Saving models...")
            acn.save_models()

        print(
            f"Episode {episode + 1}/{n_episodes}, Reward: {total_reward:.2f}, Best Reward: {np.max(test_rewards):.2f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}"
        )

if __name__ == "__main__":
    main()