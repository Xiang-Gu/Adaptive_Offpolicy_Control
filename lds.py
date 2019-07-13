import numpy as np
from distributions import Distribution


class LDS(Distribution):
    """
    Describes trajectory distribution of agent moving in a plane.

    Simple continuous action MDP:
    State is agent position and velocity and actions are acceleration.
    Actions are sampled from policies of the form: k * x + N(0, std) where
    x is the state and k is a constant matrix.
    The state evolves according to linear dynamics with additive Gaussian noise.
    The agent is rewarded according to how close it is to a goal position and
    each trajectory lasts for a fixed number, L, time-steps.
    The state is bounded in the [-10, 10] plane.
    """
    def __init__(self,
                 dimension,
                 K,
                 sigma,
                 L=20):

        # Environment constants
        # action = x * K + N(0, std)
        # state = A * x + B * u + N(0, 0.05)
        self._A = np.eye(2 * dimension)
        self._B = np.vstack([0.5 * np.eye(dimension), np.eye(dimension)])
        self._L = L
        # goal is point (5, 5)
        self._goal = np.zeros(2 * dimension)
        for i in range(dimension):
            self._goal[i] = 5

        # Agent policy
        self._K = np.array(K, copy=True)
        self._sigma = sigma #* np.ones(dimension, dtype=float)

        # Class private member variables
        self._dimension = dimension

    def sample(self, n=1):
        """Generate n trajectories from environment MDP."""
        paths = []
        gs = np.zeros(n)
        for i in range(n):
            state = np.zeros(2 * self._dimension)
            x = np.zeros((self._L, 2 * self._dimension))
            u = np.zeros((self._L, self._dimension))
            r = np.zeros(self._L)
            pi_noise = np.random.normal(0, self._sigma,
                                        (self._L, self._dimension))
            noise = np.random.normal(0, 0.05, (self._L, self._dimension * 2))
            # noise = np.zeros((self._L, self._dimension * 2))
            for t in range(self._L):
                control = self.mean(state) + pi_noise[t]
                x[t] = state
                u[t] = control
                state = self._A.dot(state) + self._B.dot(control) + noise[t]
                state = np.clip(state, -10, 10)
                r[t] = -1 * np.linalg.norm(state - self._goal)
            # Reward is how close you are to origin at end of trajectory
            # r = -100 * np.linalg.norm(state - self._goal)
            g = np.sum(r)
            gs[i] = g
            paths.append({'x': x, 'u': u, 'r': r})
        return paths, gs

    def expected_value(self):
        """Evaluate expected value of MDP and policy with Monte Carlo eval."""
        _, fs = self.sample(n=100000)
        return np.mean(fs)

    def mean(self, x):
        """Get mean of policy function for state x."""
        mean = x.dot(self._K)
        return mean.flatten()

    def _action_log_likelihood(self, x, u):
        """Log probability of action u in state x."""
        # y = self._input_fn(x)
        # mean = y.dot(self._K)
        mean = x.dot(self._K)
        log_std = np.log(self._sigma)
        zs = (u - mean) / np.exp(log_std)
        lh = - np.sum(log_std) - \
            0.5 * np.sum(np.square(zs)) - \
            0.5 * self._dimension * np.log(2 * np.pi)
        return lh

    def pdf(self, path):
        """Joint robability of actions along a trajectory."""
        xs = path['x']
        us = path['u']
        ll = [self._action_log_likelihood(x, u) for x, u in zip(xs, us)]
        return np.exp(np.sum(ll))

    def derivative_path_log_likelihood_wrt_K(self, path):
        """Sum of derivative of log pdf of action u given state x w.r.t K for all action-state pair in path"""
        xs = path['x']
        us = path['u']
        dallwK = [self.derivative_action_log_likelihood_wrt_K(x, u) for x, u in zip(xs, us)]
        return np.sum(dallwK, axis = 0)

    def derivative_path_log_likelihood_wrt_sigma(self, path):
        """Sum of derivative of log pdf of action u given state x w.r.t sigma for all action-state pair in path"""
        xs = path['x']
        us = path['u']
        # dallws is a list of arrays where each array is the derivative of log pdf of action u_i given state x_i w.r.t sigma
        # each array in dallws has two components where each of them is the partial derivative
        dallws = [self.derivative_action_log_likelihood_wrt_sigma(x, u) for x, u in zip(xs, us)]
        result = np.sum(dallws, axis = 0)

        return result

    def derivative_action_log_likelihood_wrt_K(self, x, u):
        """Derivative of log pdf of action u given state x w.r.t. K"""
        """This function is the derivative of self._action_log_likelihood w.r.t. K"""

        result = np.zeros(shape = (2 * self._dimension, self._dimension))
        result = (1 / self._sigma[0] ** 2) * (x.reshape(2 * self._dimension, 1) * (u - x.dot(self._K)))
        return result

    def derivative_action_log_likelihood_wrt_sigma(self, x, u):
        """Derivative of log pdf of action u given state x w.r.t sigma"""
        """This function is the derivative of self._action_log_likelihood w.r.t sigma"""
        mean = x.dot(self._K)
        result = np.array([-1 / self._sigma[i] + (u[i] - mean[i]) ** 2 / self._sigma[i] ** 3 for i in range(self._dimension)])
        return result

    def BPG(self):
    	max_iterations = 10
        max_trials = 1
        batch_size = 1
        alpha_K = 0.00000001
        alpha_sigma = 0.00000001

        # Store all thetas for sampling distribution history
        # Notice here we store only one sampling thetas for each trial 
        # since what we care about is the final one
        sampling_K = np.zeros(shape = (max_trials, self._K.shape[0], self._K.shape[1]))
        sampling_K[:] = self._K
        sampling_sigma = np.zeros(shape = (max_trials, self._dimension) )
        sampling_sigma[:] = self._sigma

        # initialize sampling distribution to the target distribution
        sampling_dist = LDS(self._dimension, self._K, self._sigma, self._L)

        for trial in range(max_trials):
            # Reset sampling distribution to target distribution
            sampling_dist = LDS(self._dimension, self._K, self._sigma, self._L)

            for iteration in range(max_iterations):
                # Generate a batch of samples from sampling distribution
                current_batch_IS_paths, current_batch_IS_rewards = sampling_dist.sample(n = batch_size)
                weights_current_batch_IS_paths = np.array([self.pdf(current_batch_IS_paths[i]) / sampling_dist.pdf(current_batch_IS_paths[i]) for i in range(batch_size)])

                gradient_wrt_K = 0.0
                gradient_wrt_sigma = 0.0
                for i in range(batch_size):
                    gradient_wrt_K = gradient_wrt_K + (weights_current_batch_IS_paths[i] * current_batch_IS_rewards[i]) ** 2 * sampling_dist.derivative_path_log_likelihood_wrt_K(current_batch_IS_paths[i])
                    gradient_wrt_sigma = gradient_wrt_sigma + (weights_current_batch_IS_paths[i] * current_batch_IS_rewards[i]) ** 2 * sampling_dist.derivative_path_log_likelihood_wrt_sigma(current_batch_IS_paths[i])
                gradient_wrt_K = gradient_wrt_K / batch_size
                gradient_wrt_sigma = gradient_wrt_sigma / batch_size

                # Update thetas of sampling distribution in the (opposite) direction of the gradient
                # The gradient variable, computed above, is already the negative of the true gradient
                sampling_K[trial] = sampling_K[trial] + alpha_K * gradient_wrt_K
                sampling_sigma[trial] = sampling_sigma[trial] + alpha_sigma * gradient_wrt_sigma

                # Update sampling distribution according to the new thetas
                sampling_dist = LDS(self._dimension, sampling_K[trial], sampling_sigma[trial], self._L)

        # Average the final thetas of sampling distribution (the last element in the thetas array for every trial) over all trials
        optimal_K = np.mean(sampling_K, axis = 0)
        optimal_sigma = np.mean(sampling_sigma, axis = 0)

        return LDS(self._dimension, optimal_K, optimal_sigma, self._L)