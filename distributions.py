import numpy as np
import scipy.stats
import math

# A distribution parameterized by thetas
class Distribution(object):

    def __init__(self, thetas):
        pass

    def sample(self):
        pass

    def expected_value(self):
        pass

    def pdf(self, x):
        pass

    # Return derivative of logrithm of pdf of sample -- d(ln(pdf(sample))) / d(thetas)
    # Return is an 1D-numpy.array containing thetas.shape[0] elements. 
    # Each element (at index i) is the partial derivative of score function w.r.t. thetas[i]
    def derivative_score_function(self, sample):
        pass

    # Return derivative of logrithm of pdf of a batch of samples -- d(ln(pdf(D))) / d(thetas)
    # It equals to the sum of derivative of logrithm of pdf of each sample
    def derivative_score_function_batch(self, batch_samples):
        pass

    # Return the derivative of WIS of a batch of samples w.r.t. thetas -- d(WIS(D)) / d(thetas)
    def derivative_WIS_batch_wrt_thetas(self, batch_samples, weights_batch_samples):
    	pass

    # Return the optimal sampling distribution using gradient of objective
    # which is definded as the variance of the IS estimate part of the estimator of policy gradient
    def BPG(self):
        pass

    # Return the optimal sampling distribution using gradient of another objective
    # which is defined as the sum over all components of the variance of the estimator of policy gradient
    def SUM(self, performance_function):
        pass

# Normal distribution parameterized by thetas
class Normal(Distribution):

    # thetas[0] is the mean, and thetas[1] is the standard deviation
    def __init__(self, thetas):
        assert thetas.shape == (2,)
        self.mean = thetas[0]
        self.std = thetas[1]
        self.thetas = thetas

    def sample(self, n=1):
        return np.random.normal(self.mean, self.std, n)

    def expected_value(self):
        return self.mean

    def pdf(self, sample):
        return scipy.stats.norm.pdf(sample, self.mean, self.std)

    def derivative_score_function(self, sample):
        # See handwriting for derivation detail
        return np.array([(sample - self.thetas[0]) / self.thetas[1] ** 2, -1 / self.thetas[1] + (sample - self.thetas[0]) ** 2 / self.thetas[1] ** 3])

    def derivative_score_function_batch(self, batch_samples):
        # See handwrting for derivation detail
        result = np.array([self.derivative_score_function(batch_samples[i]) for i in range(batch_samples.size)])
        return np.sum(result, axis = 0)

    def derivative_WIS_batch_wrt_thetas(self, batch_samples, weights_batch_samples):
	    # See handwriting for derivation detail
	    result = np.zeros(self.thetas.size)
	    # Compute numerator
	    for j in range(batch_samples.size):
	        for i in range(batch_samples.size):
	            result = result + weights_batch_samples[j] * weights_batch_samples[i] * (batch_samples[j] - batch_samples[i]) * self.derivative_score_function(batch_samples[i])
	    # Divide result by denominator
	    result = result / (np.sum(weights_batch_samples) ** 2)

	    return result

    def BPG(self):
        max_iterations = 10
        max_trials = 10
        batch_size = 20
        alpha = 0.00001

        # Store all thetas for sampling distribution history
        # Notice here we store only one sampling thetas for each trial 
        # since what we care about is the final one
        sampling_thetas = np.zeros(shape = (max_trials, self.thetas.size))
        sampling_thetas[:] = self.thetas
        # initialize sampling distribution to the target distribution
        sampling_dist = Normal(self.thetas)

        for trial in range(max_trials):
            # Reset sampling distribution to target distribution
            sampling_dist = Normal(self.thetas)

            for iteration in range(max_iterations):
                # Generate a batch of samples from sampling distribution
                current_batch_IS_samples = sampling_dist.sample(n = batch_size)
                weights_current_batch_IS_samples = np.array([self.pdf(current_batch_IS_samples[i]) / sampling_dist.pdf(current_batch_IS_samples[i]) for i in range(batch_size)])

                gradient = 0.0
                for i in range(batch_size):
                    gradient = gradient + (weights_current_batch_IS_samples[i] * current_batch_IS_samples[i]) ** 2 * sampling_dist.derivative_score_function(current_batch_IS_samples[i])
                gradient = gradient / batch_size

                # Update thetas of sampling distribution in the (opposite) direction of the gradient
                # The gradient variable, computed above, is already the negative of the true gradient
                sampling_thetas[trial] = sampling_thetas[trial] + alpha * gradient

                # Update sampling distribution according to the new thetas
                sampling_dist = Normal(sampling_thetas[trial])

        # Average the final thetas of sampling distribution (the last element in the thetas array for every trial) over all trials
        optimal_thetas = np.mean(sampling_thetas, axis = 0)

        return Normal(optimal_thetas)

    def SUM(self, performance_function):
        max_iterations = 10
        max_trials = 5
        batch_size = 10
        alpha = 0.0001

        # Store all thetas for sampling distribution history
        # Notice here we store only one sampling thetas for each trial 
        # since what we care about is the final one
        sampling_thetas = np.zeros(shape = (max_trials, self.thetas.size))
        sampling_thetas[:] = self.thetas
        # initialize sampling distribution to the target distribution
        sampling_dist = Normal(self.thetas)

        for trial in range(max_trials):
            # Reset sampling distribution to target distribution
            sampling_dist = Normal(self.thetas)

            for iteration in range(max_iterations):
                # Generate a batch of samples from sampling distribution
                current_batch_samples = sampling_dist.sample(n = batch_size)
                weights_current_batch_samples = np.array([self.pdf(current_batch_samples[i]) / sampling_dist.pdf(current_batch_samples[i]) for i in range(batch_size)])

                gradient = 0.0
                for i in range(batch_size):
                    gradient = gradient + np.sum((weights_current_batch_samples[i] * performance_function(current_batch_samples[i], 'Normal') * self.derivative_score_function(current_batch_samples[i])) ** 2) \
                                * sampling_dist.derivative_score_function(current_batch_samples[i])
                gradient = gradient / batch_size

                # Update thetas of sampling distribution in the (opposite) direction of the gradient
                # The gradient variable, computed above, is already the negative of the true gradient
                sampling_thetas[trial] = sampling_thetas[trial] + alpha * gradient

                # Update sampling distribution according to the new thetas
                sampling_dist = Normal(sampling_thetas[trial])

        # Average the final thetas of sampling distribution (the last element in the thetas array for every trial) over all trials
        optimal_thetas = np.mean(sampling_thetas, axis = 0)

        return Normal(optimal_thetas)



# Softmax distribution parameterized by thetas
class BanditDistribution(Distribution):

    def __init__(self, thetas, rewards):
        assert np.size(thetas) == np.size(rewards)
        # probability of rewards[i] = e^thetas[i] / (e^thetas[0] + e^thetas[1] + ... + e^thetas[n])
        self.denominator = sum([math.exp(theta) for theta in thetas])
        self.arm_probs = np.array([math.exp(theta) / self.denominator for theta in thetas])
        assert np.sum(self.arm_probs) - 1.0 < 0.01
        self.thetas = thetas
        self.rewards = rewards

    def sample(self, n=1):
        return np.random.choice(self.rewards, n, p=self.arm_probs)

    def expected_value(self):
        return np.dot(self.arm_probs, self.rewards)

    def pdf(self, sample):
        idx = list(self.rewards).index(sample)
        return self.arm_probs[idx]

    def derivative_score_function(self, sample):
        # See handwriting for derivation detail
        idx = list(self.rewards).index(sample)
        result = np.array([-math.exp(theta) / self.denominator for theta in self.thetas])
        result[idx] += 1
        return result

    def derivative_score_function_batch(self, batch_samples):
        # See handwrting for derivation detail
        result = np.array([self.derivative_score_function(batch_samples[i]) for i in range(batch_samples.size)])
        return np.sum(result, axis = 0)

    def derivative_WIS_batch_wrt_thetas(self, batch_samples, weights_batch_samples):
        # See handwriting for derivation detail
        result = np.zeros(self.thetas.size)
        # Compute numerator
        for j in range(batch_samples.size):
            for i in range(batch_samples.size):
                result = result + weights_batch_samples[j] * weights_batch_samples[i] * (batch_samples[j] - batch_samples[i]) * self.derivative_score_function(batch_samples[i])
        # Divide result by denominator
        result = result / (np.sum(weights_batch_samples) ** 2)

        return result

    def BPG(self):
        max_iterations = 10
        max_trials = 10
        batch_size = 20
        alpha = 0.00001

        # Store all thetas for sampling distribution history
        # Notice here we store only one sampling thetas for each trial 
        # since what we care about is the final one
        sampling_thetas = np.zeros(shape = (max_trials, self.thetas.size))
        sampling_thetas[:] = self.thetas
        # initialize sampling distribution to the target distribution
        sampling_dist = BanditDistribution(self.thetas, self.rewards)

        for trial in range(max_trials):
            # Reset sampling distribution to target distribution
            sampling_dist = BanditDistribution(self.thetas, self.rewards)

            for iteration in range(max_iterations):
                # Generate a batch of samples from sampling distribution
                current_batch_IS_samples = sampling_dist.sample(n = batch_size)
                weights_current_batch_IS_samples = np.array([self.pdf(current_batch_IS_samples[i]) / sampling_dist.pdf(current_batch_IS_samples[i]) for i in range(batch_size)])

                gradient = 0.0
                for i in range(batch_size):
                    gradient = gradient + (weights_current_batch_IS_samples[i] * current_batch_IS_samples[i]) ** 2 * sampling_dist.derivative_score_function(current_batch_IS_samples[i])
                gradient = gradient / batch_size

                # Update thetas of sampling distribution in the (opposite) direction of the gradient
                # The gradient variable, computed above, is already the negative of the true gradient
                sampling_thetas[trial] = sampling_thetas[trial] + alpha * gradient

                # Update sampling distribution according to the new thetas
                sampling_dist = BanditDistribution(sampling_thetas[trial], self.rewards)

        # Average the final thetas of sampling distribution (the last element in the thetas array for every trial) over all trials
        optimal_thetas = np.mean(sampling_thetas, axis = 0)

        return BanditDistribution(optimal_thetas, self.rewards)

    def SUM(self, performance_function):
        max_iterations = 10
        max_trials = 10
        batch_size = 20
        alpha = 0.00001

        # Store all thetas for sampling distribution history
        # Notice here we store only one sampling thetas for each trial 
        # since what we care about is the final one
        sampling_thetas = np.zeros(shape = (max_trials, self.thetas.size))
        sampling_thetas[:] = self.thetas
        # initialize sampling distribution to the target distribution
        sampling_dist = BanditDistribution(self.thetas, self.rewards)

        for trial in range(max_trials):
            # Reset sampling distribution to target distribution
            sampling_dist = BanditDistribution(self.thetas, self.rewards)

            for iteration in range(max_iterations):
                # Generate a batch of samples from sampling distribution
                current_batch_samples = sampling_dist.sample(n = batch_size)
                weights_current_batch_samples = np.array([self.pdf(current_batch_samples[i]) / sampling_dist.pdf(current_batch_samples[i]) for i in range(batch_size)])

                gradient = 0.0
                for i in range(batch_size):
                    
                    gradient = gradient + np.sum((weights_current_batch_samples[i] * performance_function(current_batch_samples[i], 'Bandit') * self.derivative_score_function(current_batch_samples[i])) ** 2) \
                                * sampling_dist.derivative_score_function(current_batch_samples[i])
                gradient = gradient / batch_size

                # Update thetas of sampling distribution in the (opposite) direction of the gradient
                # The gradient variable, computed above, is already the negative of the true gradient
                sampling_thetas[trial] = sampling_thetas[trial] + alpha * gradient

                # Update sampling distribution according to the new thetas
                sampling_dist = BanditDistribution(sampling_thetas[trial], self.rewards)

        # Average the final thetas of sampling distribution (the last element in the thetas array for every trial) over all trials
        optimal_thetas = np.mean(sampling_thetas, axis = 0)

        return BanditDistribution(optimal_thetas, self.rewards)






