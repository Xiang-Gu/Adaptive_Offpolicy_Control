import numpy as np
from matplotlib import pyplot as plt
import distributions
import lds
import math

class MCEstimator(object):
	
	def __init__(self, dist):
		self.dist = dist
		self.estimation = None
		self.num_samples = 0

	def estimate(self, new_sample):
		if self.estimation is None: # First sampling coming in
			self.estimation = new_sample
		else:
			# Incrementally compute the new estimate using old estimate and new sample
			self.estimation = self.estimation + (new_sample - self.estimation) / float(self.num_samples + 1)

		# Update number of samples
		self.num_samples = self.num_samples + 1

		return self.estimation

class ISEstimator(object):

	def __init__(self, target_dist, sampling_dist,
				 weighted=False, per_decision=False):
		self.target_dist = target_dist
		self.sampling_dist = sampling_dist
		self.weighted = weighted
		self.per_decision = per_decision

		# num_samples is the total number of samples seen so far used in OIS
		# Total_weights is the sum of weights of all samples seen so far used in WIS
		self.estimation = None
		self.total_weights = 0.0
		self.num_samples = 0

	def estimate(self, new_sample, weight_new_sample):
		if self.estimation is None: # First sample coming in
			if self.weighted:
				self.estimation = new_sample
			else:
				self.estimation = weight_new_sample * new_sample
		else:
			# incrementally compute the new estimate using old estimate and new sample
			if self.weighted:
				self.estimation = ( self.estimation * self.total_weights + weight_new_sample * new_sample ) / (self.total_weights + weight_new_sample)
			else:
				self.estimation = self.estimation + (weight_new_sample * new_sample - self.estimation) / float(self.num_samples + 1) 

		# Update number of samples and total_weights
		self.num_samples = self.num_samples + 1
		self.total_weights = self.total_weights + weight_new_sample

		return self.estimation

def main():

	# Implement off-policy evaluation for arbitrary domains
	max_samples = 10000
	max_trials = 10
	alpha = 0.0000001
	# alpha_sigma = 0.00000001
	# alpha_K = 0.00000000022

	# Define target and sampling distribution
	# Initialize sampling distribution with target thetas

	# Define parameters that define both target and sampling distribution
	# Notice the first thetas of sampling_thetas in each trial is initialized to target_thetas
	# So we need to add one spot to the second dimension of sampling_thetas
	target_thetas = np.array([2, 5])
	rewards = np.array([1000, -2])
	sampling_thetas = np.zeros(shape = (max_trials, max_samples + 1, target_thetas.size))
	sampling_thetas[:, 0] = target_thetas

	# # Normal Distribution
	# target_dist = distributions.Normal(target_thetas)
	# sampling_dist = distributions.Normal(target_thetas)

	# # Bandit Distribution
	target_dist = distributions.BanditDistribution(target_thetas, rewards)
	sampling_dist = distributions.BanditDistribution(target_thetas, rewards)

	# # IDS Distribution
	# dimension = 2
	# target_K = np.array([ [3,1], [2,-1], [0, 2.5], [-1.5,-1.5] ])
	# target_sigma = 1 * np.ones(dimension, dtype = float)
	# sampling_K = np.zeros(shape = (max_trials, max_samples + 1, target_K.shape[0], target_K.shape[1]))
	# sampling_K[:, 0] = target_K
	# sampling_sigma = np.zeros(shape = (max_trials, max_samples + 1, dimension))
	# sampling_sigma[:, 0] = target_sigma
	# target_dist = lds.LDS(dimension, target_K, target_sigma, L=10)
	# sampling_dist = lds.LDS(dimension, target_K, target_sigma, L=10)
	

	# Define estimates and mses history for both estimators
	ordinary_IS_estimates = np.zeros(shape = (max_trials, max_samples))
	ordinary_IS_mses = np.zeros(shape = (max_trials, max_samples))
	weighted_IS_estimates = np.zeros(shape = (max_trials, max_samples))
	weighted_IS_mses = np.zeros(shape = (max_trials, max_samples))
	MC_estimates = np.zeros(shape = (max_trials, max_samples))
	MC_mses = np.zeros(shape = (max_trials, max_samples))

	# Get the ground truth value for the target distribution and define a
	# function to give us our squared error from ground truth.
	true_value = target_dist.expected_value()
	def mse(x):
		return (x - true_value) ** 2

	for trial in range(max_trials):
		# Restore sampling distribution to initial state (used in OIS w/ BPG)
		# sampling_dist = distributions.Normal(target_thetas)
		sampling_dist = distributions.BanditDistribution(target_thetas, rewards)
		# sampling_dist = lds.LDS(dimension, target_K, target_sigma, L=10)

		# Define ordinary IS estimator and weighted IS estimator for this trial
		ordinary_IS_estimator = ISEstimator(target_dist, sampling_dist)
		weighted_IS_estimator = ISEstimator(target_dist, sampling_dist, weighted = True)
		MC_estimator = MCEstimator(target_dist)

		# Repeat for an increasing number of samples
		for idx in range(max_samples):
			# Collect samples from sampling distribution and its likelihood ratio (for IS-estimator) 
			current_IS_sample = sampling_dist.sample()
			weight_current_IS_sample = target_dist.pdf(current_IS_sample) / sampling_dist.pdf(current_IS_sample)
			# current_IS_path, reward_current_IS_path = sampling_dist.sample()
			# weight_current_IS_path = target_dist.pdf(current_IS_path[0]) / sampling_dist.pdf(current_IS_path[0])
			# print(str(trial) + 'th trial, ' + str(idx) + 'th iteration: \nGenerate sample: ' + str(reward_current_IS_path) + 
			# 	' with ratio: ' + str(weight_current_IS_path) + '\nsampling_dist_sigma: ' + str(sampling_sigma[trial, idx]) + '\nsampling_dist_K: ' + str(sampling_K[trial, idx]) + '\n\n')

			# Collect samples from target distribution (for MC-estimator)
			current_MC_sample = target_dist.sample()
			# current_MC_path, reward_current_MC_path = target_dist.sample()

			# Use same sample in the first iteration for both estimator
			if idx == 0:
				current_MC_sample = current_IS_sample
				# current_MC_path = current_IS_path
				# reward_current_MC_path = reward_current_IS_path

			# Compute latest estimate of both estimator
			ordinary_IS_estimates[trial, idx] = ordinary_IS_estimator.estimate(current_IS_sample, weight_current_IS_sample)
			weighted_IS_estimates[trial, idx] = weighted_IS_estimator.estimate(current_IS_sample, weight_current_IS_sample)
			MC_estimates[trial, idx] = MC_estimator.estimate(current_MC_sample)
			# ordinary_IS_estimates[trial, idx] = ordinary_IS_estimator.estimate(reward_current_IS_path[0], weight_current_IS_path)
			# weighted_IS_estimates[trial, idx] = weighted_IS_estimator.estimate(reward_current_IS_path[0], weight_current_IS_path)
			# MC_estimates[trial, idx] = MC_estimator.estimate(reward_current_MC_path[0])

			# Compute latest MSE using latest estimate
			ordinary_IS_mses[trial, idx] = mse(ordinary_IS_estimates[trial, idx])
			weighted_IS_mses[trial, idx] = mse(weighted_IS_estimates[trial, idx])
			MC_mses[trial, idx] = mse(MC_estimates[trial, idx])

			# Adapt sampling distribution in the (opposite) direction of 
			# gradient of MSE of ISEstimate w.r.t. thetas
			# thetas = thetas + alpha * (IS(sample) ^ 2 * d(ln(pdf(sample)))) / d(thetas)
			sampling_thetas[trial, idx + 1] = sampling_thetas[trial, idx] + alpha * (((weight_current_IS_sample * current_IS_sample) ** 2 - ordinary_IS_estimates[trial, idx - 1] ** 2) * sampling_dist.derivative_score_function(current_IS_sample))
			# sampling_dist = distributions.Normal(sampling_thetas[trial, idx + 1])
			sampling_dist = distributions.BanditDistribution(sampling_thetas[trial, idx + 1], rewards)
			# sampling_K[trial, idx + 1] = sampling_K[trial, idx] + alpha_K * ((weight_current_IS_path * reward_current_IS_path[0]) ** 2 * sampling_dist.derivative_path_log_likelihood_wrt_K(current_IS_path[0]))
			# sampling_sigma[trial, idx + 1] = sampling_sigma[trial, idx] + alpha_sigma * ((weight_current_IS_path * reward_current_IS_path[0]) ** 2 * sampling_dist.derivative_path_log_likelihood_wrt_sigma(current_IS_path[0]))
			# sampling_dist = lds.LDS(dimension, sampling_K[trial, idx + 1], sampling_sigma[trial, idx + 1], L=10)
			



	# Plot results
	# Plot MSE of IS estimator and MC estimator in Log-scaled y-axis with errorbars of 95% confidence interval
	plt.subplot(2, 1, 1)
	# plt.xscale('log')
	plt.yscale('log')

	ordinary_IS_mses_mean = np.mean(ordinary_IS_mses, axis = 0)
	ordinary_IS_mses_yerr = np.std(ordinary_IS_mses, axis = 0)
	ordinary_IS_mses_sizes = np.arange(max_samples)
	plt.errorbar(ordinary_IS_mses_sizes, ordinary_IS_mses_mean, yerr = 1.96 * ordinary_IS_mses_yerr / math.sqrt(max_samples))

	weighted_IS_mses_mean = np.mean(weighted_IS_mses, axis = 0)
	weighted_IS_mses_yerr = np.std(weighted_IS_mses, axis = 0)
	weighted_IS_mses_sizes = np.arange(max_samples)
	plt.errorbar(weighted_IS_mses_sizes, weighted_IS_mses_mean, yerr = 1.96 * weighted_IS_mses_yerr / math.sqrt(max_samples))

	MC_mses_mean = np.mean(MC_mses, axis = 0)
	MC_mses_yerr = np.std(MC_mses, axis = 0)
	MC_mses_sizes = np.arange(max_samples)
	plt.errorbar(MC_mses_sizes, MC_mses_mean, yerr = 1.96 * MC_mses_yerr / math.sqrt(max_samples))

	plt.xlabel('Samples')
	plt.ylabel('Mean Square Error (in log-scale)')
	plt.title('Importance Sampling')
	plt.legend(['OIS w/ BPG', 'WIS w/ BPG', 'MC'])
	
	
	# # Plot thetas of sampling distribution
	plt.subplot(2, 1, 2)
	sampling_thetas_mean = np.mean(sampling_thetas, axis = 0)
	sampling_thetas_yerr = np.std(sampling_thetas, axis = 0)
	sampling_thetas_sizes = np.arange(max_samples)
	# Plot each component of sampling distribution thetas separately with errorbars of 95% confidence interval, but on the same subplot
	for idx in range(np.size(target_thetas)):
		sampling_thetas_mean_idx = sampling_thetas_mean[1:, idx] # Start from the second thetas because the first thetas is target_thetas
		sampling_thetas_yerr_idx = sampling_thetas_yerr[1:, idx] # Also start from the second element because the first element is the std of the target_thetas, which is always 0
		plt.errorbar(sampling_thetas_sizes, sampling_thetas_mean_idx, yerr = 1.96 * sampling_thetas_yerr_idx / math.sqrt(max_samples), label = r'$\theta_%s$' % idx)
	plt.legend()
	plt.xlabel('Samples')
	plt.ylabel('Optimal Parameters')
	plt.title('Behavior Policy Gradient')
	plt.show()

if __name__ == '__main__':
	main()
