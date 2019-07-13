import numpy as np
from matplotlib import pyplot as plt
import distributions
import lds
import math

class MCEstimator(object):
	
	def __init__(self):
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

	def __init__(self, weighted=False):
		self.weighted = weighted

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
	max_iteration = 2000
	max_trials = 10
	max_batches = 10
	batch_size = 50
	alpha = 0.0000001

	# Bandit Distribution
	target_thetas = np.array([2, 3, 4, 5])
	rewards = np.array([10000, 1000, 100, -1])
	target_dist = distributions.BanditDistribution(target_thetas, rewards)

	sampling_thetas_OIS = target_thetas
	sampling_thetas_WIS = target_thetas
	sampling_dist_OIS = distributions.BanditDistribution(target_thetas, rewards)
	sampling_dist_WIS = distributions.BanditDistribution(target_thetas, rewards)
	
	# Define mses, and variances history for all three estimators.
	# We didn't want to draw the change of estimates so we didn't store the estimates history
	ordinary_IS_mses = np.zeros(shape = (max_trials, max_iteration))
	ordinary_IS_vars = np.zeros(shape = (max_trials, max_iteration))
	weighted_IS_mses = np.zeros(shape = (max_trials, max_iteration))
	weighted_IS_vars = np.zeros(shape = (max_trials, max_iteration))
	MC_mses = np.zeros(shape = (max_trials, max_iteration))
	MC_vars = np.zeros(shape = (max_trials, max_iteration))

	# Get the ground truth value for the target distribution and define a
	# function to give us our squared error from ground truth.
	# mse(x) is a misleading name. It just computes the squared error of current sample x
	true_value = target_dist.expected_value()
	def mse(x):
		return (x - true_value) ** 2


	for trial in range(max_trials):
		# Reset both sampling distributions
		sampling_thetas_OIS = target_thetas
		sampling_thetas_WIS = target_thetas
		sampling_dist_OIS = distributions.BanditDistribution(target_thetas, rewards)
		sampling_dist_WIS = distributions.BanditDistribution(target_thetas, rewards)

		for idx in range(max_iteration):
			print('%sth iteration in %sth trial' % (idx, trial))
			print('thetas for OIS sampling distribution: ' + str(sampling_thetas_OIS))
			print('thetas for WIS sampling distribution: ' + str(sampling_thetas_WIS))
			print('\n\n')
			# At each iteration, generate multiple batches to approximate MSE and Variance for all three estimator
			# Notice, the Variance for MC and OIS estimators are equal to their MSE, So for Variance, we just need to compute the variance of WIS estimator.

			# Var[WIS(D)] = E[WIS(D)^2] - E[WIS(D)]^2
			# Helper array to approximate E[WIS(D)^2] and E[WIS(D)]^2 terms
			vars_WIS_helper = np.zeros(2, dtype = float)

			# gradient_MSE_OIS_helper is the accumulated gradient for OIS
			# gradient_MSE_WIS_helper is the accumulated gradient for WIS. However, since the gradient of WIS 
			# has three expecteation terms (see handwriting for derivation detail), we need to an array to store them separately.
			gradient_MSE_OIS_helper = 0.0
			gradient_MSE_WIS_helper = np.zeros(shape = (3, target_thetas.size), dtype = float)

			for i in range(max_batches):
				# Discard previous batches of samples
				# Use only current batch of samples to compute estimation
				ordinary_IS_estimator = ISEstimator()
				weighted_IS_estimator = ISEstimator(weighted = True)
				MC_estimator = MCEstimator()

				# Collect a batch of samples for each estimator and compute their likelihood ratio (for OIS and WIS estimators only)
				current_batch_MC_samples = target_dist.sample(n = batch_size)
				if idx == 0: # at the first iteration use the same batch since target_dist = sampling_dist_OIS = sampling_dist_WIS
					current_batch_OIS_samples = current_batch_WIS_samples = current_batch_MC_samples
				else:
					current_batch_OIS_samples = sampling_dist_OIS.sample(n = batch_size)
					current_batch_WIS_samples = sampling_dist_WIS.sample(n = batch_size)
				weights_current_batch_OIS_samples = np.array([target_dist.pdf(current_batch_OIS_samples[i]) / sampling_dist_OIS.pdf(current_batch_OIS_samples[i]) for i in range(batch_size)])
				weights_current_batch_WIS_samples = np.array([target_dist.pdf(current_batch_WIS_samples[i]) / sampling_dist_WIS.pdf(current_batch_WIS_samples[i]) for i in range(batch_size)])
				
				# Compute estimate for all three estimators using current batch
				for j in range(batch_size):
					MC_estimator.estimate(current_batch_MC_samples[j])
					ordinary_IS_estimator.estimate(current_batch_OIS_samples[j], weights_current_batch_OIS_samples[j])
					weighted_IS_estimator.estimate(current_batch_WIS_samples[j], weights_current_batch_WIS_samples[j])
					# Accumulate the gradient for OIS
					gradient_MSE_OIS_helper = gradient_MSE_OIS_helper + (current_batch_OIS_samples[j] * weights_current_batch_OIS_samples[j]) ** 2 * sampling_dist_OIS.derivative_score_function(current_batch_OIS_samples[j])

				# Accumulate squared error of all three estimators for current batch
				MC_mses[trial, idx] = MC_mses[trial, idx] + mse(MC_estimator.estimation)
				ordinary_IS_mses[trial, idx] = ordinary_IS_mses[trial, idx] + mse(ordinary_IS_estimator.estimation)
				weighted_IS_mses[trial, idx] = weighted_IS_mses[trial, idx] + mse(weighted_IS_estimator.estimation)
				
				# Accumulate current WIS(D) estimate squared to the first and second element of temporary variance helper array 
				vars_WIS_helper[0] = vars_WIS_helper[0] + weighted_IS_estimator.estimation ** 2
				vars_WIS_helper[1] = vars_WIS_helper[1] + weighted_IS_estimator.estimation

				# Accumulate gradient for WIS estimator
				gradient_MSE_WIS_helper[0] = gradient_MSE_WIS_helper[0] + \
						weighted_IS_estimator.estimation * (weighted_IS_estimator.estimation * sampling_dist_WIS.derivative_score_function_batch(current_batch_WIS_samples) + 2 * sampling_dist_WIS.derivative_WIS_batch_wrt_thetas(current_batch_WIS_samples, weights_current_batch_WIS_samples))
				gradient_MSE_WIS_helper[1] = gradient_MSE_WIS_helper[1] + weighted_IS_estimator.estimation
				gradient_MSE_WIS_helper[2] = gradient_MSE_WIS_helper[2] + \
						(weighted_IS_estimator.estimation * sampling_dist_WIS.derivative_score_function_batch(current_batch_WIS_samples) + sampling_dist_WIS.derivative_WIS_batch_wrt_thetas(current_batch_WIS_samples, weights_current_batch_WIS_samples))

			# Average over max_batches to approximate MSE for all three estimators
			MC_mses[trial, idx] = MC_mses[trial, idx] / max_batches 
			ordinary_IS_mses[trial, idx] = ordinary_IS_mses[trial, idx] / max_batches
			weighted_IS_mses[trial, idx] = weighted_IS_mses[trial, idx] / max_batches
			
			# Average over max_batches to approximate Variance for all three estimators
			# Notice: MC and OIS estimators are unbiased estimators, so their Variances equal to their MSE
			MC_vars[trial, idx] = MC_mses[trial, idx]
			ordinary_IS_vars[trial, idx] = ordinary_IS_mses[trial, idx]
			weighted_IS_vars[trial, idx] = (vars_WIS_helper[0] / max_batches) - (vars_WIS_helper[1] / max_batches) ** 2
			
			# Average over (batch_size * max_batch) to approximate the (opposite) gradient for OIS estimator
			gradient_MSE_OIS_helper = gradient_MSE_OIS_helper / (batch_size * max_batches)
			# Average over batch size to approximate three expectation terms in gradient for WIS estimator (array operation)
			gradient_MSE_WIS_helper = gradient_MSE_WIS_helper / max_batches


			# Adapt sampling distribution in the (opposite) direction of gradient for both OIS and WIS sampling distributions
			sampling_thetas_OIS = sampling_thetas_OIS + alpha * gradient_MSE_OIS_helper
			sampling_thetas_WIS = sampling_thetas_WIS + alpha * -(gradient_MSE_WIS_helper[0] - 2 * gradient_MSE_WIS_helper[1] * gradient_MSE_WIS_helper[2])
			sampling_dist_OIS = distributions.BanditDistribution(sampling_thetas_OIS, rewards)
			sampling_dist_WIS = distributions.BanditDistribution(sampling_thetas_WIS, rewards)
			

	# Plot results
	# Plot MSE of IS estimator and MC estimator in Log-scaled y-axis with errorbars of 95% confidence interval
	plt.subplot(2, 1, 1)
	# plt.xscale('log')
	plt.yscale('log')

	MC_mses_mean = np.mean(MC_mses, axis = 0)
	MC_mses_yerr = np.std(MC_mses, axis = 0)
	plt.errorbar(np.arange(max_iteration), MC_mses_mean, yerr = 1.96 * MC_mses_yerr / math.sqrt(max_iteration))

	ordinary_IS_mses_mean = np.mean(ordinary_IS_mses, axis = 0)
	ordinary_IS_mses_yerr = np.std(ordinary_IS_mses, axis = 0)
	plt.errorbar(np.arange(max_iteration), ordinary_IS_mses_mean, yerr = 1.96 * ordinary_IS_mses_yerr / math.sqrt(max_iteration))

	weighted_IS_mses_mean = np.mean(weighted_IS_mses, axis = 0)
	weighted_IS_mses_yerr = np.std(weighted_IS_mses, axis = 0)
	plt.errorbar(np.arange(max_iteration), weighted_IS_mses_mean, yerr = 1.96 * weighted_IS_mses_yerr / math.sqrt(max_iteration))

	plt.title('Importance Sampling')
	plt.xlabel('Iteration')
	plt.ylabel('Mean Square Error (in log-scale)')
	plt.legend([ 'MC', 'OIS-BPG', 'WIS-WBPG'])
	
	
	# # Plot thetas of sampling distribution
	plt.subplot(2, 1, 2)
	# plt.xscale('log')
	plt.yscale('log')

	MC_vars_mean = np.mean(MC_vars, axis = 0)
	MC_vars_yerr = np.std(MC_vars, axis = 0)
	plt.errorbar(np.arange(max_iteration), MC_vars_mean, yerr = 1.96 * MC_vars_yerr / math.sqrt(max_iteration))

	ordinary_IS_vars_mean = np.mean(ordinary_IS_vars, axis = 0)
	ordinary_IS_vars_yerr = np.std(ordinary_IS_vars, axis = 0)
	plt.errorbar(np.arange(max_iteration), ordinary_IS_vars_mean, yerr = 1.96 * ordinary_IS_vars_yerr / math.sqrt(max_iteration))

	weighted_IS_vars_mean = np.mean(weighted_IS_vars, axis = 0)
	weighted_IS_vars_yerr = np.std(weighted_IS_vars, axis = 0)
	plt.errorbar(np.arange(max_iteration), weighted_IS_vars_mean, yerr = 1.96 * weighted_IS_vars_yerr / math.sqrt(max_iteration))

	plt.xlabel('Iteration')
	plt.ylabel('Variance (in log-scale)')
	plt.legend(['MC', 'OIS-BPG', 'WIS-WBPG'])
	
	plt.show()

if __name__ == '__main__':
	main()
