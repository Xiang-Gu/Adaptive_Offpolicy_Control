import numpy as np
from matplotlib import pyplot as plt
import distributions
import lds
import math
import time

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

	# Implement naive REINFORCE algorithm for arbitray domains

	max_iteration = 10000
	max_trials = 1
	batch_size = 300
	alpha = 0.00001

	# Define the initial target distribution (initial target policy) that we want to optimize
	initial_thetas = np.array([1, 2, 6])

	# Bandit distribution
	rewards = np.array([1000, 10, -1])

	# Target distribution which is optimzed using naive REINFORCE
	target_dist = distributions.BanditDistribution(initial_thetas, rewards) 

	# Target distribution which is optimized using Importance Sampling REINFORCE
	# The sampling distribution it uses is the optimal sampling distribution computed by BPG (for the initial target distribution)
	# and is kept fixed througout all iterations
	target_dist_IS = distributions.BanditDistribution(initial_thetas, rewards) 
	sampling_dist_IS = target_dist_IS.BPG()

	# Target distribution which is optimized using Importance Sampling REINFORCE with BPG, 
	# meaning the sampling distribution for target distribution is updated at the end of each iteration
	# after target distribution is updated.
	target_dist_IS_BPG = distributions.BanditDistribution(initial_thetas, rewards)
	sampling_dist_IS_BPG = target_dist_IS_BPG.BPG()

	# Keep track of all target thetas so we can see the optimization process to our target distribution
	# Target_thetas are thetas of target policy for naive REINFORCE
	target_thetas = np.zeros(shape = (max_trials, max_iteration + 1, initial_thetas.size))
	target_thetas[:, 0] = initial_thetas
	# Target_thetas_IS are thetas of target policy for Importance Sampling REINFORCE
	target_thetas_IS = np.zeros(shape = (max_trials, max_iteration + 1, initial_thetas.size))
	target_thetas_IS[:, 0] = initial_thetas
	# Target_thetas_IS_BPG are thetas of target policy for Importance Sampling REINFORCE with BPG 
	target_thetas_IS_BPG = np.zeros(shape = (max_trials, max_iteration + 1, initial_thetas.size))
	target_thetas_IS_BPG[:, 0] = initial_thetas


	# Keep track of all target distribution performance (depending on different context, the definition may vary)
	# Here, in the general probabilistic context, it's defined to be the expected value of target distribution
	# In convention, this quantity is denoted by J(Pi_thetas) where Pi_thetas is the target policy parameterized by thetas
	# Again, define the performance history for three variants of REINFORCE so we can compare them with each other
	# Performance of naive REINFORCE
	target_dist_performance = np.zeros(shape = (max_trials, max_iteration + 1))
	target_dist_performance[:, 0] = target_dist.expected_value()
	# Performance of IS-REINFORCE
	target_dist_performance_IS = np.zeros(shape = (max_trials, max_iteration + 1))
	target_dist_performance_IS[:, 0] = target_dist_IS.expected_value()
	# Performance of IS-REINFORCE with BPG
	target_dist_performance_IS_BPG = np.zeros(shape = (max_trials, max_iteration + 1))
	target_dist_performance_IS_BPG[:, 0] = target_dist_IS_BPG.expected_value()

	# Main loop
	for trial in range(max_trials):
		# d(J(Pi_thetas)) / d(thetas) = E[ g(sample ~ Pi_thetas) * d(ln(Pr(sample ~ Pi_thetas))) / d(thetas)]
		# where J(Pi_thetas) is the measurement of how good the target policy Pi_thetas is.
		# In general RL context, J(Pi_thetas) has three definitions.
		# Here, since it's the general probabilistic context, J(Pi_thetas) refers to the expected value of target distribution Pi_thetas

		# Reset target distribution for each trial
		target_dist = distributions.BanditDistribution(initial_thetas, rewards)
		target_dist_IS = distributions.BanditDistribution(initial_thetas, rewards)
		target_dist_IS_BPG = distributions.BanditDistribution(initial_thetas, rewards)

		# (Important:) Reset sampling distribution for IS-REINFORCE with BPG method
		sampling_dist_IS_BPG = target_dist_IS_BPG.BPG()

		for iteration in range(max_iteration):
			print('%sth iteration in %sth trial.' % (iteration, trial))
			
			# At each iteration, collect a batch of samples for all three method
			# Naive REINFORCE collects samples from target policy (on-policy)
			# whereas IS-REINFORCE and IS-REINFORCE with BPG collect samples from its sampling distribution respectively
			current_batch_MC_samples = target_dist.sample(n = batch_size)
			current_batch_IS_samples = sampling_dist_IS.sample(n = batch_size)
			current_batch_IS_BPG_samples = sampling_dist_IS_BPG.sample(n = batch_size)

			# For IS samples, compute their likelihood ratios, too
			weight_current_batch_IS_samples = np.array([target_dist_IS.pdf(current_batch_IS_samples[i]) / sampling_dist_IS.pdf(current_batch_IS_samples[i]) for i in range(batch_size)])
			weight_current_batch_IS_BPG_samples = np.array([target_dist_IS_BPG.pdf(current_batch_IS_BPG_samples[i]) / sampling_dist_IS_BPG.pdf(current_batch_IS_BPG_samples[i]) for i in range(batch_size)])
			
			# Compute the gradient for all three method. The gradient is expressed in expectation.
			# Here we replace expectation (true gradient) with averages over a batch of samples (approximated gradient) 
			gradient = 0.0
			gradient_IS = 0.0
			gradient_IS_BPG = 0.0
			for i in range(batch_size):
				gradient = gradient + current_batch_MC_samples[i] * target_dist.derivative_score_function(current_batch_MC_samples[i])
				gradient_IS = gradient_IS + current_batch_IS_samples[i] * weight_current_batch_IS_samples[i] * target_dist_IS.derivative_score_function(current_batch_IS_samples[i])
				gradient_IS_BPG = gradient_IS_BPG + current_batch_IS_BPG_samples[i] * weight_current_batch_IS_BPG_samples[i] * target_dist_IS_BPG.derivative_score_function(current_batch_IS_BPG_samples[i])
			gradient = gradient / batch_size
			gradient_IS = gradient_IS / batch_size
			gradient_IS_BPG = gradient_IS_BPG / batch_size

			# Update target distribution's thetas in the direction of the gradient
			target_thetas[trial, iteration + 1] = target_thetas[trial, iteration] + alpha * gradient
			target_thetas_IS[trial, iteration + 1] = target_thetas_IS[trial, iteration] + alpha * gradient_IS
			target_thetas_IS_BPG[trial, iteration + 1] = target_thetas_IS_BPG[trial, iteration] + alpha * gradient_IS_BPG

			# Update target distribution
			target_dist = distributions.BanditDistribution(target_thetas[trial, iteration + 1], rewards)
			target_dist_IS = distributions.BanditDistribution(target_thetas_IS[trial, iteration + 1], rewards)
			target_dist_IS_BPG = distributions.BanditDistribution(target_thetas_IS_BPG[trial, iteration + 1], rewards)

			# Update the performance of the new target distribution (expected value in this case)
			target_dist_performance[trial, iteration + 1] = target_dist.expected_value()
			target_dist_performance_IS[trial, iteration + 1] = target_dist_IS.expected_value()
			target_dist_performance_IS_BPG[trial, iteration + 1] = target_dist_IS_BPG.expected_value()

			# (Important:) Update sampling distribution for IS-REINFORCE with BPG method
			sampling_dist_IS_BPG = target_dist_IS_BPG.BPG()

	# Plot result
	# Plot the target distribution performance for all three methods on the same subplot
	plt.subplot(4, 1, 1)
	target_dist_performance_mean = np.mean(target_dist_performance, axis = 0)
	target_dist_performance_yerr = np.std(target_dist_performance, axis = 0)
	plt.errorbar(np.arange(max_iteration + 1), target_dist_performance_mean, yerr = 1.96 * target_dist_performance_yerr / math.sqrt(max_iteration))

	target_dist_performance_IS_mean = np.mean(target_dist_performance_IS, axis = 0)
	target_dist_performance_IS_yerr = np.std(target_dist_performance_IS, axis = 0)
	plt.errorbar(np.arange(max_iteration + 1), target_dist_performance_IS_mean, yerr = 1.96 * target_dist_performance_IS_yerr / math.sqrt(max_iteration))

	target_dist_performance_IS_BPG_mean = np.mean(target_dist_performance_IS_BPG, axis = 0)
	target_dist_performance_IS_BPG_yerr = np.std(target_dist_performance_IS_BPG, axis = 0)
	plt.errorbar(np.arange(max_iteration + 1), target_dist_performance_IS_BPG_mean, yerr = 1.96 * target_dist_performance_IS_BPG_yerr / math.sqrt(max_iteration))
	
	plt.legend(['REINFORCE', 'IS-REINFORCE', 'IS-REINFORCE-BPG'])
	plt.xlabel('Iteration')
	plt.ylabel('J(target_distribution)')
	plt.title('REINFORCE')

	# Plot the changes of thetas of target distribution for naive REINFORCE
	plt.subplot(4, 1, 2)
	target_thetas_mean = np.mean(target_thetas, axis = 0)
	target_thetas_yerr = np.std(target_thetas, axis = 0)
	for idx in range(np.size(initial_thetas)):
		target_thetas_mean_idx = target_thetas_mean[:, idx]
		target_thetas_yerr_idx = target_thetas_yerr[:, idx]
		plt.errorbar(np.arange(max_iteration + 1), target_thetas_mean_idx, yerr = 1.96 * target_thetas_yerr_idx / math.sqrt(max_iteration), label = r'$\theta_%s$' % idx)
	plt.legend()
	plt.xlabel('Iteration')
	plt.ylabel('Thetas of Naive REINFORCE')

	# Plot the changes of thetas of target distribution for IS-REINFORCE
	plt.subplot(4, 1, 3)
	target_thetas_IS_mean = np.mean(target_thetas_IS, axis = 0)
	target_thetas_IS_yerr = np.std(target_thetas_IS, axis = 0)
	for idx in range(np.size(initial_thetas)):
		target_thetas_IS_mean_idx = target_thetas_IS_mean[:, idx]
		target_thetas_IS_yerr_idx = target_thetas_IS_yerr[:, idx]
		plt.errorbar(np.arange(max_iteration + 1), target_thetas_IS_mean_idx, yerr = 1.96 * target_thetas_IS_yerr_idx / math.sqrt(max_iteration), label = r'$\theta_%s$' % idx)
	plt.legend()
	plt.xlabel('Iteration')
	plt.ylabel('Thetas of IS-REINFORCE')

	# Plot the changes of thetas of target distribution for IS-REINFORCE with BPG
	plt.subplot(4, 1, 4)
	target_thetas_IS_BPG_mean = np.mean(target_thetas_IS_BPG, axis = 0)
	target_thetas_IS_BPG_yerr = np.std(target_thetas_IS_BPG, axis = 0)
	for idx in range(np.size(initial_thetas)):
		target_thetas_IS_BPG_mean_idx = target_thetas_IS_BPG_mean[:, idx]
		target_thetas_IS_BPG_yerr_idx = target_thetas_IS_BPG_yerr[:, idx]
		plt.errorbar(np.arange(max_iteration + 1), target_thetas_IS_BPG_mean_idx, yerr = 1.96 * target_thetas_IS_BPG_yerr_idx / math.sqrt(max_iteration), label = r'$\theta_%s$' % idx)
	plt.legend()
	plt.xlabel('Iteration')
	plt.ylabel('Thetas of IS-REINFORCE-BPG')
	plt.show()


if __name__ == '__main__':
	main()

