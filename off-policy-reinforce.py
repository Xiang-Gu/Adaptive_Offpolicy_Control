import numpy as np
from matplotlib import pyplot as plt
import distributions
import lds
import math
import time
import argparse

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

def set_target_dist(dist_type, thetas, rewards):
	if dist_type == 'Normal':
		return distributions.Normal(thetas)
	elif dist_type == 'Bandit':
		return distributions.BanditDistribution(thetas, rewards)
	else:
		print('Unsupported Arguments Types.')
		exit()

# Define different performance metrix for different distribution
def performance(dist, dist_type):
	if dist_type == 'Normal':
		return -(np.mean(dist.sample(n=1000)) - 5) ** 2
	elif dist_type == 'Bandit':
		return dist.expected_value()
	else:
		print('Unsupported Distribution Types.')
		exit()

# Define the performance metrix for a single sample input
# This should be in accordance of the performance function
def performance_function(sample, dist_type):
	if dist_type == 'Normal':
		return -(sample - 5) ** 2
	elif dist_type == 'Bandit':
		return sample
	else:
		print('Unsupported Distribution Types.')
		exit()

def main():
	# Implement naive REINFORCE algorithm for arbitray domains
	max_iteration = 1000
	max_trials = 5
	batch_size = 100
	alpha = 0.005
	
	# Define target distribution and its sampling distribution according to input
	parser = argparse.ArgumentParser()
	parser.add_argument("--dist", help="target distribution")
	parser.add_argument("--thetas", help="parameters of target distribution")
	parser.add_argument("--rewards", help="rewards of Bandit Distribution")
	args = parser.parse_args()
	initial_thetas = np.array(args.thetas.split(","), dtype=float)
	rewards = np.array(args.rewards.split(","), dtype=float)
	target_dist_SUM = set_target_dist(args.dist, initial_thetas, rewards)
	sampling_dist_SUM = target_dist_SUM.SUM(performance_function)

	# Keep track of all target thetas so we can see the optimization process to our target distribution
	# Target_thetas are thetas of target policy for off-policy-REINFORCE with SUM 
	target_thetas = np.zeros(shape = (max_trials, max_iteration + 1, initial_thetas.size))
	target_thetas[:, 0] = initial_thetas

	# Keep track of the performance of target distribution
	target_dist_performance = np.zeros(shape = (max_trials, max_iteration + 1))
	target_dist_performance[:, 0] = performance(target_dist_SUM, args.dist)


	for trial in range(max_trials):
		# Reset target distribution and its sampling distribution
		target_dist_SUM = set_target_dist(args.dist, initial_thetas, rewards)
		sampling_dist_SUM = target_dist_SUM.SUM(performance_function)

		for iteration in range(max_iteration):
			print('%sth iteration in %sth trial.' % (iteration, trial))

			# Collect a batch of samples from the new optimal sampling distribution
			# and use IS to get an approximated gradient of performance of target distribution 
			current_batch_SUM_samples = sampling_dist_SUM.sample(n=batch_size)
			# Compute their likelihood ratios, too
			weight_current_batch_SUM_samples = np.array([target_dist_SUM.pdf(current_batch_SUM_samples[i]) / sampling_dist_SUM.pdf(current_batch_SUM_samples[i]) for i in range(batch_size)])

			# Compute the gradient for all three method. The gradient is expressed in expectation.
			# Here we replace expectation (true gradient) with averages over a batch of samples (approximated gradient) 
			gradient = 0.0
			for i in range(batch_size):
				gradient = gradient + weight_current_batch_SUM_samples[i] * performance_function(current_batch_SUM_samples[i], args.dist) * target_dist_SUM.derivative_score_function(current_batch_SUM_samples[i])
			gradient = gradient / batch_size

			# Update target distribution's thetas in the direction of the gradient
			target_thetas[trial, iteration + 1] = target_thetas[trial, iteration] + alpha * gradient

			# Update target distribution
			target_dist_SUM = set_target_dist(args.dist, target_thetas[trial, iteration + 1], rewards)

			# Update the performance of the new target distribution
			target_dist_performance[trial, iteration + 1] = performance(target_dist_SUM, args.dist)

			# (Important:) Update sampling distribution for this new target distribution
			sampling_dist_SUM = target_dist_SUM.SUM(performance_function)

	# Plot result
	# Plot change of performance of target distribution
	plt.subplot(2, 1, 1)
	target_dist_performance_mean = np.mean(target_dist_performance, axis = 0)
	target_dist_performance_yerr = np.std(target_dist_performance, axis = 0)
	plt.errorbar(np.arange(max_iteration + 1), target_dist_performance_mean, yerr = 1.96 * target_dist_performance_yerr / math.sqrt(max_iteration))
	plt.xlabel('Iteration')
	plt.ylabel('J(target_distribution)')
	plt.title('off-policy-REINFORCE w/ SUM')

	# Plot the change of parameters of target distribution
	plt.subplot(2, 1, 2)
	target_thetas_mean = np.mean(target_thetas, axis = 0)
	target_thetas_yerr = np.std(target_thetas, axis = 0)
	for idx in range(np.size(initial_thetas)):
		target_thetas_mean_idx = target_thetas_mean[:, idx]
		target_thetas_yerr_idx = target_thetas_yerr[:, idx]
		plt.errorbar(np.arange(max_iteration + 1), target_thetas_mean_idx, yerr = 1.96 * target_thetas_yerr_idx / math.sqrt(max_iteration), label = r'$\theta_%s$' % idx)
	plt.legend()
	plt.xlabel('Iteration')
	plt.ylabel('Thetas')
	plt.show()


if __name__ == '__main__':
	main()

