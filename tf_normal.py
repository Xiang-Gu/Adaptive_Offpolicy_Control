import numpy as np
import tensorflow as tf


class Normal:

    def __init__(self, mean, sigma, trainable=False, learn_std=True,
                 precision_type='log_std'):
        self._mean = mean
        self._sigma = sigma
        self._dim = 1
        if trainable:
            mean_init = tf.constant_initializer(mean)
            mean_var = tf.get_variable('mean_var',
                                       shape=[1], dtype=tf.float64,
                                       initializer=mean_init)
            if precision_type == 'variance':
                std_init = float(sigma ** 2)
            elif precision_type == 'std':
                std_init = float(sigma)
            else:
                std_init = np.log(sigma)
            train_std = trainable and learn_std
            precision_var = tf.get_variable(
                'precision_var', initializer=tf.constant(std_init,
                                                         dtype=tf.float64),
                trainable=train_std)
            params = {'mean': mean_var, 'precision': precision_var}

            x = tf.placeholder(tf.float64, shape=[None, 1], name='x_in')
            if precision_type == 'variance':
                log_std = tf.log(tf.sqrt(precision_var))
            elif precision_type == 'std':
                log_std = tf.log(precision_var)
            else:
                log_std = precision_var
            zs = (x - mean_var) / tf.exp(log_std)
            log_likelihood = log_std + 0.5 * tf.square(zs)
            log_likelihood = - tf.reduce_sum(log_likelihood + self._dim *
                                             0.5 * np.log(2 * np.pi))
            # log_likelihood = - (tf.reduce_sum(log_std) +
            #                     0.5 * tf.reduce_sum(tf.square(zs)) +
            #                     0.5 * self._dim * np.log(2 * np.pi))
            neg_log_likelihood = tf.reduce_mean(tf.negative(log_likelihood))
            optimizer = tf.train.GradientDescentOptimizer(1e-03)
            fit = optimizer.minimize(neg_log_likelihood)
            self._x = x
            self._fit = fit
            self._loss = neg_log_likelihood
            self._params = params
            self._session = tf.Session()
            self._session.run(tf.global_variables_initializer())
            self._learn_std = learn_std
            self._precision_type = precision_type

    def sample(self, n=1):
        xs = np.random.normal(self._mean, self._sigma, size=n)
        return xs, np.array(xs)

    @property
    def mean(self):
        return self._mean

    def mle_fit(self, xs):
        self._mean = np.mean(xs)
        self._sigma = np.std(xs)

    def grad_descent_fit(self, xs):
        mean_var, std_var = self._params['mean'], self._params['precision']
        feed_dict = {self._x: xs.reshape(-1, 1)}
        self._session.run(self._fit, feed_dict=feed_dict)
        loss = self._session.run(self._loss, feed_dict=feed_dict)
        mean, precision = self._session.run([mean_var, std_var])
        self._mean = mean
        if self._precision_type == 'variance':
            self._sigma = precision ** (0.5)
        elif self._precision_type == 'std':
            self._sigma = precision
        else:
            self._sigma = np.exp(precision)
        return loss

    def _pdf(self, x):
        if self._sigma == 0:
            if x == self._mean:
                return 1.0
            else:
                return 0.0
        const = 1 / (2 * np.pi * self._sigma ** 2) ** (0.5)
        diff = x - self._mean
        return const * np.exp(-1 * diff ** 2 / (2 * self._sigma ** 2))

    def pdf(self, xs):
        return np.vectorize(self._pdf)(xs)

    def nlh(self, xs):
        if self._sigma == 0:
            if xs == self._mean:
                return 0.0
            else:
                return -1e6
        log_std = np.log(self._sigma)
        zs = (xs - self._mean) / self._sigma
        nlh = (log_std + 0.5 * np.square(zs) +
               0.5 * self._dim * np.log(2 * np.pi))
        return np.sum(nlh)
