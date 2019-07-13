import numpy as np
import tensorflow as tf


class Softmax:

    def __init__(self, logits, rewards, trainable=False):
        assert np.size(logits) == np.size(rewards)
        self._logits = logits
        self._rewards = rewards
        self._dim = np.size(logits)
        if trainable:
            logit_init = tf.constant_initializer(logits)
            logit_var = tf.get_variable('logit_var',
                                        shape=[self._dim], dtype=tf.float64,
                                        initializer=logit_init)
            params = {'logit_var': logit_var}
            x = tf.placeholder(tf.float64, shape=[None, self._dim],
                               name='x_in')
            probs = tf.nn.softmax(logit_var)
            cross_entropy = tf.negative(x * tf.log(probs))
            loss = tf.reduce_mean(cross_entropy)
            optimizer = tf.train.GradientDescentOptimizer(1e-02)
            fit = optimizer.minimize(loss)
            self._x = x
            self._fit = fit
            self._loss = loss
            self._params = params
            self._session = tf.Session()
            self._session.run(tf.global_variables_initializer())

    @property
    def probs(self):
        probs = np.exp(self._logits)
        return probs / np.sum(probs)

    def sample(self, n=1):
        p = self.probs
        xs = np.random.choice(np.size(p), size=n, p=p)
        return xs, self._rewards[xs]

    def expected_value(self):
        return self.probs.dot(self._rewards)

    def _pdf(self, x):
        return self.probs[x]

    def pdf(self, xs):
        return np.vectorize(self._pdf)(xs)

    def mle_fit(self, xs):
        print(self._logits)
        sample_counts = np.zeros(self._dim)
        unique, counts = np.unique(xs, return_counts=True, axis=0)
        sample_counts[unique] = counts
        ps = np.array(sample_counts) * 1. / np.size(xs, axis=0)
        self._logits = np.log(ps + 1e-08)

    def grad_descent_fit(self, xs):
        logit_var = self._params['logit_var']
        ins = np.zeros((np.size(xs, axis=0), self._dim))
        for i, j in enumerate(xs):
            ins[i, j] = 1.
        feed_dict = {self._x: ins}
        self._session.run(self._fit, feed_dict=feed_dict)
        loss = self._session.run(self._loss, feed_dict=feed_dict)
        self._logits = self._session.run(logit_var)
        return loss

    def nlh(self, xs):
        return -1 * np.sum(np.log(self.pdf(xs)))
