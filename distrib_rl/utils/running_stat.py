import numpy as np


class WelfordRunningStat(object):
    """
    https://www.johndcook.com/blog/skewness_kurtosis/
    """

    def __init__(self, shape):
        self.ones = np.ones(shape=shape, dtype=np.float32)
        self.zeros = np.zeros(shape=shape, dtype=np.float32)

        self.running_mean = np.zeros(shape=shape, dtype=np.float32)
        self.running_variance = np.zeros(shape=shape, dtype=np.float32)

        self.count = 0
        self.shape = shape

    def increment(self, samples, num):
        if num > 1:
            for i in range(num):
                self.update(samples[i])
        else:
            self.update(samples)

    def update(self, sample):
        current_count = self.count
        self.count += 1
        delta = (sample - self.running_mean).reshape(self.running_mean.shape)
        delta_n = (delta / self.count).reshape(self.running_mean.shape)

        self.running_mean += delta_n
        self.running_variance += delta * delta_n * current_count

    def reset(self):
        del self.running_mean
        del self.running_variance

        self.__init__(self.shape)

    @property
    def mean(self):
        if self.count < 2:
            return self.zeros
        return self.running_mean

    @property
    def std(self):
        if self.count < 2:
            return self.ones
        var = np.where(self.running_variance == 0, 1.0, self.running_variance) / (
            self.count - 1
        )
        # var = self.running_variance / (self.count - 1)

        return np.sqrt(var)

    def increment_from_obs_update(self, obs_stats_update):
        other_mean = np.asarray(obs_stats_update[0], dtype=np.float32)
        other_var = np.asarray(obs_stats_update[1], dtype=np.float32)
        other_count = obs_stats_update[2]

        count = self.count + other_count

        mean_delta = other_mean - self.running_mean
        mean_delta_squared = mean_delta * mean_delta

        combined_mean = (
            self.count * self.running_mean + other_count * other_mean
        ) / count

        combined_variance = (
            self.running_variance
            + other_var
            + mean_delta_squared * self.count * other_count / count
        )

        self.running_mean = combined_mean
        self.running_variance = combined_variance
        self.count = count
