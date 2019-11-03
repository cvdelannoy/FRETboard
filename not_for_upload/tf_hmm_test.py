import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

# A simple weather model.

# Represent a cold day with 0 and a hot day with 1.
# Suppose the first day of a sequence has a 0.8 chance of being cold.
# We can model this using the categorical distribution:

initial_distribution = tfd.Categorical(probs=[0.8, 0.2])

# Suppose a cold day has a 30% chance of being followed by a hot day
# and a hot day has a 20% chance of being followed by a cold day.
# We can model this as:

transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],
                                                 [0.2, 0.8]])

# Suppose additionally that on each day the temperature is
# normally distributed with mean and standard deviation 0 and 5 on
# a cold day and mean and standard deviation 15 and 10 on a hot day.
# We can model this with:

observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

# We can combine these distributions into a single week long
# hidden Markov model with:

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

# The expected temperatures for each day are given by:
ses = tf.Session()
ses.run(tf.global_variables_initializer())
ses.run([model.mean(), model.log_prob(tf.zeros(shape=[7]))])

