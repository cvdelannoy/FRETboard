import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

# test: a 2-state model
nb_variables=2
seq_length=5

input = tf.placeholder(tf.float32, [1,5])

p_start = tf.Variable([0.8, 0.2])

# emissions
e_mu = tf.Variable([0.0, 2.0])
e_sd = tf.Variable([1.0, 0.5])
ndist = tfd.Normal(loc=e_mu, scale=e_sd)

# transition matrix
tm = tf.Variable([[0.9, 0.3],
                  [0.1, 0.7]])
tf.



bwd_table = tf.placeholder([seq_length, nb_variables])
bwd_table = tf.placeholder([seq_length, nb_variables])

tst1 = ndist.prob([0.0, 2.0])
tst2 = ndist.prob([2.0, 0.0])

ses = tf.Session()
ses.run(tf.global_variables_initializer())
print(ses.run([tst1]))
print(ses.run([tst2]))
