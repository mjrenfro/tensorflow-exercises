#Topics Covered
    #1. Placeholders
    #2. TFRecord
    #3  Queue
    #4  Read CSV files
    #5  Read Image Files

from  __future__ import print_function
import tensorflow as tf
import numpy as np
import re
import matplotlib.pyplot as plt

from datetime import date
date.today()

tf.__version__

np.__version__

# sess=tf.InteractiveSession()
#---------Kyubyong code---------#
#Make data and save to npz.
_x = np.zeros((100, 10), np.int32)
for i in range(100):
    _x[i] = np.random.permutation(10)
_x, _y = _x[:, :-1], _x[:, -1]

import os
if not os.path.exists('example'): os.mkdir('example')
np.savez('example/example.npz', _x=_x, _y=_y)

data = np.load('example/example.npz')
_x, _y = data["_x"], data ["_y"]
#---------End of Kyubyong code---------#




#Q1. Make a placeholder for x such that it should be of dtype=int32
#shape = (None, 9)
# Inputs and targets

#------------------Added code-------------------#
x_pl =tf.placeholder(tf.int32, shape=(None,9))
#------------------End Added code-------------------#

#DESC: computes the sum across the
y_hat = 45 - tf.reduce_sum(x_pl, axis=2)

#Session
with tf.Session () as sess:
    _y_hat = sess.run(y_hat, {x_pl: _x})
    print("y_hat =", _y_hat[:30])
    print("true y=", _y[:30])
