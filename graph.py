from __future__ import print_function
import numpy as np
import tensorflow as tf

from datetime import date
date.today()

#Q1 Creating a Graph...basically with the default ctor

#use this for multiple graphs should be used in the
#same process

g=tf.Graph()
with g.as_default():
    #operations to be added to the graph
    with tf.name_scope("inputs"):
        a=tf.constant(2,tf.int32, name="a")
        b=tf.constant(3,tf.int32, name ="b")

    #ops
    with tf.name_scope("ops"):
        c=tf.multiply(a,b, name="c")
        d=tf.add(a,b, name="d")
        e=tf.subtract(c,d,name="e")

#Q2
sess=tf.Session(graph=g)

_c, _d, _e=sess.run([c,d,e])

#TODO: why are the values not being retrieved correctly
print("c =", c)
print("d =", d)
print("e =", e)

sess.close()

############Summary###############

tf.reset_default_graph()

#Define inputs0
a=tf.Variable(tf.random_uniform([]))
b_pl=tf.placeholder(tf.float32, [None])

#ops
c= a* b_pl
d=a+b_pl
e=tf.reduce_sum(c)
f=tf.reduce_mean(d)
g=e-f

#initialize variables(s)
init = tf.global_variables_initializer()

#Update variable
update_op = tf.assign(a, a+g)

#Q4. Create a (summary) writer to 'asset'

writer=tf.summary.FileWriter("asset", tf.get_default_graph())

#Q5. Add `a` to summary scalar
tf.summary.scalar("a", a)

#Q6. Add `c` and `d` to summary.histogram
tf.summary.histogram("c",c)
tf.summary.histogram("d",d)

#Q7. Merge all summaries
summaries=tf.summary.merge_all()

sess = tf.Session()

sess.run(init)

#Fetching values
for step in range(5):
    _b = np.arange(10, dtype=np.float32)
    #TODO: understand what the following line means

    _, summaries_proto = sess.run([update_op, summaries], {b_pl:_b})

    #Q8. Attach summaries_proto to TensorBoard
    writer.add_summary(summaries_proto, global_step=step)

sess.close()
