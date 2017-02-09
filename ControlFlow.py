#also just following the tensorflow api
import tensorflow as tf
import  numpy as np

print (tf.__version__)
print (np.__version__)

sess=tf.InteractiveSession()

#Using cond
#Q1: Let x and y be random 0-D tensors. Return x+y if x<y and x-y otherwise
x=tf.random_uniform([])
y=tf.random_uniform([])
result=tf.cond(x<y, lambda: tf.add(x,y), lambda: tf.sub(x,y))
print("Q1: ", result.eval())

#Using case
#Q2: Let x and y be 0-D int tensors randomly selected from 0 to 5.
#Return x+y if x<y. Elif x>y return x-y, 0 if equal
x=tf.random_uniform([], maxval=5, dtype=tf.int32)
y=tf.random_uniform([], maxval=5, dtype=tf.int32)
result=tf.case({x<y: lambda: tf.add(x,y), x>y: lambda: tf.sub(x,y)}, default=lambda:tf.constant(0), exclusive=True)
print("Q2: ", result.eval())

#Using equal
#Q3: Let x be a tensor [[-1,-2,-3],[0,1,2]] and y be a tensor of zeros with the
#same shape as x. Return a boolean tensor that yields Trues if x equals y elementwise
x=tf.constant([[-1,-2,-3],[0,1,2]])
y=tf.zeros(tf.shape(x), "int32")
result=tf.equal(x,y)
print ("Q3:\n", result.eval())

#Using not_equal
#Q4 Let x be a tensor [[-1,-2,-3],[0,1,2]] and y be a tensor of zeros
#with the same shape as x. Return a boolean tensor that yields True if
#x does not equal y elementwise
x=tf.constant([[-1,-2,-3],[0,1,2]])
y=tf.zeros(tf.shape(x), "int32")
result=tf.not_equal(x,y)
print("Q4:\n", result.eval())

#Q5 Let x be a tensor [[-1,-2,-3],[0,1,2]] and y be a tensor of zeros
#with the same shape as x. Return a boolean tensor that yields
#Trues if x is greater than or equal to y elementwise
x=tf.constant([[-1,-2,-3],[0,1,2]])
y=tf.zeros(tf.shape(x), "int32")
result=tf.greater_equal(x,y)
print("Q5:\n", result.eval())

#Q6 Let x be a tensor [[-1,-2,-3],[0,1,2]] and y be a tensor of zeros
#with the same shape as x. Return a boolean tensor that yields
#Trues if x is less than or equal to y elementwise
x=tf.constant([[-1,-2,-3],[0,1,2]])
y=tf.zeros(tf.shape(x), "int32")
result=tf.less_equal(x,y)
print("Q6:\n", result.eval())

#Q7. Let x be a 0-D tensor randomly selected from -5 to 5.
#Return a boolean tensor that yields Trues if x is less than
#3 and x is greater than 0
x=tf.random_uniform([],minval=-5,maxval=5, dtype=tf.int32)
z=tf.constant(3)
y=tf.constant(0)
result=tf.logical_and(tf.less(x,z), tf.greater(x,y))
print("Q7:\n", result.eval())

#Using select
#Q8. Let x be a tensor [[1,2],[3,4]], y be a tensor [[5,6],[7,8]] and
#z be a boolean tensor [[True, False],[False, True]]. Create
#a 2*2 tensor such that each element corresponds to x if True
#otherwise y.
x=tf.constant([[1,2],[3,4]])
y=tf.constant([[5,6],[7,8]])
z=tf.constant([[True,False],[False,True]])
result=tf.select(z,x,y)
print("Q8:\n", result.eval())

#Q9. Let x be a tensor [1,2,3,...,100]. Extract elements of x
#greater than 30
x=tf.range(1,101)
#need to learn more about tensor transformations
