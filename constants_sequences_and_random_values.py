import tensorflow as tf
import numpy as np

sess=tf.InteractiveSession()

#Q1:Create a tensor of the shape [2,3] with all elements set to zero
y=tf.zeros([3,2], "float")
print("Q1: ", y.eval())

#Q2. Let x be a tensor of [[1,2,3], [4,5,6]]
#Create a tensor of the same shape and dtype as X with all elemens set to zero
x=tf.constant([[1,2,3],[4,5,6]])
y=tf.zeros(tf.shape(x), "float")
print("Q2: ", y.eval())

#Q3 Create a tensor of shape [2,3] with all elements set to one
y=tf.fill([2,3], 1)
print("Q3: ", y.eval())

#Q4 Let x be a tensor of [[1,2,3],[4,5,6]]
#Create a tensor of the same shape and dtype as x with all elements set to one
x=tf.constant([[1,2,3],[4,5,6]])
y=tf.fill(tf.shape(x), 1)
print("Q4: ", y.eval())

#Q5 Create a tensor of the shape [3,2], with all elements of 5
y=tf.fill([3,2],5)
print ("Q5: ", y.eval())

#Q6 Create a constant tensor of [[1,3,5],[4,6,8]]
y=tf.constant([[1,3,5],[4,6,8]])
print("Q6: ",y.eval())

#Q7 Create a constant tensor of the shape [2,3] with all elements set to 4
y=tf.fill([2,3],4)
print("Q7: ", y.eval())


#Q8 Create a 1=D tensor of 50 evenly spaced elements between 5 and 10 inclusive
y=tf.linspace(5.0, 10.0, 50)
print("Q8: ", y.eval())

#Q9 Create a tensor which is the sequence of even #'s between 10 and 100 inclusive
y=tf.range(10,102,2)
print ("Q9: ",y.eval())

#Q10 Create a random tensor of the shape [3,2], with elements from a normal distribution of u=0 and std=2
y=tf.random_normal([3,2], 0.0,2)
print ("Q10: ", y.eval())

#Q11 Create a random tensor of the shape [3,2], with elements from a normal distribution of u=0 and std=1 and no values exceed 2 stds from the u
y=tf.truncated_normal([3,2])
print ("Q11: ", y.eval())

#Q12 Create a random tensor of the shape [3,2] with all elements from a uniform distribution that ranges from 0 to 2 (exclusive)
y=tf.random_uniform([3,2],0,2)
print ("Q12: ", y.eval())

#Q13 Let x be a tensor of [[1,2],[3,4],[5,6]]. Shuffle x along its first dimension
x=tf.constant([[1,2],[3,4],[5,6]])
y=tf.random_shuffle(x)
print ("Q13: ", y.eval())

#Q14 Let x be a random tensor of the shape [10, 10, 3] with elements from a unit normal distribution. Crop x with the size of [5,5,3]
x=tf.random_normal([10,10,3])
y=tf.random_crop(x, [5,5,3])
print ("Q14: ", y.eval())

 




