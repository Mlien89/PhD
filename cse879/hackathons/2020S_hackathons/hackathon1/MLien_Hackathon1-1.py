#!/usr/bin/env python
# coding: utf-8

# # Hackathon #1
# 
# Mason Lien

# ### Homework
# 
# Your homework is to specify an equation that you will solve with gradient descent (as above). Then, play around with the learning rate and number of update iterations to get an intuitive understanding of how they affect your solver. Write up a paragraph or two describing your equation, how learning rate and number of iterations gave a better or worse solution, and with your intuition for why. Submit this writeup in a `.pdf` with a `.py` of your code.
# 
# I'm expecting this to take about an hour (or less if you're experienced). Feel free to use any code from this or previous hackathons. If you don't understand how to do any part of this or if it's taking you longer than that, please let me know in office hours or by email (both can be found on the syllabus). I'm also happy to discuss if you just want to ask more questions about anything in this notebook!

# The equation Ax^2 + Bx + c = 0 was used for the homework. A & B are fixed while x is optimized to match c. 

# In[ ]:


# We'll start with our library imports...
from __future__ import print_function

import numpy as np       # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs


# In[5]:


# Create a fixed matrix, A
A = tf.random.normal([4,4])
# Create a fixed matrix, B
B = tf.random.normal([4,4])
# Create x using an arbitrary initial value
x = tf.Variable(tf.ones([4, 1]))
# Create a fixed vector b
c = tf.random.normal([4, 1])

# Check the initial values
print("A:", A.numpy())
print("B:", B.numpy())
print("c:", c.numpy())
print("Initial x:", x.numpy())
print("Ax:", (A@x).numpy())
print("Bx:", (B@x).numpy())
print()


# In[12]:


learning_rate = 0.1
num_iterations = 20

# the optimizer allows us to apply gradients to update variables
optimizer = tf.keras.optimizers.Adam(learning_rate)


# In[13]:


# We want Ax^2 + Bx - c = 0, so we'll try to minimize its value
for step in range(num_iterations):
    print("Iteration", step)
    with tf.GradientTape() as tape:
        # Calculate Ax^2
        product_A = tf.matmul(A, x**2)
        # Calculate Bx
        product_B = tf.matmul(B, x)
        #calculate loss value we want to minimize
        difference_sq = tf.math.square(product_A + product_B - c)
        print("Squared error:", tf.norm(tf.math.sqrt(difference_sq)).numpy())
        # calculate the gradient
        grad = tape.gradient(difference_sq, [x])
        print("Gradients:")
        print(grad)
        # update x
        optimizer.apply_gradients(zip(grad, [x]))
        print()
        


# learning rate greatly influences the convergence of gradient descent to the solution. When I increased the lr to 1.0, the squared error increased. This is due to learning too quickly and basically making big adjustments too quickly. I kept the iterations constant when adjusting lr for the sake of understanding. When I reduced learning rate .25 -> 0.1, the loss value we that we want to minimize was reduced. This means making smaller adjustments improves convergence, meaning we are closer to the solution. When changing iterations, this is essentially how many times we want to update x and optimize the gradient. in the first couple of iterations, there are big changes, but as it gets closer and closer to the solution, the changes become gradually less. The lr and iterations need to be optimized in order for good convergence. Typically this is when you would plot the loss value over time (iterations) to see where optimal iterations are for that specific learning rate.

# In[ ]:




