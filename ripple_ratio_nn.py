# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:10:37 2017

@author: eric
"""

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Predict the normalized ripple ratio for a network of neurons with varying amounts of DC input.

Network of biophysical neurons modified from Brunel and Wang (2003).
Artificial neural network modified from the TensorFlow MNIST tutorial
"""
# This artificial neural network predicts the normalized ripple ratio, which is
# defined as the normalized ratio of ripple power to the low frequency power,
# for a network of neurons. The ripple power is the integral of power of the
# frequency at the peak of the power spectrum +-20 Hz. Low frequency power is
# the integral of power in the range [4,50] Hz.
# The network of neurons from which the power spectrum was taken from Brunel
# and Wang (2003) and includes 4000 pyramidal neurons (p) and 1000 interneurons
# (i), biophysically modeled with the LIF model. Their code was modified to
# include DC current input to each neuron type.
# This artificial neural network takes the amount of DC current as an input,
# and predicts the normalized ripple ratio as the output.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# load ripple ratio data
data_all=np.load('data/rippleratio_data.npz')
additional_inputs=np.load('data/additional_inputs.npz')
num_data=np.size(data_all['rippleratio'])
y_data_all=np.reshape(data_all['rippleratio'], (num_data, 1))
x_data_all=np.reshape(np.array(np.meshgrid(data_all['DCp']*2, data_all['DCi'])).T, (num_data, 2)) # multiply DCp by 2 for feature scaling
add_x_data=np.reshape(additional_inputs['maxpowerfreq'], (num_data, 1)) # frequency of the peak of the power spectrum
x_data_all=np.hstack((x_data_all,(add_x_data-np.min(add_x_data))/(np.max(add_x_data)-np.min(add_x_data)))) # feature scale add_x_data

##create more input features
#feature1=x_data_all**2
#feature2=np.log(x_data_all)
#x_data_all=np.hstack((x_data_all,(feature1-np.min(feature1))/(np.max(feature1)-np.min(feature1)),(feature2-np.min(feature2))/(np.max(feature2)-np.min(feature2))))

dataset_sizes=30 # number of dataset sizes to try
final_cost_output_train=np.zeros(dataset_sizes)
final_cost_output_test=np.zeros(dataset_sizes)

for j in range(dataset_sizes):
    # randomize data so you can choose training and test sets
    data_indices=np.arange(num_data)
    np.random.shuffle(data_indices)

    # divide out train and test sets
    train_pct=0.7 # proportion of dataset that will be used for training
    #test_pct=1-train_pct

    x_data_train=x_data_all[data_indices[:int(len(data_indices)*train_pct)], :]
    y_data_train=y_data_all[data_indices[:int(len(data_indices)*train_pct)], :]
    x_data_test=x_data_all[data_indices[int(len(data_indices)*train_pct):], :]
    y_data_test=y_data_all[data_indices[int(len(data_indices)*train_pct):], :]

    x_data_train=x_data_train[:int(len(x_data_train)-(1*j)),:]
    y_data_train=y_data_train[:int(len(y_data_train)-(1*j)),:]

    # Create the model
    input_nodes=3
    hidden_layer_nodes=50
    hidden_layer2_nodes=50
    #hidden_layer3_nodes=10
    output_nodes=1
    x = tf.placeholder(tf.float64, [None, input_nodes])
    W = tf.Variable(tf.zeros([input_nodes, hidden_layer_nodes],dtype=tf.float64))
    b = tf.Variable(tf.zeros([hidden_layer_nodes],dtype=tf.float64))
    h = tf.nn.relu(tf.matmul(x, W) + b)
    hW = tf.Variable(tf.zeros([hidden_layer_nodes, hidden_layer2_nodes],dtype=tf.float64))
    hb = tf.Variable(tf.zeros([hidden_layer2_nodes],dtype=tf.float64))
    h2 = tf.nn.relu(tf.matmul(h, hW) + hb)
    hW2 = tf.Variable(tf.zeros([hidden_layer2_nodes, output_nodes],dtype=tf.float64))
    hb2 = tf.Variable(tf.zeros([output_nodes],dtype=tf.float64))
    #h3 = tf.nn.relu(tf.matmul(h2, hW2) + hb2)
    #hW3 = tf.Variable(tf.zeros([hidden_layer3_nodes, output_nodes],dtype=tf.float64))
    #hb3 = tf.Variable(tf.zeros([output_nodes],dtype=tf.float64))
    y = tf.matmul(h2, hW2) + hb2

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float64, [None, 1])

    learning_rate=0.05
    iterations=100
    costtrain = tf.reduce_sum((y - y_)**2)/(2 * len(y_data_train))#tf.nn.l2_loss(y-y_)
    costtest = tf.reduce_sum((y - y_)**2)/(2 * len(y_data_test))#tf.nn.l2_loss(y-y_)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(costtrain)

    sess = tf.InteractiveSession()

    # Train
    tf.global_variables_initializer().run()
    cost_output_train=np.zeros(iterations)
    cost_output_test=np.zeros(iterations)
    for k in range(iterations):
        sess.run(train_step, feed_dict={x: x_data_train, y_: y_data_train})
        cost_output_train[k]=sess.run(costtrain, feed_dict={x: x_data_train, y_: y_data_train})
        cost_output_test[k]=sess.run(costtest, feed_dict={x: x_data_test, y_: y_data_test})
    final_cost_output_train[j]=cost_output_train[-1]
    final_cost_output_test[j]=cost_output_test[-1]

#    plt.figure()
#    plt.plot(range(iterations),cost_output_train,'k',label='train')
#    plt.plot(range(iterations),cost_output_test,'r',label='test')
#    plt.xlabel('iteration number')
#    plt.ylabel('cost')
#    plt.legend()

    # Test trained model
    correct_threshold=0.1
    correct_prediction = tf.less(tf.abs(y-y_),correct_threshold)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))*100
    print('Correct on ' + str(sess.run(accuracy, feed_dict={x: x_data_test, y_: y_data_test}))[:5] + '% of examples')
    sess.close()

plt.figure()
plt.plot(range(dataset_sizes),final_cost_output_train,'k',label='train')
plt.plot(range(dataset_sizes),final_cost_output_test,'r',label='test')
plt.xlabel('examples subtracted from dataset (higher = fewer training examples)')
plt.ylabel('final cost')
plt.legend()
