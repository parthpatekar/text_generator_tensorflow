from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections

# Text file containing words for training
training_file = 'belling_the_cat.txt'

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content

training_data = read_data(training_file)
print("Loaded training data...")

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)

# Parameters
learning_rate = 0.001
training_iters = 50000
display_step = 1000
batch_size = 3

# number of units in RNN cell
n_hidden = 128

# tf Graph input
x = tf.placeholder("float", [None, batch_size, 1])
y = tf.placeholder("float", [None, vocab_size])

weights = tf.Variable(tf.random_normal([n_hidden, vocab_size]))
biases = tf.Variable(tf.random_normal([vocab_size]))

def RNN(x, weights, biases):
    #bring in shape
    x = tf.reshape(x, [-1, batch_size])
    x = tf.split(x,batch_size,1)

    #initialize lstm-rnn
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(batch_size),rnn.BasicLSTMCell(n_hidden)])
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    #return last of the produced output and multiply with weights and add biases
    return tf.matmul(outputs[-1], weights) + biases

pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0,batch_size+1)
    end_offset = batch_size + 1
    acc_total = 0
    loss_total = 0

    while step < training_iters:
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, batch_size+1)

        #
        X_train = np.array([np.array([[dictionary[ str(training_data[i])]] for i in range(offset, offset+batch_size)])])

        y_train_onehot = np.zeros([vocab_size], dtype=float)        
        y_train_onehot[dictionary[str(training_data[offset+batch_size])]] = 1.0
        y_train_onehot=np.array([y_train_onehot])

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: X_train, y: y_train_onehot})
        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0            
        step += 1
        offset += (batch_size+1)
    print("Optimization Finished!")

    while True:
        prompt = "%s words: " % batch_size
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != batch_size:
            continue
        try:
            len_words=len(words)
            X_train = [dictionary[str(words[i])] for i in range(len_words)]
            words_generated=32
            for i in range(words_generated):
                keys = np.reshape(np.array(X_train), [-1, batch_size, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
                X_train = X_train[1:]
                X_train.append(onehot_pred_index)
            print(sentence)
        except:
            print("Word not in dictionary")
