import argparse
import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import AlexNet
from datagenerator import ImageDataGenerator

train_file = 'train.txt'        # File containing the paths for the training images
val_file = 'val.txt'            # File containing the paths for the validation images

# Network params
num_classes = 2                 # Number of classes to finetune the model for prediction
num_epochs = 3                  # Number of epochs to run for the training
input_size = 227                # Assuming it to be RGB
train_layers = ['fc8', 'fc7']   # The layers that we want to train

display_step = 1                # How often tf.summary is written

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = '/tmp/finetune_alexnet/dogs_vs_cats'
checkpoint_path = '/tmp/finetune_alexnet/'


def train(batch_size, learning_rate, conv, fc, dropout_rate, additional):
    x = tf.placeholder(tf.float32, [batch_size, input_size, input_size, 3])
    y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    # If lesser number of convolutions are to be used, the first fully connected
    # layer weights also need to be learned
    if conv < 5:
        train_layers.append('fc6')

    # Load the model with the desired parameters
    model = AlexNet(x, keep_prob, num_classes, train_layers, fc, conv, additional)
    score = model.fc8

    # List of trainable variables of the layers we want to train
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in
                train_layers]

    # Op for calculating the loss
    with tf.name_scope('cross_entropy'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                              labels=y))

    # Train op
    with tf.name_scope('train'):
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    # Add gradients to summary
    for gradient, var in gradients:
        tf.summary.histogram(var.name + '/gradient', gradient)

    # Add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name, var)

    # Add the loss to the summary
    tf.summary.scalar('cross_entropy', loss)

    # Op for the accuracy of the model
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar('accuracy', accuracy)

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)

    saver = tf.train.Saver()

    train_generator = ImageDataGenerator(train_file, horizontal_flip=True,
                                         shuffle=True)
    val_generator = ImageDataGenerator(val_file, shuffle=False)

    # Get the number of training / validation steps per epoch
    train_batches_per_epoch = np.floor(train_generator.data_size /
                                       batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(int)

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.import_meta_graph('/tmp/finetune_alexnet/model_epoch0.ckpt.meta')

    # saver.restore(sess, '/tmp/finetune_alexnet/model_epoch0.ckpt')
    
    # Comment this line and uncomment the above ones to reuse the model after a checkpoint
    # for faster training
    model.load_initial_weights(sess)
    print("{} Restored model...".format(datetime.now()))

    test_acc_prev = 0
    test_count = 0
    for _ in range(val_batches_per_epoch):
        batch_tx, batch_ty = val_generator.next_batch(batch_size)
        acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, keep_prob: 1.})
        test_acc_prev += acc
        test_count += 1

    test_acc_prev /= test_count
    test_acc = test_acc_prev
    print("{} Initial Validation Accuracy = {:.4f}".format(datetime.now(), test_acc_prev))
    writer.add_graph(sess.graph)

    print('{} Start training...'.format(datetime.now()))
    print('{} Open TensorBoard at --logdir {}'.format(datetime.now(),
          filewriter_path))

    for epoch in range(num_epochs):
        print('{} Epoch number: {}'.format(datetime.now(), epoch + 1))

        step = 1

        while step < train_batches_per_epoch:
            print('{} Step number: {}'.format(datetime.now(), step))
            batch_xs, batch_ys = train_generator.next_batch(batch_size)
            sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys,
                                          keep_prob: dropout_rate})
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: batch_xs,
                             y: batch_ys, keep_prob: dropout_rate})
                writer.add_summary(s, epoch * train_batches_per_epoch + step)

            step += 1
    
        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        test_acc = 0.
        test_count = 0
        for ind in range(val_batches_per_epoch):
            print('{} Valid batch number: {}'.format(datetime.now(), ind))
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            acc = sess.run(accuracy, feed_dict={x: batch_tx,
                                                y: batch_ty,
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))

        if test_acc > test_acc_prev:
            print("{} Saving checkpoint of model...".format(datetime.now()))

            #save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch)+'.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

            if abs(test_acc - test_acc_prev) < 0.05:
                print("Early stopping.... exiting")
                break
    
            test_acc_prev = test_acc

        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()
    return test_acc


def plot(x, y, name):
    fig, ax = plt.subplots(1, 1)
    plt.plot(x, y, 'b')
    fig.savefig(name + '.png')
    plt.close(fig)


def plot_scatter(x, y, name, labels):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, label in enumerate(labels):
        ax.annotate(label, (x[i], y[i]))
    plt.savefig(name + '.png')


def main(_):
    if tf.gfile.Exists(filewriter_path):
        tf.gfile.DeleteRecursively(filewriter_path)
    tf.gfile.MakeDirs(filewriter_path)

    train(FLAGS.batch_size, FLAGS.learning_rate, FLAGS.conv, FLAGS.fc,
          FLAGS.dropout_rate, FLAGS.additional)

    # During varying different parameters, when 'conv' or 'fc' is varied,
    # the pre-trained model must be loaded, whereas for the other cases,
    # the model checkpoint updated each epoch must be included, because here
    # the architecture doesn't change and hence, we can use the same weights
    # as before to speed up the training process.
    
    # # Vary number of convolutional layers
    # accuracies = []
    # convs = [4, 5]
    # for conv in convs:
    #     accuracies.append(train(FLAGS.batch_size, FLAGS.learning_rate, conv, FLAGS.fc,
    #                             FLAGS.dropout_rate, FLAGS.additional))
    #     tf.reset_default_graph()

    # plot_scatter(convs, accuracies, 'Changing number of convolutions', ['Conv=4', 'Conv=5'])

    # # Vary number of neurons in the fully connected layers
    # accuracies = []
    # fcs = range(1000, 5000, 1000)
    # for fc in fcs:
    #     accuracies.append(train(FLAGS.batch_size, FLAGS.learning_rate, FLAGS.conv, fc,
    #                             FLAGS.dropout_rate, FLAGS.additional))
    #     tf.reset_default_graph()

    # plot(fcs, accuracies, 'Changing the fully connected layer')

    # Vary the keep probability to be used during dropout
    # accuracies = []
    # keep_probs = np.arange(0.1, 1, 0.2)
    # for keep_prob in keep_probs:
    #      accuracies.append(train(FLAGS.batch_size, FLAGS.learning_rate, FLAGS.conv, FLAGS.fc,
    #                              keep_prob, FLAGS.additional))
    #      tf.reset_default_graph()

    # plot(keep_probs, accuracies, 'Changing the dropout rate')

    # Check the two instances when an additional shallow layers is used or not
    # accuracies = [0.98]
    # choices = [1]
    # for choice in choices:
    #     accuracies.append(train(FLAGS.batch_size, FLAGS.learning_rate, FLAGS.conv, FLAGS.fc,
    #                             FLAGS.dropout_rate, choice))
    #     tf.reset_default_graph()

    # choices = [0, 1]
    # plot_scatter(choices, accuracies, 'With or Without additional layers', ['No additional', 'Using additional'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--conv',
        type=int,
        default=5,
        help='Number of convolutional layers in the AlexNet to be used (max=5).'
    )
    parser.add_argument(
        '--fc',
        type=int,
        default=4096,
        help='Number of neurons in the fully connected layer of the model.'
    )
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.5,
        help='The keep probability for neurons during dropout.'
    )
    parser.add_argument(
        '--additional',
        type=int,
        default=0,
        help='If value=1, an additional shallow network is added before the fully connected layer (only if conv=5).'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
