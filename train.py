import tensorflow as tf
# import matplotlib.pyplot as plt
import logging
import time
import os

import model as md
import data_factory

logging.basicConfig(level=logging.DEBUG)

#def train(batch_size, num_epoch, pretrain, save_per_epoch, train_pixels, train_labels, val_pixels, val_labels, model_name=None):
def train(train_pixels, train_labels, val_pixels, val_labels):
    epoch = 10000
    batch_size = 100
    num_of_batch_per_epoch = int(len(train_pixels) / batch_size)

    display_epoch = 1
    max_tolerance_epoch = 801

    model = md.HelloCNN()
    ground_truth = tf.placeholder(tf.float32, [None, 7])
    ground_truth_depth = tf.placeholder(tf.int32)
    #prepare loss and optimizer op
    # onehot_labels = tf.one_hot(indices=tf.cast(ground_truth, tf.int32), depth=ground_truth_depth)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=ground_truth, logits=model.output)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.005)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    # prepare accuracy op
    test_label = tf.placeholder(tf.int32, [None, 7])
    correct_prediction = tf.equal(tf.argmax(test_label, 1), tf.argmax(model.output, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    epoch_set = []
    train_accuracy_set = []
    val_accuracy_set = []

    best_val_accuracy = 0
    tolerance_counter = 0

    checkpoint_epoch = 10
    checkpoint_dir = os.getcwd() + '/tmp/tfsaver1/'
    directory = os.path.dirname(checkpoint_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    saver = tf.train.Saver()


    isTrain = True

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    isCkpt = ckpt and ckpt.model_checkpoint_path

    isRestore = isTrain and isCkpt

    global_step_base = tf.Variable(1000, trainable=False, name='global_step_base')
    global_step_var = tf.Variable(1000, trainable=False, name='global_step_counter')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if isRestore:
            # global_step.assign(1000)
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step_base = global_step_var
            print('restore step {}'.format(global_step_var.eval()))
        else:
            pass

        if isTrain :
            summary_dir = os.getcwd() + '/tmp/tfsaver1_summary/'
            print('summary_dir {}'.format(summary_dir))
            summary_directory = os.path.dirname(summary_dir)
            if not os.path.exists(summary_directory):
                os.makedirs(summary_directory)
            writer = tf.summary.FileWriter(summary_directory, sess.graph)
            loss_summary_op = tf.summary.scalar("loss", loss)
            val_accuracy_summary_op = tf.summary.scalar("val_accuracy", accuracy_op)
            train_accuracy_summary_op = tf.summary.scalar("train_accuracy", accuracy_op)

            # merged_summary = tf.summary.merge_all()

            start_t = time.time()
            for i in range(epoch):
                logging.debug("epoch i {}".format(i))
                data_factory.shuffle(train_pixels, train_labels)

                for n in range(num_of_batch_per_epoch):
                    batch_train_pixels = train_pixels[n * batch_size : (n + 1) * batch_size]
                    batch_train_label = train_labels[n * batch_size : (n + 1) * batch_size]

                    _, loss_summary = sess.run([train_op, loss_summary_op], feed_dict={model.x: batch_train_pixels, ground_truth: batch_train_label})
                writer.add_summary(loss_summary, i)


                if i % display_epoch == 0:
                    epoch_set.append(i)

                    train_accuracy, train_accuracy_summary = sess.run([accuracy_op, train_accuracy_summary_op], feed_dict={model.x: train_pixels, test_label: train_labels})
                    train_accuracy_set.append(train_accuracy)

                    val_accuracy, val_accuracy_summary = sess.run([accuracy_op, val_accuracy_summary_op], feed_dict={model.x: val_pixels, test_label: val_labels})
                    val_accuracy_set.append(val_accuracy)

                    writer.add_summary(train_accuracy_summary, i)
                    writer.add_summary(val_accuracy_summary, i)

                    logging.debug("epoch i {}, train {}, val {}".format(i, train_accuracy, val_accuracy))

                if (i + 1) % checkpoint_epoch == 0:
                    global_step_var = global_step_base + i
                    saver.save(sess, checkpoint_dir + 'model.ckpt', global_step = global_step_var)

                if best_val_accuracy < val_accuracy:
                    best_val_accuracy = val_accuracy
                    tolerance_counter = 0
                else:
                    tolerance_counter += 1

                if tolerance_counter > max_tolerance_epoch:
                    logging.debug('no enhancement in epoch {}, best_val_accuracy{}'.format(i + 1, best_val_accuracy))
                    break

            writer.flush()

            logging.debug('Elapsed time in epoch ' + str(i + 1) + ': ' + str(time.time() - start_t))
        else :

            pass


    # plt.plot(epoch_set, val_accuracy_set, 'o', label='val')
    # plt.plot(epoch_set, train_accuracy_set, 'o', label='train')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.legend()
    # plt.show()

    print("val_accuracy {}".format(zip(epoch_set, val_accuracy_set)))
    print("train_accuracy {}".format(zip(epoch_set, train_accuracy_set)))
        # prediction = model
        # accuracy, _ = sess.run(tf.metrics.accuracy(labels=val_labels, predictions=model.out))

if __name__=='__main__':
    pixels, labels = data_factory.load_train_data()
    data_factory.shuffle(pixels, labels)
    train_pixels, train_labels, val_pixels, val_labels = data_factory.separate(pixels, labels, int(len(pixels)*0.2))
    train(train_pixels, train_labels, val_pixels, val_labels)
