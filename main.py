# -*- coding: UTF-8 -*-
# tensorflow version 1.0

import tensorflow as tf

from collections import namedtuple
import numpy as np
import timeit
import random

#import os
import model
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

start = timeit.default_timer()
# global variables
num_gpus = 1

train_num = 6000
test_num = 6000

bound = test_num // 15
HParams = namedtuple('HParams',
                     'batch_size, num_classes, start_lrn_rate, decay_rate, feature_row, feature_col, channels, '
                     'is_training, num_residual_units, weight_decay_rate, BN_decay, optimizer')
hps = HParams(batch_size=16,  # 'number of samples in each iteration',
              num_classes=2,  # 'binary classfication',
              start_lrn_rate=0.001,  # 'starting learning rate',
              decay_rate=0.95,  # 'decaying rate of learning rate ',
              feature_row=256,
              feature_col=256,
              channels=4,   # 'number of initial channel'
              is_training=True,  # "is training or not."
              num_residual_units=5,  # 'number of residual unit in each different residual module',
              weight_decay_rate=0.0002, # 'decaying rate of weight '
			  BN_decay=0.9997
              optimizer='adam')  # "optimizer, 'adam' or 'sgd', default to 'adam'."

# Read file name and convert all file name into a name
# list including cover and stego.
def read_filename():
    path = './{folder}/{n}.txt'
    folder = ['sign_0.1', 'sign_0.2', 'sign_0.3', 'sign_0.5', 'sign_1.0', 'min_0.1', 'min_0.2',
              'min_0.3', 'min_0.5','min_1.0', 'lsb_ee_0.1', 'lsb_ee_0.2', 'lsb_ee_0.3', 'lsb_ee_0.5', 'lsb_ee_1.0', 'cover']
    coef_list = []

    for i in np.arange(0, (train_num) // 2):
        filepath = path.format(folder=folder[15], n=i + 1)
        coef_list.append(filepath)

    random.shuffle(coef_list)
    print('Shuffle filename complete')

    name_list = []
    for i in range(len(coef_list)):
        cover_name = coef_list[i]
        name_list.append(cover_name)
        if i % 3 == 0:
            stego_name = cover_name.replace(folder[15], folder[2])
        elif i % 3 == 1:
            stego_name = cover_name.replace(folder[15], folder[7])
        elif i % 3 == 2:
            stego_name = cover_name.replace(folder[15], folder[12])
        name_list.append(stego_name)
    train_name = name_list

    test_name = []
    for i in np.arange(train_num//2, train_num//2+test_num//2):
        filepath = path.format(folder=folder[15], n=i + 1)
        test_name.append(filepath)
        tmp = filepath.replace(folder[15],folder[(i-train_num//2)/(bound//2)])
        test_name.append(tmp)
    return train_name,test_name

# read train data from file name list
def read_train_data():
    train_name, _ = read_filename()

    train_data = []
    for i in range(len(train_name)):
        f = open(train_name[i])
        list_tmp = []
        for line in f.readlines()[:hps.feature_row]:
            lines = line.strip().split('\t')
            tmp_tmp_list = []
            for x in lines[:hps.feature_col]:
                tmp_tmp_list.append(round(float(x)))
            list_tmp.append(tmp_tmp_list)
        train_data.append(list_tmp)
    print('Read train set complete')

    train_labels_data = [0, 1] * (train_num // 2)
    train_data = np.asarray(train_data)
    train_labels_data = np.asarray(train_labels_data)

    return train_data, train_labels_data

# read train data from file name list
def read_test_data():
    _, test_name = read_filename()

    test_data = []
    for i in range(len(test_name)):
        f = open(test_name[i])


        list_tmp_1 = []
        for line in f.readlines()[:hps.feature_row]:
            lines = line.strip().split('\t')
            tmp_tmp_list = []
            for x in lines[:hps.feature_col]:
                tmp_tmp_list.append(round(float(x)))
            list_tmp_1.append(tmp_tmp_list)
        test_data.append(list_tmp_1)

    print('Read test set complete')

    test_labels_data = [0, 1]*(test_num // 2)
    test_data = np.asarray(test_data)
    test_labels_data = np.asarray(test_labels_data)

    return test_data, test_labels_data

#shuffle train set before each epochs begin
def shuffle(data):
    data_len = len(data)
    data_cpy = copy.copy(data)
    tmp0 = np.arange(data_len / 2, dtype=np.int64)
    np.random.shuffle(tmp0)
    tmp1 = np.reshape(tmp0, (-1, 1))
    tmp1 = tmp1 * 2
    tmp2 = tmp1 + 1
    tmp3 = np.concatenate((tmp1, tmp2), axis=1)
    perm = np.reshape(tmp3, (data_len,))
    for i in range(data_len):
        data[i] = data_cpy[perm[i]]

#divide original train set into trian set and validation set
def divide_train_set(train_files, train_labels):
    proportion = 0.8# division ratio

    shuffle(train_files)

    new_train_files = train_files[: int(len(train_files) * proportion)]
    new_train_labels = train_labels[: int(len(train_files) * proportion)]

    valid_files = train_files[int(len(train_files) * proportion):]
    valid_labels = train_labels[int(len(train_files) * proportion):]

    return new_train_files, new_train_labels, valid_files, valid_labels


# Compute running time
def compute_time():
    stop = timeit.default_timer()
    seconds = stop - start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    print('Run time: %d:%02d:%02d:%02d' % (d, h, m, s))

#calculate the prediction accuracy
def  get_acc(sess, correct_count, mdct_placeholder, labels_placeholder, mdct_data, labels_data):
    batch_size = hps.batch_size

    true_count = 0
    steps_per_epoch = len(mdct_data)//batch_size
    num_examples = steps_per_epoch*batch_size

    for step in xrange(steps_per_epoch):
        reshaped_label = np.reshape(labels_data, [len(mdct_data), 1])
        feed_dict = {mdct_placeholder:mdct_data[step*batch_size:step*batch_size+batch_size],
                     labels_placeholder:reshaped_label[step*batch_size:step*batch_size+batch_size]}
        true_count += correct_count.eval(feed_dict=feed_dict, session=sess)

    accuracy = float(true_count)/num_examples

    return accuracy

def main():
    loss_list = []
    valid_acc_list = []
#    max_acc = 0
    if hps.is_training == True:
        mdct_placeholder = tf.placeholder(tf.int32, [hps.batch_size, hps.feature_row, hps.feature_col], name='data')
        labels_placeholder = tf.placeholder(tf.int32, [hps.batch_size, 1], name='labels')
        model = test_model.model(hps, mdct_placeholder, labels_placeholder)
        model.build_model()
        print('Construct model complete')
        sess = tf.Session()
        batch_index = 0
        train_data, train_labels_data = read_train_data()
        train_datas, train_labels, valid_datas, valid_labels = divide_train_set(train_data, train_labels_data)
        sess.run(tf.global_variables_initializer())
        for j in xrange(100000):
            start = batch_index
            batch_index += hps.batch_size

            if batch_index > len(train_datas):
                shuffle(train_datas)
                start = 0
                batch_index = hps.batch_size

            end = batch_index
            data_batch = train_datas[start:end]
            labels_batch = train_labels[start:end]
            liu_temp = np.reshape(labels_batch, [hps.batch_size,1])
            _, decay, loss_value, lr, step, cnt = sess.run([model.train_op, model._decay(), model.cost, model.learning_rate, model.global_step, model.correct_count],
                    feed_dict={mdct_placeholder: data_batch, labels_placeholder: liu_temp})
            if step % 10 == 0:
                print('Tensorflow Info : step {:d}, lr {:.8f}, loss {:.8f}, correct {:d}'.format(
                    step, lr, loss_value, cnt
                ))
                loss_list.append(loss_value)
                if step % 300 == 0:
                    valid_acc = get_acc(sess, model.correct_count,mdct_placeholder,labels_placeholder,valid_datas,valid_labels)
                    tp_cont_acc = get_acc(sess, model.tp_count, mdct_placeholder, labels_placeholder,valid_datas,valid_labels)
                    tn_cont_acc = get_acc(sess, model.tn_count, mdct_placeholder, labels_placeholder,valid_datas,valid_labels)
                    print('Tensorflow valid accuracy {:.8f}'.format(valid_acc))
                    print('Tensorflow tp_cont_acc : accuracy {:.8f}'.format(tp_cont_acc))
                    print('Tensorflow tn_cont_acc : accuracy {:.8f}'.format(tn_cont_acc))
                    valid_acc_list.append(valid_acc)
                    saver = tf.train.Saver(max_to_keep=5)
                    if valid_acc > max_acc:
                        max_acc = valid_acc
                        saver_path = saver.save(sess, './checkpoint/{}.ckpt'.format(step))
                        print("Model saved in file:", saver_path)
                    if step >= 10000:
                        file = open('./loss.txt', 'w')
                        for length in range(0, len(loss_list), 1):
						    file.write(str(length+1))
                            file.write('\t')
                            file.write(str(loss_list[length]));
                            file.write('\n')
                        file.close()
                        file = open('./valid_acc.txt', 'w')
                        for length in range(0, len(valid_acc_list), 1):
							file.write(str(length+1))
                            file.write('\t')
                            file.write(str(valid_acc_list[length]));
                            file.write('\n')
                        file.close()
                        break;
        sess.close()
    elif hps.is_training == False:
        feature = []
        test_data, test_labels_data = read_test_data()
        sess = tf.Session()
        #restore trained model from saved meta graph.
        saver = tf.train.import_meta_graph('./checkpoint/1800.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
        graph = tf.get_default_graph()
        mdct_placeholder = graph.get_tensor_by_name('data:0')
        labels_placeholder = graph.get_tensor_by_name('labels:0')
        feature_map = graph.get_tensor_by_name('unit_last/feature:0')
        correct_count = graph.get_tensor_by_name('acc:0')
        tp_count = graph.get_tensor_by_name('tp:0')
        tn_count = graph.get_tensor_by_name('tn:0')
        for i in range(0, 15):
            acc_cnt = 0
            tp_acc_cnt = 0
            tn_acc_cnt = 0
            data = test_data[bound * i:bound * (i + 1)]
            labels = test_labels_data[bound * i:bound * (i + 1)]
            print('---------test_data_{:.2f}--------'.format(i + 1))
            for step in xrange(bound//hps.batch_size):
                reshaped_label = np.reshape(labels, [len(data), 1])
                feed_dict = {mdct_placeholder: data[step * hps.batch_size:step * hps.batch_size + hps.batch_size],
                             labels_placeholder: reshaped_label[step * hps.batch_size:step * hps.batch_size + hps.batch_size]}
                acc_cnt += sess.run(correct_count, feed_dict=feed_dict)
                tp_acc_cnt += sess.run(tp_count, feed_dict=feed_dict)
                tn_acc_cnt += sess.run(tn_count, feed_dict=feed_dict)
                batch_feature = sess.run(feature_map,feed_dict=feed_dict)
                feature.extend(batch_feature.tolist())
            total = (bound / hps.batch_size) * hps.batch_size
            acc = float(acc_cnt) / total
            tp_acc = float(tp_acc_cnt)*2 / total
            tn_acc = float(tn_acc_cnt)*2 / total
            print('Tensorflow test_accuracy : accuracy {:.8f}'.format(acc))
            print('Tensorflow tp_accuracy : accuracy {:.8f}'.format(tp_acc))
            print('Tensorflow tn_accuracy : accuracy {:.8f}'.format(tn_acc))
        sess.close()
        array = np.asarray(feature)
        np.savetxt('./feature.txt', array, fmt='%.6f')
if __name__ == '__main__':
    tf.app.run()
