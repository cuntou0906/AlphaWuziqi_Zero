# encoding:utf-8

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt
########################################################################################################################
# 神经网络预测
# 输入： 当前棋盘的状态+当前下棋的玩家、
# 输出： V + A   V 为当前棋局可能胜率-价值  A 为 当前棋局下一步落子的概率
########################################################################################################################


from Wuziqi_constant import board_hight
from Wuziqi_constant import board_width
x_dim1 = board_hight* board_width + 1  # 输入维度 (棋盘状态 + 当前落子的玩家)
y_dim1 = board_hight* board_width + 1  # 输入维度 (所有动作的可能性 + 胜率)

class BPnet(object):
    def __init__(self,lr=0.005,x_dim = x_dim1,y_dim= y_dim1,sess = None):
        self.n_input=x_dim
        self.n_output =y_dim
        self.learing_rate =lr
        self.path = 'model/predict.ckpt'
        np.random.seed(1)  # 随机种子
        tf.set_random_seed(1)
        self._build_net()

        if sess is None:
            self.sess = tf.Session()  # 初始化session
            self.sess.run(tf.global_variables_initializer())  # 初始化全局变量
        else:
            self.sess = sess  # session赋值

        self.saver = tf.train.Saver()


    def _build_net(self):
        def build_layers(s, c_names, n_l1, n_l2, w_initializer, b_initializer):  # 四层BP网络
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_input, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w2', [n_l2, self.n_output], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b2', [1, self.n_output], initializer=b_initializer, collections=c_names)
                Q_value = tf.matmul(l2, w3) + b3

            return Q_value

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_input], name='s')  # 输入
        self.q_target = tf.placeholder(tf.float32, [None, self.n_output], name='Q_target')  # 标签值

        with tf.variable_scope('eval_net'):
            c_names, n_l1, n_l2,  w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 200, 200, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # 参数配置

            self.q_eval = build_layers(self.s, c_names, n_l1, n_l2,  w_initializer, b_initializer)  # 建立BP网络

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))  # 损失函数
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.learing_rate).minimize(self.loss)  # 优化器

    def learn(self,x_data,y_data):
        for i in range(50):
            self.sess.run(self._train_op, feed_dict={self.s: x_data, self.q_target: y_data})
            pass
        self.save()
        pass

    def predict(self, x_data):
        #self.restore(self.path)
        self.prediction_value = self.sess.run(self.q_eval, feed_dict={self.s: x_data})
        return self.prediction_value

    def save(self):
        self.saver.save(self.sess,self.path)

    def restore(self):
        self.saver.restore(self.sess, self.path)

    pass

