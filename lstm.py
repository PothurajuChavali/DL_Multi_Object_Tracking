#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


class VanillaLstm(object):
    """
    Long Short-Term Memory Nerual Net Base Class
    """
    def __init__(self, num_in=2, num_hid=10, num_out=2, batch_size=1):
        self._num_in = num_in
        self._num_hid = num_hid
        self._num_out = num_out
        self._batch_size = batch_size

        with tf.name_scope("Init_State"):
            self._init_memory = tf.zeros(shape=[self._batch_size, self._num_hid], dtype=tf.float32)
            self._init_state = tf.zeros(shape=[self._batch_size, self._num_hid], dtype=tf.float32)
            self._init_memory_state = tf.stack([self._init_memory, self._init_state])

        with tf.variable_scope("RNNWeights"):
            with tf.variable_scope("I_Gate"):
                self._U_xi = tf.get_variable("XtoIGate_U", shape=[self._num_in, self._num_hid], dtype=tf.float32,
                                             initializer=xavier_initializer())
                self._W_si = tf.get_variable("StoIGate_W", shape=[self._num_hid, self._num_hid], dtype=tf.float32,
                                             initializer=xavier_initializer())
                self._W_ci = tf.get_variable("CtoIGate_W", shape=[self._num_hid, self._num_hid], dtype=tf.float32,
                                             initializer=xavier_initializer())
                self._b_i = tf.get_variable("IGate_B", shape=[self._num_hid], dtype=tf.float32,
                                            initializer=xavier_initializer())

            with tf.variable_scope("O_Gate"):
                self._U_xo = tf.get_variable("XtoOGate_U", shape=[self._num_in, self._num_hid], dtype=tf.float32,
                                             initializer=xavier_initializer())
                self._W_so = tf.get_variable("StoOGate_W", shape=[self._num_hid, self._num_hid], dtype=tf.float32,
                                             initializer=xavier_initializer())
                self._W_co = tf.get_variable("CtoOGate_W", shape=[self._num_hid, self._num_hid], dtype=tf.float32,
                                             initializer=xavier_initializer())
                self._b_o = tf.get_variable("OGate_B", shape=[self._num_hid], dtype=tf.float32,
                                            initializer=xavier_initializer())

            with tf.variable_scope("F_Gate"):
                self._U_xf = tf.get_variable("XtoFGate_U", shape=[self._num_in, self._num_hid], dtype=tf.float32,
                                             initializer=xavier_initializer())
                self._W_sf = tf.get_variable("StoFGate_W", shape=[self._num_hid, self._num_hid], dtype=tf.float32,
                                             initializer=xavier_initializer())
                self._W_cf = tf.get_variable("CtoFGate_W", shape=[self._num_hid, self._num_hid], dtype=tf.float32,
                                             initializer=xavier_initializer())
                self._b_f = tf.get_variable("FGate_B", shape=[self._num_hid], dtype=tf.float32,
                                            initializer=xavier_initializer())

            with tf.variable_scope("G_Gate"):
                self._U_xg = tf.get_variable("XtoGGate_U", shape=[self._num_in, self._num_hid], dtype=tf.float32,
                                             initializer=xavier_initializer())
                self._W_sg = tf.get_variable("StoGGate_W", shape=[self._num_hid, self._num_hid],
                                             dtype=tf.float32,
                                             initializer=xavier_initializer())
                self._b_g = tf.get_variable("GGate_B", shape=[self._num_hid], dtype=tf.float32,
                                            initializer=xavier_initializer())

    def recurrence(self, c_s_tm1, x_t):
        with tf.name_scope("LSTM_Cell"):
            c_tm1, s_tm1 = tf.unstack(c_s_tm1)

            with tf.variable_scope("InputGate"):
                i_t = tf.sigmoid(tf.matmul(x_t, self._U_xi) + tf.matmul(c_tm1, self._W_ci) +
                                 tf.matmul(s_tm1, self._W_si) + self._b_i)
            with tf.variable_scope("OutputGate"):
                o_t = tf.sigmoid(tf.matmul(x_t, self._U_xo) + tf.matmul(c_tm1, self._W_co) +
                                 tf.matmul(s_tm1, self._W_si) + self._b_o)
            with tf.variable_scope("ForgetGate"):
                f_t = tf.sigmoid(tf.matmul(x_t, self._U_xf) + tf.matmul(c_tm1, self._W_cf) +
                                 tf.matmul(s_tm1, self._W_sf) + self._b_f)
            with tf.variable_scope("GateGate"):
                g_t = tf.tanh(tf.matmul(x_t, self._U_xg) + tf.matmul(s_tm1, self._W_sg) + self._b_g)
            with tf.variable_scope("Memory"):
                c_t = tf.multiply(c_tm1, f_t) + tf.multiply(g_t, i_t)
            with tf.variable_scope("State"):
                s_t = tf.multiply(tf.tanh(c_t), o_t)

            return tf.stack([c_t, s_t])


class Lstm(VanillaLstm):
    """
    Long Short-Term Memory Nerual Net with a single LSTM Cell at each recurrence step.
    """

    def __init__(self, num_in=2, num_hid=10, num_out=2, batch_size=1):
        super().__init__(num_in=num_in, num_hid=num_hid, num_out=num_out, batch_size=batch_size)

        with tf.variable_scope("RNNWeights"):
            with tf.variable_scope("EOut"):
                self._V_py = tf.get_variable("PStoEOut_V", shape=[self._num_hid, self._num_out], dtype=tf.float32,
                                             initializer=xavier_initializer())
                self._b_y = tf.get_variable("EOut_B", shape=[self._num_out], dtype=tf.float32,
                                            initializer=xavier_initializer())

    def predict_sequence(self, features):
        with tf.name_scope('Sequence'):
            c_s_t = tf.scan(self.recurrence, tf.transpose(features, [1, 0, 2]),
                            initializer=self._init_memory_state)
            c_t, s_t = tf.unstack(c_s_t, axis=1)

        with tf.name_scope('State'):
            states = tf.transpose(s_t, [1, 0, 2], name="States")

        with tf.name_scope('Memory'):
            memories = tf.transpose(c_t, [1, 0, 2], name="Memories")

        with tf.name_scope('Output'):
            y_all = tf.add(tf.matmul(tf.reshape(s_t, [-1, self._num_hid]), self._V_py), self._b_y)
            y_t = tf.reshape(y_all, [-1, s_t.shape[1], self._num_out])
            logits = tf.transpose(y_t, [1, 0, 2], name="Logits")
            predictions = tf.nn.softmax(logits, name="Predictions")
            outputs = tf.argmax(predictions, axis=2, name="Outputs")

        return memories, states, logits, predictions, outputs



