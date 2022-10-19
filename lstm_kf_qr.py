#!/usr/bin/env python3

# ******************************************    Libraries to be imported    ****************************************** #
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from kalman_tf import KalmanTF
from lstm import VanillaLstm


# ******************************************    Class Declaration Start     ****************************************** #
class LSTM_KF(object):
    """
    LayerTemplate Class.
    Implements the Neural Network / Tensorflow Compute Graph Definition.
    """

    def __init__(self, features, num_in=2, num_hid=10, num_out=2, batch_size=1, nn_scope='lstm_kf'):
        """
        Constructor of the LayerTemplate Class.

        :param tf.Tensor features: A Tensor / Placeholder containing the input to the neural network.
        :param int num_out: Number of neurons in the Logits Layer.
        :param str nn_scope: String representing the variable used for creating the Layer.
        """

        self._nn_scope = nn_scope

        with tf.variable_scope(self._nn_scope):
            self._features = features
            self._num_in = num_in
            self._num_hid = num_hid
            self._num_out = num_out
            self._batch_size = batch_size

            self._kalman_filter = KalmanTF(kf_num_in=num_in, kf_num_out=num_out, dt=dt, batch_size=batch_size)

            with tf.variable_scope("RNNWeightsR"):
                self.lstm_R = VanillaLstm(self._features, self._num_hid)
            with tf.variable_scope("RNNWeightsQ"):
                self.lstm_Q = VanillaLstm(self._features, self._num_hid)

            with tf.name_scope("Init_State"):
                self._init_memory = tf.zeros(shape=[self._batch_size, self._num_hid], dtype=tf.float32)
                self._init_state = tf.zeros(shape=[self._batch_size, self._num_hid], dtype=tf.float32)

                self._y_state = tf.transpose(self._kalman_filter._init_state)
                self._P_state = self._kalman_filter._init_covariance
                self._P_state = tf.reshape(self._P_state, [1, self._num_out * self._num_out])
                self._R_t = tf.get_variable("R_t", initializer=tf.zeros([1, self._num_in], dtype=tf.float32))
                self._Q_t = tf.get_variable("Q_t", initializer=tf.zeros([1, self._num_out], dtype=tf.float32))

                self._init_memory_state = tf.concat(
                    [self._init_memory, self._init_state, self._init_memory, self._init_state, self._y_state,
                     self._P_state, self._R_t, self._Q_t], axis=1)

            with tf.variable_scope("RNNWeightsE"):
                with tf.variable_scope("EOut"):
                    self._VQ_py = tf.get_variable("PStoEOut_VQ", shape=[self._num_hid, self._num_out], dtype=tf.float32,
                                                  initializer=xavier_initializer())
                    self._bQ_y = tf.get_variable("EOut_BQ", shape=[self._num_out], dtype=tf.float32,
                                                 initializer=xavier_initializer())
                    self._VR_py = tf.get_variable("PStoEOut_VR", shape=[self._num_hid, self._num_in], dtype=tf.float32,
                                                  initializer=xavier_initializer())
                    self._bR_y = tf.get_variable("EOut_BR", shape=[self._num_in], dtype=tf.float32,
                                                 initializer=xavier_initializer())


    def recurrence(self, c_s_y_p_r_q_tm1, z_t):
        with tf.name_scope("LSTM_KF_Cell"):
            c_q_tm1, s_q_tm1, c_r_tm1, s_r_tm1, y_hat_tm1, p_hat_tm1, _, _ = tf.split(c_s_y_p_r_q_tm1,
                                                                                      [self._num_hid,
                                                                                       self._num_hid,
                                                                                       self._num_hid,
                                                                                       self._num_hid,
                                                                                       self._num_out,
                                                                                       self._num_out
                                                                                       * self._num_out,
                                                                                       self._num_in,
                                                                                       self._num_out], axis=1)

            with tf.variable_scope("LSTM_R"):
                with tf.variable_scope("InputGate"):
                    i_t_r = tf.sigmoid(tf.matmul(z_t, self.lstm_R._U_xi) + tf.matmul(c_r_tm1, self.lstm_R._W_ci) +
                                       tf.matmul(s_r_tm1, self.lstm_R._W_si) + self.lstm_R._b_i)
                with tf.variable_scope("OutputGate"):
                    o_t_r = tf.sigmoid(tf.matmul(z_t, self.lstm_R._U_xo) + tf.matmul(c_r_tm1, self.lstm_R._W_co) +
                                       tf.matmul(s_r_tm1, self.lstm_R._W_si) + self.lstm_R._b_o)
                with tf.variable_scope("ForgetGate"):
                    f_t_r = tf.sigmoid(tf.matmul(z_t, self.lstm_R._U_xf) + tf.matmul(c_r_tm1, self.lstm_R._W_cf) +
                                       tf.matmul(s_r_tm1, self.lstm_R._W_sf) + self.lstm_R._b_f)
                with tf.variable_scope("GateGate"):
                    g_t_r = tf.tanh(
                        tf.matmul(z_t, self.lstm_R._U_xg) + tf.matmul(s_r_tm1,
                                                                      self.lstm_R._W_sg) + self.lstm_R._b_g)
                with tf.variable_scope("Memory"):
                    c_t_r = tf.multiply(c_r_tm1, f_t_r) + tf.multiply(g_t_r, i_t_r)
                with tf.variable_scope("State"):
                    s_t_r = tf.multiply(tf.tanh(c_t_r), o_t_r)

                with tf.variable_scope("Output_LSTM_R"):
                    y_t_r = tf.add(tf.matmul(tf.reshape(s_t_r, [-1, self._num_hid]), self._VR_py), self._bR_y)

                # l_t_r = tf.sigmoid(y_t_r)
                l_t_r = tf.abs(y_t_r)
                r_t = tf.matrix_diag(l_t_r)
                r_t = tf.squeeze(r_t, [0])

                l_t_r = tf.reshape(tf.diag_part(r_t), [1, self._num_in])

            with tf.variable_scope("Kalman_predict_state"):
                self._kalman_filter.predict_state(tf.transpose(y_hat_tm1))
                y_hat_t_p = tf.transpose(self._kalman_filter.get_predicted_state())

            with tf.variable_scope("LSTM_Q"):
                with tf.variable_scope("InputGate"):
                    i_t_q = tf.sigmoid(
                        tf.matmul(y_hat_t_p, self.lstm_Q._U_xi) + tf.matmul(c_q_tm1, self.lstm_Q._W_ci) +
                        tf.matmul(s_q_tm1, self.lstm_Q._W_si) + self.lstm_Q._b_i)
                with tf.variable_scope("OutputGate"):
                    o_t_q = tf.sigmoid(
                        tf.matmul(y_hat_t_p, self.lstm_Q._U_xo) + tf.matmul(c_q_tm1, self.lstm_Q._W_co) +
                        tf.matmul(s_q_tm1, self.lstm_Q._W_si) + self.lstm_Q._b_o)
                with tf.variable_scope("ForgetGate"):
                    f_t_q = tf.sigmoid(
                        tf.matmul(y_hat_t_p, self.lstm_Q._U_xf) + tf.matmul(c_q_tm1, self.lstm_Q._W_cf) +
                        tf.matmul(s_q_tm1, self.lstm_Q._W_sf) + self.lstm_Q._b_f)
                with tf.variable_scope("GateGate"):
                    g_t_q = tf.tanh(
                        tf.matmul(y_hat_t_p, self.lstm_Q._U_xg) + tf.matmul(s_q_tm1, self.lstm_Q._W_sg) +
                        self.lstm_Q._b_g)
                with tf.variable_scope("Memory"):
                    c_t_q = tf.multiply(c_q_tm1, f_t_q) + tf.multiply(g_t_q, i_t_q)
                with tf.variable_scope("State"):
                    s_t_q = tf.multiply(tf.tanh(c_t_q), o_t_q)

                with tf.variable_scope("Output_LSTM_Q"):
                    y_t_q = tf.add(tf.matmul(tf.reshape(s_t_q, [-1, self._num_hid]), self._VQ_py), self._bQ_y)

                # l_t_q = tf.sigmoid(y_t_q)
                l_t_q = tf.abs(y_t_q)
                q_t = tf.matrix_diag(l_t_q)
                q_t = tf.squeeze(q_t, [0])

            with tf.variable_scope("Kalman_predict_covariance"):
                self._kalman_filter.predict_covarince(tf.reshape(p_hat_tm1, [self._num_out, self._num_out]), q_t,
                                                      y_hat_t_p)
                p_hat_t_p = self._kalman_filter.get_predicted_covariance()

            with tf.variable_scope("Kalman_update"):
                self._kalman_filter.update(tf.transpose(y_hat_t_p), p_hat_t_p, tf.transpose(z_t, [1, 0]), r_t)
                y_hat_t = self._kalman_filter.get_updated_state()
                p_hat_t = self._kalman_filter.get_updated_covariance()

            with tf.variable_scope("LSTM_KF_Output"):
                self._y_state = tf.transpose(y_hat_t)
                self._P_state = tf.reshape(p_hat_t, [1, self._num_out * self._num_out])

                c_s_y_p_r_q_t = tf.concat([c_t_q, s_t_q, c_t_r, s_t_r, self._y_state, self._P_state, l_t_r, l_t_q],
                                          axis=1)

            return c_s_y_p_r_q_t

    def predict_sequence(self, features):
        with tf.name_scope('Sequence'):
            c_s_y_p_r_q = tf.scan(self.recurrence, tf.transpose(features, [1, 0, 2]),
                                  initializer=self._init_memory_state)

            c_t_q, s_t_q, c_t_r, s_t_r, y_hat_t, p_hat_t, r_t, q_t = tf.split(c_s_y_p_r_q,
                                                                              [self._num_hid, self._num_hid,
                                                                               self._num_hid, self._num_hid,
                                                                               self._num_out,
                                                                               self._num_out * self._num_out,
                                                                               self._num_in, self._num_out], axis=2)

        with tf.name_scope('State'):
            states_q = s_t_q
            states_r = s_t_r

        with tf.name_scope('Memory'):
            memories_q = c_t_q
            memories_r = c_t_r

        with tf.name_scope('Output'):
            kf_state = tf.transpose(y_hat_t, [1, 0, 2])
            # kf_state_recon = tf.transpose(y_hat_t, [1, 0, 2])
            kf_covariance = tf.transpose(p_hat_t, [1, 0, 2])
            r_t = tf.transpose(r_t, [1, 0, 2])
            q_t = tf.transpose(q_t, [1, 0, 2])

        return memories_r, memories_q, states_r, states_q, kf_state, kf_covariance, r_t, q_t

    def get_logits(self):
        return self._logits

    def get_net_scope(self):
        return self._nn_scope

