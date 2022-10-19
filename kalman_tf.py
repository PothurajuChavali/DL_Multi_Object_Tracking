import tensorflow as tf
import numpy as np


class KalmanTF(object):
    """
    Tensorflow based Kalman Filter Base Class
    """
    def __init__(self, kf_num_in=4, kf_num_out=8, dt=1, batch_size=1):
        # measurement and state vector sizes
        self._kf_meas_size = kf_num_in
        self._kf_state_size = kf_num_out
        self._batch_size = batch_size

        # Initial state (location and velocity) [[pos_x], [pos_y], [vel_x], [vel_y]]
        self._state = tf.expand_dims(tf.constant(np.zeros([self._kf_state_size, 1],
                                     dtype=np.float32), dtype=tf.float32), axis=0)
        self._state = tf.tile(self._state, tf.constant([batch_size, 1, 1], dtype=tf.int32))

        # Known external motion vector
        self._ext_motion = tf.constant(np.zeros([self._kf_state_size, self._batch_size], dtype=np.float32))

        # Initial uncertainty covariance [cov_pos_x_self, cov_pos_y_self, cov_vel_x_self, cov_vel_y_self]
        self._covariance = tf.scalar_mul(100000, tf.eye(self._kf_state_size, dtype=tf.float32,
                                                        batch_shape=[self._batch_size]))

        # Time step between frames
        self._dt = dt

        # Next State Function (State Transition Matrix)
        state_fn = np.eye(self._kf_state_size, dtype=np.float32)
        row_id = np.arange(self._kf_state_size - self._kf_meas_size)
        col_id = np.arange(self._kf_meas_size, self._kf_state_size)
        state_fn[row_id, col_id] = dt

        row_id = np.arange(self._kf_meas_size)
        col_id = np.arange(self._kf_meas_size * 2, self._kf_state_size)
        state_fn[row_id, col_id] = dt*dt/2
        # self._state_fn = tf.constant(state_fn, dtype=tf.float32)
        self._state_fn = tf.constant(np.tile(state_fn, (self._batch_size, 1, 1)), dtype=tf.float32)

        # Measurement function: we observe x and y but not the two velocities (Maps states to measurements)
        measurement_fn = np.zeros([self._kf_meas_size, self._kf_state_size], dtype=np.float32)
        measurement_fn[row_id, row_id] = 1
        self._measurement_fn = tf.constant(np.tile(measurement_fn, (batch_size, 1, 1)), dtype=tf.float32)

    def get_state(self):
        return self._state

    def get_covariance(self):
        return self._covariance

    def predict_state(self, state):
        # Next State Estimate
        self._state = tf.matmul(self._state_fn, state)

    def predict_covarince(self, covariance, process_noise):
        # Uncertainty Covariance
        self._covariance = tf.add(tf.matmul(tf.matmul(self._state_fn, covariance),
                                            tf.transpose(self._state_fn, [0, 2, 1])), process_noise)

    def update(self, state, covariance, measurement, measurement_noise):
        # measurement error
        error = tf.subtract(measurement, tf.matmul(self._measurement_fn, state))

        # Project the system uncertainty to measurement space using measurement function and add the measurement noise
        uncertainty = tf.add(
            tf.matmul(tf.matmul(self._measurement_fn, covariance), tf.transpose(self._measurement_fn, [0, 2, 1])),
            measurement_noise)

        # Kalman Gain
        kalman_gain = tf.matmul(tf.matmul(covariance, tf.transpose(self._measurement_fn)),
                                tf.matrix_inverse(uncertainty))

        # Update the state estimate
        self._state = tf.add(state, tf.matmul(kalman_gain, error))

        # Update the uncertainty covariance
        self._covariance = tf.matmul(
            tf.subtract(tf.eye(self._kf_state_size, dtype=tf.float32, batch_shape=[self._batch_size]), 
                        tf.matmul(kalman_gain, self._measurement_fn)), covariance)

