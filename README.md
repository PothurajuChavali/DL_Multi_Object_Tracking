# DL_Multi_Object_Tracking

Multi object tracking based on Long short term memory(LSTM) networks and Kalman filter(KF)

Learn process noise(Q) and measurement noise(R) from LSTM networks. In general, these parameters used in kalman filtering
are set/derived by user. The basic idea here is to learn Q and R by data

I refered https://arxiv.org/abs/1708.01885 and adapting it to object tracking.
