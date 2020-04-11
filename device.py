# # import tensorflow as tf
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
#
# # from keras import backend as K
# # print(K.tensorflow_backend._get_available_gpus())
# import tensorflow as tf
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
from datetime import datetime


print(datetime.now().strftime('%Y-%m-%d-%H-%M'))