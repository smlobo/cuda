#!/usr/bin/env python3

import tensorflow as tf
# from tensorflow import device_lib

print(tf.sysconfig.get_build_info())
print(tf.config.experimental.list_physical_devices(device_type='GPU'))

# print(device_lib.list_local_devices())