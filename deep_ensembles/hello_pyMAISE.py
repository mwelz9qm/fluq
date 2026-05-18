import pyMAISE as mai
import tensorflow as tf

if __name__ == "__main__":
    print("Are you running on CPU or GPU?")
    print(tf.config.list_physical_devices())
