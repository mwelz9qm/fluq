import pyMAISE as mai

if __name__ == "__main__":
    print("Are you running on CPU or GPU?")  # TODO can't get Apple silicon GPU working on this version of pyMAISE
    print(mai.tf.config.list_physical_devices())
