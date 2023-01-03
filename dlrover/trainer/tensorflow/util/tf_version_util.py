import tensorflow as tf

def get_tf_version():
    version = tf.__version__
    version = version.split(".")
    return version

def is_tf_1():
    version = get_tf_version()
    return int(version[0])==1

def is_tf_2():
    version = get_tf_version()
    return int(version[0])>1

def is_tf_113():
    version = get_tf_version()
    return int(version[0])==1 and int(version[1])==13

def is_tf_115():
    version = get_tf_version()
    return int(version[0])==1 and int(version[1])==15