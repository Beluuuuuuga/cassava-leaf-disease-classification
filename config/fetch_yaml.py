import os


BASEDIR = os.path.dirname(__file__)

def catalog():
    data = "catalog.yaml"
    fullpath = os.path.join(BASEDIR, data)
    return fullpath

def parameters():
    data = "parameters.yaml"
    fullpath = os.path.join(BASEDIR, data)
    return fullpath