from configparser import ConfigParser
import os.path as osp

path = osp.dirname(__file__)


CONF_PATH = osp.join(path, ".settings.conf")
print(CONF_PATH)

def getconfig():
    conf = ConfigParser()
    conf.read(CONF_PATH)
    return conf

