import configparser
import os

config = configparser.ConfigParser()
curr_dir = os.getcwd()
config.read(os.path.join(curr_dir, 'configs.ini'))

HOST = config['API'].get('host')
PORT = config['API'].getint('port')

FUSE_OVERLAPS = config['L-SYSTEM'].getboolean('fuse_overlaps')