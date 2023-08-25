import configparser
import logging

# Configure logging settings
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')

# Read the config file
CONFIG = configparser.ConfigParser()
CONFIG_PATH = '../config.ini'
CONFIG.read(CONFIG_PATH)

# Set debug mode based on the configuration
DEBUG_MODE = CONFIG.get('General', 'debug_mode')
if DEBUG_MODE == 'high':
    logging.getLogger().setLevel(logging.DEBUG)

# Read general configuration
USE_FULL_VOLUME_LENGTH = CONFIG.getboolean('General', 'use_full_volume_length')
CUSTOM_VOLUME_LENGTH = CONFIG.getint('General', 'custom_volume_length')
CUSTOM_VOLUME_START = CONFIG.getint('General', 'custom_volume_start')
