import os

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir) # This is your Project Root
CONFIG_PATH = os.path.join(ROOT_DIR, 'configuration.conf')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
RESUTLS_DIR = os.path.join(ROOT_DIR, 'reports')
MODEL_OUTPUTS =  os.path.join(RESUTLS_DIR, 'model_outputs/')