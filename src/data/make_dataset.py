from src.utils import definitions, utilities
from numerapi.numerapi import NumerAPI
import os
import pandas


TOURNAMENT_DATA = os.path.join(definitions.DATA_DIR, 'raw/numerai_tournament_data.csv')
TRAINING_DATA = os.path.join(definitions.DATA_DIR, 'raw/numerai_training_data.csv')


def download_new_dataset():
    napi = NumerAPI()
    print("Downloading the current dataset...")
    napi.download_current_dataset(dest_path=os.path.join(definitions.DATA_DIR, 'raw'), unzip=True)


def clean_train_data(data_file):

    df = pandas.read_csv(data_file)
    df = df.set_index('id')
    x = df.drop(['data_type', 'target', 'era'], 1)
    y = df['target']
    return x, y


def split_val_test(data_file):

    df = pandas.read_csv(data_file)
    df = df.set_index('id')
    val = df[df.data_type == 'validation']
    val_x = val.drop(['data_type', 'target', 'era'], 1)
    val_y = val['target']

    return val_x, val_y


def generate_train_test():
    """ define how to clean your data in the utilities module """

    x, y = clean_train_data(TRAINING_DATA)
    x.to_csv(os.path.join(definitions.DATA_DIR, 'processed/xtrain.csv'))
    y.to_csv(os.path.join(definitions.DATA_DIR, 'processed/ytrain.csv'))

    x_val, y_val = split_val_test(TOURNAMENT_DATA)
    x_val.to_csv(os.path.join(definitions.DATA_DIR, 'processed/xval.csv'))
    y_val.to_csv(os.path.join(definitions.DATA_DIR, 'processed/yval.csv'))

    to_pred = pandas.read_csv(TOURNAMENT_DATA).drop(['data_type', 'target', 'era'], 1)
    to_pred.to_csv(os.path.join(definitions.DATA_DIR, 'processed/xtest.csv'))

    return 'Data now accessible at %s as csv files: xtrain.csv, xtest.csv, ytrain.csv,' \
           ' ytest.csv' % os.path.join(definitions.DATA_DIR, 'processed')



if __name__ == '__main__':
    df = pandas.read_csv(definitions.DATA_DIR + '/raw/numerai_tournament_data.csv')
    print(df.tail())
