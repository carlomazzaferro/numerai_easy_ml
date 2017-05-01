import numpy
import tensorflow as tf
import tflearn
import os
import pandas
from src.utils import definitions, utilities
from src.data import make_dataset


class Project(object):

    types = ['classification', 'regression']

    def __init__(self, problem_type, num_classes):

        self.params = {}
        self.raw_data_files = [i for i in os.listdir(os.path.join(definitions.DATA_DIR, 'raw/')) if i.startswith('.') == False]
        self.num_classes = num_classes
        self.problem_type = problem_type
        if num_classes > 0 and problem_type == 'regression':
            raise ValueError('Num classes must be left unfilled if you are working on a regression problem')
        print('Available data to be preprocessed: %s' % ", ".join(self.raw_data_files))

    def populate_params(self, k, v):
        for val, idx in enumerate(k):
            self.params[idx] = v[val]


class Base(Project):

    def __init__(self, problem_type='multiclass', num_classes=0):

        super().__init__(problem_type, num_classes)

        self.data_path = os.path.join(definitions.DATA_DIR, 'processed/')
        self.res_dir = os.path.join(definitions.RESUTLS_DIR, 'model_outputs/')
        self.models_dir = definitions.MODELS_DIR
        self.x_train, self.x_test, self.y_train, self.y_test, self.x_tournament = (None, None, None, None, None)
        self.id = None

    def load_data(self):
        """ loader function, customize as needed """
        make_dataset.generate_train_test()
        self.x_train = pandas.read_csv(os.path.join(definitions.DATA_DIR, 'processed/xtrain.csv'))
        self.x_test = pandas.read_csv(os.path.join(definitions.DATA_DIR, 'processed/xval.csv'))
        self.y_train = pandas.read_csv(os.path.join(definitions.DATA_DIR, 'processed/ytrain.csv'), names=['id', 'target'])
        self.y_test = pandas.read_csv(os.path.join(definitions.DATA_DIR, 'processed/yval.csv'), names=['id', 'target'])
        self.x_tournament = pandas.read_csv(os.path.join(definitions.DATA_DIR, 'processed/xtest.csv'))
        self.id = self.x_tournament['id']

        return 'Data now accessible as class attribute'

    @staticmethod
    def optimizer(lr):
        """ define your optimizer here """
        opt = tflearn.RMSProp(learning_rate=lr)  # , decay=0.9)
        return opt

    @staticmethod
    def loss_func(y_pred, y_true):
        """ define your loss function here here """
        loss = tf.reduce_mean(tf.square(y_pred - y_true))
        return loss

    @staticmethod
    def accuracy(y_pred, y_true):
        """ define your accuracy measure here """
        acc = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_pred, y_true))))
        return acc

    def train(self, model, num_epochs=20, batch_size=64, validation_set=0.1):

        run_id = utilities.assign_id(self.params, self.res_dir)

        model.fit(self.x_train, self.y_train,
                  n_epoch=num_epochs,
                  validation_set=validation_set,
                  batch_size=batch_size,
                  shuffle=True,
                  run_id=run_id)

        if run_id is None:
            run_id = str(numpy.random.randint(50000))

        self.populate_params(['n_epochs', 'batch_size', 'data_dir', 'run_id'],
                             [num_epochs, batch_size, self.res_dir, run_id])

        print("Done!")
        return model

    def predict(self, model):

        y_prediction = model.predict(self.x_tournament)
        y_prediction = [i[0] for i in y_prediction]
        results_df = pandas.DataFrame(data={'probability': y_prediction})
        joined = pandas.DataFrame(self.id).join(results_df).set_index('id')

        validation_preds = model.predict(self.x_test)
        as_df, fpr, tpr, roc_auc, plot = utilities.scoring(validation_preds, self.params['model_type'], self.y_test)
        self.populate_params(['roc_auc'], [roc_auc])
        utilities.save_results_to_file(self.params, as_df, plot, fpr, tpr, joined)

        return as_df, roc_auc


