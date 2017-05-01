from tfpred_base import TFLearnBase
import tflearn
import tensorflow as tf
import numpy
import utilities


class RNN(TFLearnBase):

    def __init__(self, drug_no):

        super().__init__(drug_no)
        self.model_type = 'RNN'
        self.mod_input()

    def mod_input(self):

        self.xTr = numpy.reshape(self.xTr, (self.xTr.shape[0], 1, self.xTr.shape[1]))
        self.xTe = numpy.reshape(self.xTe, (self.xTe.shape[0], 1, self.xTe.shape[1]))

    @staticmethod
    def add_deep_layers(net, layer_size):
        return tflearn.simple_rnn(net, layer_size)

    def model(self, layer_size=32, tensorboard_verbose=3, batch_norm=2,
              learning_rate=0.001):

        input_shape = [None, 1, self.xTr.shape[2]]
        net = tflearn.input_data(shape=input_shape)

        net = tflearn.layers.normalization.batch_normalization(net)
        deep_layers_output = self.add_deep_layers(net, layer_size)
        net = tflearn.layers.normalization.batch_normalization(deep_layers_output)
        net = tflearn.fully_connected(net, 300, activation='prelu')

        if batch_norm > 0:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.dropout(net, 0.1)
        net = tflearn.fully_connected(net, 1, activation='sigmoid')
        if batch_norm > 1:
            net = tflearn.layers.normalization.batch_normalization(net)

        with tf.name_scope("TargetsData"):  # placeholder for target variable (i.e. trainY input)
            targetY = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="Y")

        network = tflearn.regression(net,
                                     placeholder=targetY,
                                     optimizer=self.optimizer(learning_rate),
                                     learning_rate=learning_rate,
                                     loss=tflearn.mean_square(net, targetY),
                                     metric=self.accuracy(net, targetY))

        model = tflearn.DNN(network, tensorboard_verbose=tensorboard_verbose)

        self.populate_params(['model_type', 'layer_size', 'tensorboard_verbose', 'batch_norm', 'learning_rate'],
                             [self.model_type, layer_size, tensorboard_verbose, batch_norm, learning_rate])
        return model

    def train(self, model, num_epochs=20, batch_size=64, validation_set=0.1):

        run_id = utilities.assign_id(self.params, self.res_dir)
        model.fit(self.xTr, self.yTr,
                  n_epoch=num_epochs,
                  validation_set=validation_set,
                  batch_size=batch_size,
                  shuffle=True,
                  run_id=run_id)

        if run_id is None:
            run_id = str(numpy.random.randint(50000))

        self.populate_params(['n_epochs', 'validation_set', 'batch_size', 'data_dir', 'run_id'],
                             [num_epochs, validation_set, batch_size, self.res_dir, run_id])

        print("Done!")
        return model

    def predict(self, model):

        res = model.predict(self.xTe)
        as_df, fpr, tpr, roc_auc, plot = utilities.scoring(res, self.params['model_type'], self.yTe)
        self.populate_params(['roc_auc'], [roc_auc])
        utilities.save_results_to_file(self.params, as_df, plot, fpr, tpr)

        return as_df, roc_auc

if __name__ == '__main__':

    DRUG_ID = 152
    pipe = RNN(DRUG_ID)

    mymodel = pipe.model(layer_size=90,
                         tensorboard_verbose=1,
                         batch_norm=1,
                         learning_rate=0.0001)

    trained_model = pipe.train(mymodel,
                               num_epochs=1,
                               batch_size=80,
                               validation_set=0.1)

    df, r = pipe.predict(trained_model)

    print(df)
    print(r)


