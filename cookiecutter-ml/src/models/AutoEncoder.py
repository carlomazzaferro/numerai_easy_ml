import tensorflow as tf
import tflearn
import utilities
from base import Base


class AutoEncoder(Base):

    def __init__(self, drug_no):

        super().__init__(drug_no)
        self.model_type = 'AutoEncoder'
        self.mod_input()

    def mod_input(self):
        pass

    @staticmethod
    def add_deep_layers(net, layer_sizes):

        for idx, layer in enumerate(layer_sizes[0:-1]):
            net = tflearn.fully_connected(net, layer_sizes[idx], activation='prelu')

        out_rnn = tflearn.fully_connected(net, layer_sizes[-1], activation='prelu')

        return out_rnn

    def model(self, layer_size=None, tensorboard_verbose=3, batch_norm=2, learning_rate=0.001):

        input_shape = [None, self.x_train.shape[1]]

        # Building the encoder
        encoder = tflearn.input_data(shape=input_shape)
        encoder = tflearn.fully_connected(encoder, 5000)
        encoder = tflearn.fully_connected(encoder, 750)
        encoder = tflearn.fully_connected(encoder, 250)

        # Building the decoder
        decoder = tflearn.fully_connected(encoder, 750)
        decoder = tflearn.fully_connected(decoder, 5000)
        decoder = tflearn.fully_connected(decoder, self.x_train.shape[1])

        # Training the auto encoder

        with tf.name_scope("TargetsData"):  # placeholder for target variable (i.e. trainY input)
            targetY = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="Y")

        network = tflearn.regression(decoder,
                                     placeholder=targetY,
                                     optimizer=self.optimizer(learning_rate),
                                     learning_rate=learning_rate,
                                     loss=tflearn.mean_square(net, targetY),
                                     metric=self.accuracy(net, targetY))

        model = tflearn.DNN(network, tensorboard_verbose=tensorboard_verbose)

        self.populate_params(['model_type', 'layer_size', 'tensorboard_verbose', 'batch_norm', 'n_layers',
                              'learning_rate', 'drug_no'], [self.model_type, layer_size, tensorboard_verbose, batch_norm,
                                                            len(layer_size), learning_rate, self.drug_no])
        return model




if __name__ == '__main__':

    DRUG_ID = 152
    pipe = AutoEncoder(DRUG_ID)

    mymodel = pipe.model(layer_size=[700, 500, 200],
                         tensorboard_verbose=1,
                         batch_norm=1,
                         learning_rate=0.0001)

    trained_model = pipe.train(mymodel,
                               num_epochs=40,
                               batch_size=80,
                               validation_set=0.1)

    df, r = pipe.predict(trained_model)

    print(df)
    print(r)



    batch_size = 100
    epochs = 100
    lr = 0.0001


    # Regression, with mean square error
    net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001,
                             loss='mean_square', metric=None)

    # Training the auto encoder
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(xTr, xTr, n_epoch=epochs, validation_set=(xTe, xTe),
              run_id="auto_encoder", batch_size=256)

    # Encoding X[0] for test
    print("\nTest encoding of X[0]:")
    # New model, re-using the same session, for weights sharing
    encoding_model = tflearn.DNN(encoder, session=model.session)
    print(encoding_model.predict([xTr[0]]))

    # Testing the image reconstruction on new data (test set)
    print("\nVisualizing results after being encoded and decoded:")
    xTe = tflearn.data_utils.shuffle(xTe)[0]
    # Applying encode and decode over test set
    encode_decode = model.predict(xTe)