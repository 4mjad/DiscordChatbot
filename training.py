# Import modules
from tensorflow import keras
from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import Model
from preprocessing import num_encoder_characters, num_decoder_characters, decoder_target_data, encoder_input_data, decoder_input_data, decoder_target_data, max_encoder_seq_length, max_decoder_seq_length

# Adapted from https://keras.io/examples/nlp/lstm_seq2seq/

# Number of units in the LSTM layer
dimensionality = 256
# Number of samples per gradient update
batch_size = 10
# Number of times the model is exposed to the training dataset
epochs = 4600

# Encoder Training
# Defining the input layer of the encoder
encoder_inputs = Input(shape = (None, num_encoder_characters))
# Creating a LSTM layer with 256 units and returning the hidden state and cell state
encoder_lstm = LSTM(dimensionality, return_state = True)
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
# Creating a list of the hidden state and cell state of the encoder
encoder_states = [state_hidden, state_cell]

# Decoder Training
# Defining the input layer of the decoder
decoder_inputs = Input(shape = (None, num_decoder_characters))
# Creating a LSTM layer with 256 units and returning the hidden state and cell state
decoder_lstm = LSTM(dimensionality, return_sequences = True, return_state = True)
# The decoder LSTM layer is taking the decoder input data and the encoder states as input. The encoder
# states are the hidden state and cell state of the encoder LSTM layer. The decoder LSTM layer is
# returning the decoder outputs, the hidden state and cell state of the decoder LSTM layer.
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state = encoder_states)
# A regularization technique that randomly sets half of the input units to 0 at each update during
# training time, which helps prevent overfitting
dropout = Dropout(rate = 0.5)
# Applying the dropout regularization technique to the decoder outputs
decoder_outputs = dropout(decoder_outputs)
# Creating a dense layer with the number of decoder characters as the number of units and the
# activation function as softmax
decoder_dense = Dense(num_decoder_characters, activation = 'softmax')
# Applying the dense layer to the decoder outputs
decoder_outputs = decoder_dense(decoder_outputs)

# Model
# Creating a model with the encoder inputs and decoder inputs as the input layers and the decoder
# outputs as the output layer.
training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# The `compile` method is used to configure the model for training. The `optimizer` parameter is used
# to specify the optimizer to be used for training. The `loss` parameter is used to specify the loss
# function to be used for training. The `metrics` parameter is used to specify the metrics to be used
# for training. The `sample_weight_mode` parameter is used to specify the sample weight mode to be
# used for training.
training_model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'], sample_weight_mode = 'temporal')
# The `fit` method is used to train the model. The `encoder_input_data` and `decoder_input_data` are
# the input data for the encoder and decoder. The `decoder_target_data` is the target
# data for the decoder. The `batch_size` is the number of samples per gradient update. The `epochs` is
# the number of times the model is exposed to the training dataset. The `validation_split` is the
# fraction of the training data to be used as validation data.
training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size = batch_size, epochs = epochs, validation_split = 0.2)
# Saving the model to a file
training_model.save('training_model.h5')