# Import modules
from tensorflow import keras
from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import Model, load_model
import numpy as np
from preprocessing import input_features_index, target_features_index, reverse_input_features_index, reverse_target_features_index, max_decoder_seq_length, input_characters, target_characters, input_text, target_texts, max_encoder_seq_length
from training import decoder_inputs, decoder_lstm, decoder_dense, encoder_input_data, num_decoder_characters, num_encoder_characters

# Loading the model that was trained in the previous step
training_model = load_model('training_model.h5')

# Keras inference setup from https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
# Construct the encoder
# Getting the first input layer of the model
encoder_inputs = training_model.input[0]
# Getting the output of the encoder LSTM layer
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
# Creating a list of the hidden and cell states of the encoder
encoder_states = [state_h_enc, state_c_enc]
# Creating a new model that takes the encoder inputs and outputs the encoder states
encoder_model = Model(encoder_inputs, encoder_states)

# Number of dimensions of the hidden state of the LSTM.
latent_dim = 256
# Creating a new input layer for the decoder model.
decoder_state_input_hidden = Input(shape=(latent_dim,))
decoder_state_input_cell = Input(shape=(latent_dim,))
# Creating a list of the hidden and cell states of the decoder
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
# Passing the decoder inputs and the initial state of the decoder to the decoder LSTM layer
decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
# Creating a list of the hidden and cell states of the decoder
decoder_states = [state_hidden, state_cell]
# Regularization technique that prevents overfitting
dropout = Dropout(rate=0.5)
# Passing the output of the decoder LSTM layer to the decoder dense layer
decoder_outputs = decoder_dense(decoder_outputs)
# Decoder model
# Creating a new model that takes the decoder inputs and the initial state of the decoder as inputs
# and outputs the output of the decoder LSTM layer and the hidden and cell states of the decoder
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Implementing the inference loop
def decode_sequence(test_input):
  # Encode the input as state vectors.
  states_value = encoder_model.predict(test_input)

  # Generate empty target sequence of length 1.
  target_seq = np.zeros((1, 1, num_decoder_characters))
  # Populate the first token of target sequence with the start token.
  target_seq[0, 0, target_features_index['<START>']] = 1.

  # Sampling loop for a batch of sequences
  # (to simplify, here we assume a batch of size 1).
  decoded_sentence = ''

  stop_condition = False
  while not stop_condition:
    # Run the decoder model to get possible 
    # output tokens (with probabilities) & states
    output_tokens, hidden_state, cell_state = decoder_model.predict(
      [target_seq] + states_value)

    # Choose token with highest probability
    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    sampled_char = reverse_target_features_index[sampled_token_index]
    decoded_sentence += " " + sampled_char

    # Exit condition: either hit max length
    # or find stop token.
    if (sampled_char == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
      stop_condition = True

    # Update the target sequence (of length 1).
    target_seq = np.zeros((1, 1, num_decoder_characters))
    target_seq[0, 0, sampled_token_index] = 1.

    # Update states
    states_value = [hidden_state, cell_state]

  return decoded_sentence