# Import modules
import numpy as np
import re
from prep import pairs

# Adapted from https://keras.io/examples/nlp/lstm_seq2seq/

# Creating empty lists to store the sentences
input_texts = []
target_texts = []
# Creating empty vocabulary sets
input_characters = set()
target_characters = set()

# Creating a list of the first 400 lines of the pairs list
for line in pairs[:400]:
  # Assigning the first and second elements of the list to the variables `input_text` and `target_text`
  input_text, target_text = line[0], line[1]
  # Adding the input text to the input_texts list
  input_texts.append(input_text)
  # Splitting the target text into words and adding a space between each word
  target_text = " ".join(re.findall(r"[\w']+|[^\s\w]", target_text))
  # Adding the start and end tags to the target text
  target_text = '<START> ' + target_text + ' <END>'
  # Adding the target text to the target_texts list
  target_texts.append(target_text)
  
  # Creating a list of unique words for the input and output sentences
  for char in re.findall(r"[\w']+|[^\s\w]", input_text):
    if char not in input_characters:
      input_characters.add(char)
  # Splitting the target text into words and adding a space between each word.
  for char in target_text.split():
    if char not in target_characters:
      target_characters.add(char)

# Sorting the list of unique words in the input and output sentences
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
# Counting the number of unique words in the input and target text.
num_encoder_characters = len(input_characters)
num_decoder_characters = len(target_characters)

# Creating a dictionary of the input and output words and their index.
input_features_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_features_index = dict([(char, i) for i, char in enumerate(target_characters)])

# Store the input characters as key-value pairs but this time they are swapped where word = index & value = key
# Is reversed to decode back to a readable format
reverse_input_features_index = dict((i, char) for char, i in input_features_index.items())
reverse_target_features_index = dict((i, char) for char, i in target_features_index.items())

# Finding the maximum length of the input and output sentences
max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_text)) for input_text in input_texts])
max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_text)) for target_text in target_texts])

# Prints to the console what is being fed to the model
print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_characters)
print("Number of unique output tokens:", num_decoder_characters)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)


# Creating a 3D array of zeros with the dimensions of the number of input texts, the maximum length of
# the input text and the number of unique words in the input text
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_characters), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_characters), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_characters), dtype='float32')

# For loop to fill out the 1s in each vector
# Iterating through the input and target texts and assigning them to the `input_text` and `target_text`
for line, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for timestep, char in enumerate(re.findall(r"[\w']+|[^\s\w]", input_text)):
        # Assigns a value of 1 for the current word, timestep and line 
        encoder_input_data[line, timestep, input_features_index[char]] = 1.0
    
    # Same method applied to the decoder
    # Is creating a one-hot encoded vector for each word in the target text
    for timestep, char in enumerate(target_text.split()):
        decoder_input_data[line, timestep, target_features_index[char]] = 1.0
        # But if the timestep is not 0 then decreases it by 1
        if timestep > 0:
            decoder_target_data[line, timestep - 1, target_features_index[char]] = 1.0