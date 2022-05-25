# Import modules
import nest_asyncio
nest_asyncio.apply()
import numpy as np
import re
import discord
from test import encoder_model, decoder_model, num_decoder_characters, num_encoder_characters, input_features_index, target_features_index, reverse_target_features_index, max_decoder_seq_length, max_encoder_seq_length

# Creating a class that contains the discord method for running the chatbot
class ChatBot(discord.Client):

  # Function from https://www.codecademy.com/learn/deep-learning-and-generative-chatbots/modules/generative-chatbots/cheatsheet
  # Converts user input into a matrix  
  def convert_to_matrix(self, user_input):
    # Splitting the user input into tokens
    tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
    # Creating a matrix of zeros with the dimensions of 1, max_encoder_seq_length, and
    # num_encoder_characters
    user_input_matrix = np.zeros((1, max_encoder_seq_length, num_encoder_characters), dtype = 'float32')
    # Converting the user input into a matrix
    for timestep, token in enumerate(tokens):
      if token in input_features_index:
        user_input_matrix[0, timestep, input_features_index[token]] = 1.0
    return user_input_matrix
  
  # Creating a response using the seq2seq model
  # Body copied from decode_sequence in test.py but uses user input this time
  def generate_response(self, user_input):
    # Gets user input and convert it to matrix
    input_matrix = self.convert_to_matrix(user_input)
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_matrix)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_characters))
     # Populate the first token of target sequence with the start token.
    target_seq[0, 0, target_features_index['<START>']] = 1.0
    # Sampling loop for a batch of sequences
    # # (to simplify, here we assume a batch of size 1).
    chatbot_response = ''

    stop_condition = False
    while not stop_condition:
      # Run the decoder model to get possible 
      # # output tokens (with probabilities) & states
      output_tokens, hidden_state, cell_state = decoder_model.predict(
        [target_seq] + states_value)
      
      # Choose token with highest probability
      sampled_token_index = np.argmax(output_tokens[0, -1, :])
      sampled_char = reverse_target_features_index[sampled_token_index]
      chatbot_response += " " + sampled_char
      
      # Exit condition: either hit max length
      # # or find stop token.
      if (sampled_char == '<END>' or len(chatbot_response) > max_decoder_seq_length):
        stop_condition = True

      # Update the target sequence (of length 1).  
      target_seq = np.zeros((1, 1, num_decoder_characters))
      target_seq[0, 0, sampled_token_index] = 1.0
      
      # Update states
      states_value = [hidden_state, cell_state]
    
    # Sets response by marking the start and end of each sentence 
    chatbot_response = chatbot_response.replace("<START>", "").replace("<END>", "")

    # Returns response  
    return chatbot_response + "\n"

  async def on_ready(self):
        # Prints out information when the bot wakes up
        print('Logged in as')
        print(self.user.name)
        print(self.user.id)
        print('------')

  # Sending and receiving messages      
  async def on_message(self, message):
    if message.author.id == self.user.id:
            return
    
    # Waiting for a user reply:
    user_input = message.content
    if user_input.startswith("$"):
        await message.channel.send(self.generate_response(user_input))

# Client running
def main():
  DiscordChatbot = ChatBot()
  DiscordChatbot.run("OTEzNDcwODAwNzYwMjkxMzc4.YZ-9-g.6Mit3B2mC4WtkQ8DZnlWqrK7OLg")

if __name__ == '__main__':
  main()