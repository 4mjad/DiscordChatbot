# Import modules
import re

# Load the datasets
dataset = "human_text.txt"
dataset2 = "robot_text.txt"

# Opening the files and reading them
with open(dataset, 'r', encoding = 'utf-8') as f:
  questions = f.read().split('\n')
with open(dataset2, 'r', encoding = 'utf-8') as f:
  answers = f.read().split('\n')

# Removing the brackets and the text inside the brackets and then replacing them with the word hi
questions = [re.sub("[\(\[].*?[\)\]]", "hi", line) for line in questions]
# Removing all the punctuation from the questions.
questions = [" ".join(re.findall(r"\w+", line)) for line in questions]
answers = [re.sub("[\(\[].*?[\)\]]", "hi", line) for line in answers]
answers = [" ".join(re.findall(r"\w+", line)) for line in answers]

# Zipping the questions and answers together.
pairs = list(zip(questions, answers))