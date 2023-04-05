from datasets import load_dataset
import re

def remove_non_alphabetic_chars(s):
  output_string = re.sub('[^a-zA-Z\s]', '', s)
  return output_string

dataset = load_dataset("statworx/haiku", split='train')

haiku_data = [remove_non_alphabetic_chars(haiku['text']) for haiku in dataset]

for haiku in haiku_data:
    print(haiku)
    print('\n')