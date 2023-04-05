import pinecone
import os
import re
import openai
from tqdm.auto import tqdm
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL = 'text-embedding-ada-002'

# convert text into an embedding
def embed(text):
   return openai.Embedding.create(input=text, engine=MODEL)

def create_index(name, dim):
  pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV'))
  if name not in pinecone.list_indexes():
      pinecone.create_index(name, dimension=dim)
      
  return pinecone.Index(name)

def get_index(name):
  pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV'))
  return pinecone.Index(name)

def remove_non_alphabetic_chars(s):
  output_string = re.sub('[^a-zA-Z\s]', '', s)
  return output_string

def vectorize():
    #load haiku dataset and extract relevant fields
  dataset = load_dataset("statworx/haiku", split='train')
  haiku_data = [remove_non_alphabetic_chars(haiku['text']) for haiku in dataset]
  haiku_data = [haiku['text'] for haiku in dataset if haiku['text'] != '']

  # embed haiku texts as embeddings
  haiku_embeddings = embed(haiku_data[0])

  index_dim = len(haiku_embeddings['data'][0]['embedding'])
  # create index for a new pinecone vector db
  index = create_index('haikus', index_dim)
  #create vector embedding for each haiku and upsert the original text and embedding
  batch_size = 1  # process everything in batches of 32
  for i in tqdm(range(0, len(haiku_data), batch_size)):
      # set end position of batch
      i_end = min(i + batch_size, len(haiku_data))
      # get batch of haikus and title (i.e ['song'])
      haikus_batch = haiku_data[i: i + batch_size]
      ids_batch =  [str(n) for n in range(i, i + i_end)]
      # create embeddings
      embeds_batch = [haiku['embedding'] for haiku in embed(haikus_batch)['data']]
      # embeds_batch = [haiku['embedding'] for haiku in haiku_embeddings['data']]
      # prep metadata and upsert batch
      meta_data = [
          {
          'text': haiku
          } 
          for haiku in haikus_batch]
      to_upsert = zip(ids_batch, embeds_batch, meta_data)
      # upsert to pinecone vector db
      index.upsert(vectors=list(to_upsert))

# vectorize()









