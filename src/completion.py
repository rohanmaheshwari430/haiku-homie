import embedder
import openai

MODEL='text-davinci-003'

def retrieve(query):
  embedded_query = embedder.embed(query)
  # retrieve from Pinecone
  vectorized_embedding = embedded_query['data'][0]['embedding']
  index = embedder.get_index('haikus')
  # get relevant contexts (including the questions)
  index_res = index.query(vectorized_embedding, top_k=200, include_metadata=True)
  contexts = [retrieved_haiku['metadata']['text'] for retrieved_haiku in index_res['matches']]
  return contexts
    

def complete(query):
    limit = 3750
    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: Given the context above, write a haiku that rhymes about the following topics {query}\nAnswer:"
    )
    # append contexts until hitting limit
    contexts = retrieve(query)
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return execute_prompt(prompt)

def execute_prompt(prompt):
    res = openai.Completion.create(
        engine=MODEL,
        prompt=prompt,
        temperature=0.7,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()

def generate_haiku():
    #query = input('Enter some topics for a haiku: ')
   
    results = complete('chocolate, paris, sweet')
    print(results)
    

print(generate_haiku())