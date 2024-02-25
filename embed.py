import cohere
import numpy as np
import os

API_KEY = os.environ['COHERE_API_KEY']
EMBEDDINGS_PATH = 'data/embeddings.npy'

def generate_embeddings(texts):
    co = cohere.Client(API_KEY)
    embeddings = co.embed(model='embed-english-v3.0', texts=texts, input_type='classification').embeddings
    np.save(EMBEDDINGS_PATH, embeddings)

def load_embeddings():
    if os.path.exists(EMBEDDINGS_PATH):
        return np.load(EMBEDDINGS_PATH)
    else:
        return None
