import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import json
import os

from embed import generate_embeddings, load_embeddings
from cluster import perform_clustering, load_cluster_labels, reduce_embeddings
from summarize import generate_summaries

EMBEDDINGS_PATH = 'data/embeddings.npy'
REDUCED_EMBEDDINGS_PATH = 'data/reduced_embeddings.npy'
CLUSTER_PATH = 'data/cluster_labels.npy'
DESCRIPTION_FILE = 'data/descriptions.txt'

# Load your JSON data
with open('yc_companies.json', 'r') as file:
    data = json.load(file)

# Extract text data for embedding, company names, and descriptions
texts = [item['description'] for item in data]
company_names = [item['name'] for item in data]
descriptions = [item['description'] for item in data]

# Check for and generate/load embeddings
if not os.path.exists(EMBEDDINGS_PATH):
    generate_embeddings(texts)
embeddings = load_embeddings()

# Check for and perform/reduce clustering
if not os.path.exists(REDUCED_EMBEDDINGS_PATH) or not os.path.exists(CLUSTER_PATH):
    reduced_embeddings = reduce_embeddings(embeddings)
    labels = perform_clustering(reduced_embeddings)
else:
    reduced_embeddings = np.load(REDUCED_EMBEDDINGS_PATH)
    labels = load_cluster_labels()

# Prepare data for Altair plot
plot_data = pd.DataFrame({
    'x': reduced_embeddings[:, 0],
    'y': reduced_embeddings[:, 1],
    'label': labels,
    'company': company_names,
    'description': descriptions
})

# Streamlit visualization with Altair
st.title("Clustered Company Data Visualization")

chart = alt.Chart(plot_data).mark_circle(size=60).encode(
    x='x',
    y='y',
    color='label:N',
    tooltip=['company', 'description', 'label']
).interactive()

st.altair_chart(chart, use_container_width=True)

# Add a UI element to let the user choose the summarization API
use_openai = st.checkbox("Use OpenAI for summarization", value=True)

# Then, when generating summaries:
summaries = generate_summaries(labels, company_names, descriptions, use_openai=use_openai)

st.write(summaries)