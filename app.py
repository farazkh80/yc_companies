import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import json
import os

from embed import generate_embeddings, load_embeddings
from cluster import perform_clustering, load_cluster_labels, reduce_embeddings
from summarize import get_titles_and_summaries

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
links = [item['link'] for item in data]

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
    'description': descriptions,
    'link': links
})

# Generate titles and summaries
titles, summaries = get_titles_and_summaries(labels, company_names, descriptions)

# Add titles to the plot data
plot_data['title'] = plot_data['label'].map(titles)

# Streamlit visualization with Altair
st.title("Clustered Visualization of Y Combinator Companies")
st.write(f"**Total Companies analyzed**:\t{len(data)}")

chart = alt.Chart(plot_data).mark_circle(size=60).encode(
    x='x',
    y='y',
    color='title',
    tooltip=['company', 'description', 'title', 'link']
).interactive()

st.altair_chart(chart, use_container_width=True)

# Display summaries
for label, summary in summaries.items():
    st.write(f"## Cluster {titles[label]}")
    # number of companies in the cluster
    st.write(f"**Number of companies in this cluster**: {len(plot_data[plot_data['label'] == label])}")
    st.write(summary)