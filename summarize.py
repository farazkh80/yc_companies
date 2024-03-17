import os
import collections
from tqdm import tqdm
import json

from openai import OpenAI
import cohere

from cluster import K

# Constants
API_KEY = os.environ['COHERE_API_KEY']
COMMAND = 'command'
CHAT_COMMAND = 'command-r'


# Define the paths to the files
TITLES_FILE = 'data/titles.json'
SUMMARIES_FILE = 'data/summaries.json'


TITLE_FORMAT = "word_1, word_2, word_3"
SUMMARY_FORMAT = f"""
**Area of focus**: a paragraph describing the area of focus, target audience, target market, and commonly used technologies of the companies in this cluster.

**Trends and Patterns**:
- **Trend 1**: Brief description
- **Trend 2**: Brief description
- **Trend 3**: Brief description

**Examples**:
- Company 1: one line summary from the given context
- Company 2: one line summary from the given context
- Company 3: one line summary from the given context
- Company 4: one line summary from the given context
- Company 5: one line summary from the given context

"""




def chat_cluster_title_cohere(combined_text):
    """Generate a 3-word title for a cluster using Cohere's 'chat' model"""
    co = cohere.Client(API_KEY)
    chat_history = [
        {"role": "system", "message": "You are given a group of companies and their one line summaries, and supposed to generate a 3-word title that represents the companies and their descriptions. The output should be a 3-word title with no extra tokens."},
    ]
    message =f"""Generate a 3-word title that represents the following companies and their descriptions down below.  The title should be 3 words and entail things like area of focus, target audience and market, and used technologies.\n
    Your output should follow format of
    {TITLE_FORMAT}

    # Company Descriptions
    {combined_text}
    """
    response = co.chat(model=CHAT_COMMAND, message=message, max_tokens=10, chat_history=chat_history)
    return response.text

def chat_summaries_cohere(combined_text):
    """Generate summaries using Cohere's 'chat' model"""
    co = cohere.Client(API_KEY)
    chat_history = [
        {"role": "system", "message": "You are given a group of companies and their one line summaries, and supposed to generate a summary that represents the companies and their descriptions."}
    ]
    message = f"""Generate a summary that represents the following companies and their description provided down below, your output should follow the fornat
    {SUMMARY_FORMAT}

    # Company Descriptions   
    {combined_text}
    """
    response = co.chat(model=CHAT_COMMAND, message=message, max_tokens=400, chat_history=chat_history)
    return response.text

def get_titles_and_summaries(labels, company_names, descriptions):
    """Main function to generate summaries"""
    # If both files exist, read the titles and summaries from the files
    if os.path.exists(TITLES_FILE) and os.path.exists(SUMMARIES_FILE):
        with open(TITLES_FILE, 'r') as f:
            titles = json.load(f)
            titles = {int(k): v for k, v in titles.items()}
        with open(SUMMARIES_FILE, 'r') as f:
            summaries = json.load(f)
            summaries = {int(k): v for k, v in summaries.items()}
    else:
        # Group companies by cluster labels
        clustered_companies = collections.defaultdict(list)
        for i, label in enumerate(labels):
            cluster_entry = {'name': company_names[i], 'description': descriptions[i]}
            clustered_companies[label].append(cluster_entry)

        # Store the titles in a dictionary
        titles = {}
        # Summaries in a dictionary
        summaries = {}
        
        for label in tqdm(range(K), desc='Summarizing clusters'):
            # Combine company names and descriptions
            companies = clustered_companies[label]
            combined_text = ' '.join([f"{company['name']}: {company['description']}" for company in companies])

            # Generate the cluster title
            title = chat_cluster_title_cohere(combined_text)

            # Store the title in the dictionary
            titles[label] = title

            # Generate the summary
            summary = chat_summaries_cohere(combined_text)


            # Store the summary in the dictionary
            summaries[label] = summary
            
        # Write the titles and summaries to files
        with open(TITLES_FILE, 'w') as f:
            json.dump(titles, f)
        with open(SUMMARIES_FILE, 'w') as f:
            json.dump(summaries, f)

    return titles, summaries