import os
import collections
from openai import OpenAI
import cohere
from tqdm import tqdm

API_KEY = os.environ['COHERE_API_KEY']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
SUMMARY_FILE = 'data/descriptions.txt'
K = 8

def generate_summaries_openai(prompt):
    client = OpenAI()
    # use gpt-3.5-turbo for faster response times
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [
            {
                "role": "system",
                "content": "You are given a group of companies with similar descriptions of what they do. Analyze these companies' descriptions, find trends and insights from them as a group. Report any patterns you find from these companies' field of focus, target customers, and why they relate to each other. Lastly, give at least 5 examples of companies in this cluster that support your trend and pattern findings. Here are the companies in this cluster for you to analyze:"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=512
    )

    print(response.choices[0].message.content)
    return response.choices[0].message.content
        

def generate_summaries_cohere(prompt):
    co = cohere.Client(API_KEY)
    response = co.generate(model='command_nightly', prompt=prompt, stop_sequences=[], max_tokens=200)
    return response.generations[0].text

def generate_summaries(labels, company_names, descriptions, use_openai=True):
    sdk = 'cohere' if not use_openai else 'openai'
    summary_file = SUMMARY_FILE.replace('.txt', f'_{sdk}.txt')
    if not os.path.exists(summary_file):
        clustered_companies = collections.defaultdict(list)
        for i, label in enumerate(labels):
            cluster_entry = {
                'name': company_names[i],
                'description': descriptions[i]
            }
            clustered_companies[label].append(cluster_entry)

        with open(summary_file, 'w') as f:
            for label in tqdm(range(K), desc='Summarizing clusters'):
                companies = clustered_companies[label]
                combined_text = ' '.join([f"{company['name']}: {company['description']}" for company in companies])
                prompt = f"""
                You are given a group of companies with similar descriptions of what they do. Analyze these companies' descriptions, find trends and insights from them as a group.
                Report any patterns you find from these companies' field of focus, target customers, and why they relate to each other.
                Lastly, give at least 5 examples of companies in this cluster that support your trend and pattern findings.

                Your answer should be in following format:

                \n**Area of focus**: 1 paragraph explaining the area of focus of the companies in this cluster and their target customers
                \n**Trends and Patterns**:
                    - 
                    -
                    -
                \n**Examples**:
                    - company name: company description
                    - company name: company description
                    - company name: company description
                    - company name: company description
                    - company name: company description

                Here are the companies in this cluster for you to analyze:
                {combined_text}"""

                if use_openai:
                    summary = generate_summaries_openai(prompt)
                else:
                    summary = generate_summaries_cohere(prompt)
                
                f.write(f"## Cluster {label} Summary\n")
                f.write(summary)
                f.write("\n\n")
        
    with open(summary_file, 'r') as f:
            summaries = f.read()
    
    return summaries
