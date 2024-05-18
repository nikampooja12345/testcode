from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from langchain.document_loaders import UnstructuredURLLoader

# Directories and file paths

# title
st.title("Automated Scheme Research Tool.")
st.title("Pm Kisan Yojna")


def load_urls(file_path=None, urls=None):
    if file_path:
        with open(file_path, 'r') as file:
            urls = file.readlines()
        urls = [url.strip() for url in urls]
    return urls

urls = load_urls(urls=['https:/www.google.com'])  # Exampleor loading from a text file
# urls = load_urls(urls=['https://example.com/article1', 'https://example.com/article2'])  # Example for direct URL input


def fetch_articles(urls):
    loader = UnstructuredURLLoader(urls)
    articles = loader.load()
    return articles

articles = fetch_articles(urls)


import openai

openai.api_key = 'your_openai_api_key'

def get_embeddings(texts):
    response = openai.Embedding.create(
      input=texts,
      engine="text-embedding-ada-002"  # Example embedding engine
    )
    embeddings = [embedding['embedding'] for embedding in response['data']]
    return embeddings

texts = [article['content'] for article in articles]
embeddings = get_embeddings(texts)


import faiss
import numpy as np

def create_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype(np.float32))
    return index

index = create_faiss_index(embeddings)

def search_faiss(index, query_embedding, top_k=5):
    distances, indices = index.search(np.array([query_embedding]).astype(np.float32), top_k)
    return distances, indices

query = "Your search query here"
query_embedding = get_embeddings([query])[0]
distances, indices = search_faiss(index, query_embedding)

def get_relevant_articles(indices, articles):
    return [articles[idx] for idx in indices[0]]

relevant_articles = get_relevant_articles(indices, articles)

# Generate summaries and display results
for article in relevant_articles:
    summary = article['content'][:200]  # Example: first 200 characters as summary
    url = article['source']
    print(f"URL: {url}\nSummary: {summary}\n")

def query_articles(query, file_path=None, urls=None):
    urls = load_urls(file_path, urls)
    articles = fetch_articles(urls)
    texts = [article['content'] for article in articles]
    embeddings = get_embeddings(texts)
    index = create_faiss_index(embeddings)
    query_embedding = get_embeddings([query])[0]
    distances, indices = search_faiss(index, query_embedding)
    relevant_articles = get_relevant_articles(indices, articles)
    results = []
    for article in relevant_articles:
        summary = article['content'][:200]
        url = article['source']
        results.append({'url': url, 'summary': summary})
    return results

# Example Usage
query = "Latest advancements in AI technology"
results = query_articles(query, file_path='urls.txt')
for result in results:
    print(f"URL: {result['url']}\nSummary: {result['summary']}\n")




# Image
st.image("pm_kisan.png", width=600)


st.write ( '''The Pradhan Mantri Kisan Samman Nidhi (PM-Kisan) is a Central Sector Scheme that provides income support to the families of landholding farmers in India. 
          This scheme provides supplemental financial support to the farmers to procure various inputs related to agriculture and allied activities and their domestic needs.
PM-Kisan provides income support to all landholding farmers’ families who have cultivable lands. 
Under this scheme, 100% of funding is provided by the Government of India. 
It aims to supplement the financial needs of farmers in obtaining agriculture inputs for ensuring proper crop health and appropriate yield.
The State Government and UT administration identify the farmer families eligible for financial support under the scheme guidelines. 
After the beneficiaries are identified, the funds are directly transferred to their bank accounts under this scheme.''')



Menu = st.sidebar.radio("MENU",["Process","Scheme Benefits", "Eligibility","Documents required"])    

if Menu == "process":
    st.title("Application process of PMKY")
    st.image("app_process.png", width=400)
    
st.write('''One would need to first visit PMKSNY's official website and click on “New Farmer Registration” within the Farmer's Corner section. 
         Farmers who self-register and enrol via a CSC can check their PM Kisan Samman Nidhi Yojana status by clicking on the
         “Status of Self-registered/CSC farmers” option under Farmer's Corner.''')
    
if Menu == "Scheme Benefits":
    st.title("Scheme Benefits of PMKY")
    st.image("image.png",width=400)
    

st.write(" PM-Kisan provides financial aid of Rs. 6,000 per year in three installments of Rs. 2,000 each to landholding farmer families")
    
if Menu == "Eligibility":
    st.title("Scheme Eligibility of PMKY")
    st.image("eligibility.jpg",width=400)
    
st.write('''One of the most crucial features of this government scheme is its eligibility criteria. 
         Farmer families that qualify these criteria can benefit from this Yojana: Small and marginal farmers are eligible for PMKSNY. 
Farmer families that hold cultivable land can apply for the benefits of this plan.
A beneficiary should be an Indian citizen.Alongside these, farmers from both rural and urban regions can be enlisted under 
the Pradhanmantri Kisan Samman Nidhi Yojana. However, its guidelines also exclude certain categories of individuals from its beneficiary list.''')    


if Menu == "Documents required":
    st.title("Documents Required of PMKY")
    st.image("documents.jpeg",width=400)
st.write('''Documents Required to Register for PM Kisan Samman Nidhi Yojana 
    Aadhar Card.
    Domicile Certificate.
    Documents proclaiming ownership of land.
    Bank Account Details.''')