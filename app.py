from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
import dotenv
import json
import yfinance as yf
import concurrent.futures
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import numpy as np
import requests
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

#initialize Pinecone
pc = Pinecone(api_key = os.getenv("PINECONE_API_KEY"))
index_name = "stocks"
namespace = "stock-description_detailed"
# Connect to the Pinecone index
pinecone_index = pc.Index(index_name)

# initialize OpenAI Client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

def format_matches(top_matches):
    ticker_details = []
    for match in top_matches['matches']:
        ticker_details.append({
            'ticker': match['metadata']['Ticker'],
            'name': match['metadata']['Name'],
            'sector': match['metadata']['Sector'],
            'market_cap': match['metadata']['Market Cap'],
            '52weekchange': match['metadata']['52 Week Change'],
            'recommendation_key': match['metadata']['Recommendation Key'],
            'business_summary':match['metadata']['Business Summary'],
            'text':match['metadata']['text']
        })
    return ticker_details
        
def augment_query_context(query,  top_matches_formatted):
    context = '<CONTEXT>\n'
    for ticker in top_matches_formatted:
        context += f"\n\n--------\n\n {ticker['text']}\n"

    augmented_query = f"{context} \nMY QUESTION:\n {query}"
    return augmented_query


# Generate embeddings for the text
def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

# Perform rag
def perform_rag(query):

    # embed the query
    raw_query_embedding = get_huggingface_embeddings(query)

    # apply filter to the metadata
    filter= {"$and": [
        {"52weekchange": {"$gt": 0}}, 
        {"recommendation_key": {"$in": ["buy", "hold"] }}
        ]
    }
        
    # find the top matches
    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k = 20, filter = filter, include_metadata=True, namespace=namespace)
    top_matches_formatted = format_matches(top_matches)

    augmented_query = augment_query_context(query, top_matches_formatted)

    # modify the prompt below as needed to improve the quality of the response
    system_prompt = """You are an expert at providing answers about stocks. If ticker symbols are mentioned, give more info about that stock. Please answer my question provided.
    """

    llm_response = client.chat.completions.create(
        model = 'llama-3.1-70b-versatile',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    response = llm_response.choices[0].message.content
    return response


# Main UI
st.title("Automated Stock Analysis")
user_query = st.text_area(
        "Enter a description for the kind of stocks you are looking for:",
        placeholder ="Type here"
    )

if(st.button("Find Stocks")):
    st.write(perform_rag(user_query))





