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
pc = Pinecone(api_key = os.getenNV("PINECONE_API_KEY"))
index_name = "stocks"
namespace = "stock-descriptions"
# Connect to the Pinecone index
pinecone_index = pc.Index(index_name)

# initialize OpenAI Client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

# Generate embeddings for the text
def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

# Perform rag
def perform_rag(query):

    # embed the query
    raw_query_embedding = get_huggingface_embeddings(query)

    # find the top matches
    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k = 5, include_metadata=True, namespace="stock-descriptions")

    # get the list of retrieved texts
    context = [item['metadata']['text'] for item in top_matches['matches']]
    for item in top_matches['matches']:
        context.append([item['metadata']['text'], item['metadata']['Sector'], item['metadata']['Industry']])

    # augment the query with the contexts retrieved
    augmented_query = "<CONTEXT>\n" + "\n\n---------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\nMY QUESTION:\n" + query

    # modify the prompt below as needed to improve the quality of the response
    system_prompt = """You are a financial analyst specializing in stock market analysis and decision-making. For each stock-related question I provide, you will:

        Analyze the stock's current performance and potential future trends.
        Identify any notable connections or relationships with other stocks (e.g., industry, market correlation, or shared factors).
        Provide a concise, actionable insight to guide investment decisions.
    """

    llm_response = client.chat.completions.create(
        model = 'llama-3.1-70b-versatile',
        messages=[
            {"role": "system", "content", system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    response = llm_response.choices[0].message.content
    return response