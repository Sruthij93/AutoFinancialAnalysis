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
import utils as ut

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
            'business_summary':match['metadata']['Business Summary'],
            'website':match['metadata']['Website'],
            'revenue_growth':match['metadata']['Revenue Growth'],
            'gross_margins':match['metadata']['Gross Margins'],
            'target_m_price':match['metadata']['Target Mean Price'],
            'current_price':match['metadata']['Current Price'],
            '52weekchange': match['metadata']['52 Week Change'],
            'sector': match['metadata']['Sector'],
            'market_cap': match['metadata']['Market Cap'],
            'volume': match['metadata']['Volume'],
            'recommendation_key': match['metadata']['Recommendation Key'],
            'text':match['metadata']['text']
        })
    return ticker_details


# Generate embeddings for the text
def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

# Augment query with top matches as context
def augment_query_context(query,  top_matches_formatted):
    context = '<CONTEXT>\n'
    for ticker in top_matches_formatted:
        context += f"\n\n--------\n\n {ticker['text']}\n"

    augmented_query = f"{context} \nMY QUESTION:\n {query}"
    return augmented_query

# Perform rag
def perform_rag(query, user_filters):

    # embed the query
    raw_query_embedding = get_huggingface_embeddings(query)

    # apply filter to the metadata
    if user_filters:
         market_cap_min = user_filters['Market Cap min'] * 1_000_000_000
         market_cap_max = user_filters['Market Cap max'] * 1_000_000_000
         volume_min = user_filters['Volume min'] * 1_000_000
         volume_max = user_filters['Volume max'] * 1_000_000
         recommendation_keys = user_filters['Recommendation Keys']

         filter= {"$and": [
            {"Market Cap": {"$gte": market_cap_min, "$lte": market_cap_max}},
            {"Volume": {"$gte": volume_min, "$lte": volume_max}},
            {"52 Week Change": {"$gte": -0.1}},
            {"Recommendation Key": {"$in": recommendation_keys }}
         ]}
    else:      
        filter= {"$and": [
            {"52 Week Change": {"$gt": 0}}, 
            {"Recommendation Key": {"$in": ["buy", "strong buy", "hold"] }},
            ]
        }

    print("filter: ", filter)    
        
    # find the top matches
    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), filter = filter, top_k = 12, include_metadata=True, namespace=namespace)
    top_matches_formatted = format_matches(top_matches)

    print(top_matches)

    augmented_query = augment_query_context(query, top_matches_formatted)


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
    return top_matches_formatted, response

def render_stock_block(stock):
            st.markdown("""<style>.small-font {font-size:12px; color:rgb(128, 128, 128);} a::after { content: none; }</style>""", unsafe_allow_html=True)
            st.markdown("""<style>.number-font {font-size:20px; font-weight:bold;} a::after { content: none;}</style>""", unsafe_allow_html=True)
            with st.container(border = True, height=300):
                st.markdown(f"#### {stock['name']} ({stock['ticker']})")
                st.markdown(f"Website: {stock['website']}", unsafe_allow_html=True)

                cols = st.columns(3)
                with cols[0]:
                    st.markdown("""<div class='small-font'>Revenue Growth</div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div class='number-font'>{stock['revenue_growth']* 100:.1f}%</div>""", unsafe_allow_html=True)
                    st.markdown("""<div class='small-font'>Gross Margins</div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div class='number-font'>{stock['gross_margins'] * 100:.1f}%</div>""", unsafe_allow_html=True)

                with cols[1]:
                    st.markdown("""<div class='small-font'>Market Cap</div>""", unsafe_allow_html=True)
                    market_cap = ut.format_large_number(stock['market_cap'])
                    st.markdown(f"""<div class='number-font'>{market_cap}</div>""", unsafe_allow_html=True)
                    st.markdown("""<div class='small-font'>Volume</div>""", unsafe_allow_html=True)
                    volume = ut.format_large_number(stock['volume'])
                    st.markdown(f"""<div class='number-font'>{volume}</div>""", unsafe_allow_html=True)

                with cols[2]:
                    st.markdown("""<div class='small-font'>Valuation</div>""", unsafe_allow_html=True)
                    valuation = (stock['target_m_price'] - stock['current_price'])/stock['current_price']
                    valuation = ut.format_colored_number(valuation)
                    st.markdown(f"""<div class='number-font'>{valuation}</div>""", unsafe_allow_html=True)
                    st.markdown("""<div class='small-font'>52 Week Change</div>""", unsafe_allow_html=True)
                    weekchange = ut.format_colored_number(stock['52weekchange'])
                    st.markdown(f"""<div class='number-font'>{weekchange}</div>""", unsafe_allow_html=True)

# Main UI
st.title("📉 Automated Stock Analysis")
user_query = st.text_area(
        "Enter a description for the kind of stocks you are looking for:",
        placeholder ="Type here"
    )

with st.expander("Apply filters"):
        # Market Cap Range
        market_cap_range = st.slider(
            "Market Cap (in billions):",
            min_value=0,
            max_value=1000,  
            value=(0, 1000),
            step=10,
            format="$%dB"
        )

        # Volume Range
        volume_range = st.slider(
            "Volume (in millions):",
            min_value=0,
            max_value=1000,  
            value=(0, 1000), 
            step=10,
            format="$%dM"
        )

        # Recommendation Key
        recommendation_keys = ["strong buy", "buy", "hold", "sell", "strong sell"]
        selected_recommendation_keys = st.multiselect(
            "Recommendation Keys:",
            recommendation_keys,
            ["strong buy", "buy", "hold"]
        )

        # print(selected_recommendation_keys)
        # print(type(selected_recommendation_keys))

        user_filters = {
             "Market Cap min" : market_cap_range[0],
             "Market Cap max" : market_cap_range[1],
             "Volume min" : volume_range[0],
             "Volume max" : volume_range[1],
             "Recommendation Keys" : selected_recommendation_keys
        }


if(st.button("Find Stocks")):
    top_matches, results = perform_rag(user_query, user_filters)
    # st.write(top_matches)
    with st.container():
        if top_matches:
            
            cols_main = st.columns(2)
            with cols_main[0]:
                 for match in top_matches[ : 3]:
                      render_stock_block(match)

            with cols_main[1]:
                 for match in top_matches[3:6]:
                      render_stock_block(match)          
                
        else:
            st.write("No stocks found. Please refine your query.")

        st.divider()

        st.write("## More information")    

        st.write(results)    





