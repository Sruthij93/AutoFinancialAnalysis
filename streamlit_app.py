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
        context += f"\n\n--------\n\n {ticker['text']}\n Sector is {ticker['sector']}. \n Market Cap is {ticker['market_cap']}. \n Volume is {ticker['volume']}."
    print(context)
    augmented_query = f"{context} \nMY QUESTION:\n {query}"
    return augmented_query

# Perform rag
def perform_rag(query, user_filters):
    # embed the query
    raw_query_embedding = get_huggingface_embeddings(query)

    # apply filter to the metadata
    if user_filters:
        
         market_cap= user_filters['Market Cap']
         volume = user_filters['Volume']
         recommendation_keys = user_filters['Recommendation Keys']
        
         filter= {"$and": [
            {"Market Cap": {"$gte": market_cap}},
            {"Volume": {"$gte": volume}},
            {"52 Week Change": {"$gte": -0.2}},
            {"Recommendation Key": {"$in": recommendation_keys }}
         ]}
    else:      
        filter= {"$and": [
            {"52 Week Change": {"$gt": 0}}, 
            {"Recommendation Key": {"$in": ["buy", "strong buy", "hold"] }},
            ]
        }

    # print("filter: ", filter)    
        
    # find the top matches
    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), filter = filter, top_k = 12, include_metadata=True, namespace=namespace)
    top_matches_formatted = format_matches(top_matches)

    print(top_matches)

    augmented_query = augment_query_context(query, top_matches_formatted)


    system_prompt = """You are a financial expert at providing answers about stocks. Please answer my question provided.
            Analyze the stocks' in detail and explain current performance and potential future trends.
            Identify any notable connections or relationships with other stocks (e.g., industry, market correlation, or shared factors).
            Provide a concise, actionable insight to guide investment decisions.
    """
    try:
        llm_response = client.chat.completions.create(
            model = 'llama-3.1-70b-versatile',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ]
        )
    except:
        llm_response = client.chat.completions.create(
                model = 'llama-3.1-8b-instant',
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
                    st.markdown(f"""<div class='number-font'>{ut.safe_format(stock['revenue_growth'])}</div>""", unsafe_allow_html=True)
                    st.markdown("""<div class='small-font'>Gross Margins</div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div class='number-font'>{ut.safe_format(stock['gross_margins'])}</div>""", unsafe_allow_html=True)

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
st.title("📉 StockLit")
st.caption("Get Automated Stock Analysis done right here!")
user_query = st.text_area(
        "Enter a description for the kind of stocks you are looking for:",
        placeholder ="Type here"
    )

with st.expander("Apply filters"):
        # Market Cap Range
        market_cap = st.number_input(
             "Market Cap",
             min_value=0,
             max_value=10000000000000,
             value= 1000000,
             step = 1000
        )

        # Volume Range
        volume = st.number_input(
            "Volume",
            min_value=0,
            max_value=1000000000,  
            value=10000, 
            step=100
        )

        # Recommendation Key
        recommendation_keys = ["strong buy", "buy", "hold", "sell", "strong sell"]
        selected_recommendation_keys = st.multiselect(
            "Recommendation Keys:",
            recommendation_keys,
            ["strong buy", "buy", "hold"]
        )


        user_filters = {
             "Market Cap" : market_cap,
             "Volume": volume,
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

        st.write("## Analysis")    

        st.write(results)    





