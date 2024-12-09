# 🧑‍💻 **Automated Financial Analysis: StockLit**

This project is an interactive Stock Financial Analysis Application that allows users to query a Pinecone-powered vector database containing detailed stock information. Users can apply filters such as market cap, volume, and recommendation keys to identify stocks matching their criteria. The app utilizes advanced embeddings and retrieval-augmented generation (RAG) to provide detailed responses.

---

## 🚀 **Features**

- **Interactive UI**: Built with Streamlit, enabling users to query stocks and apply filters with ease.
- **Vector Search**: Powered by Pinecone to retrieve relevant stock data based on metadata and vector embeddings.
- **RAG-based Analysis**: Uses OpenAI LLMs for augmented answers based on query context.
- **Custom Filters**: Users can filter results by market cap, volume, and recommendation keys.
- **Visualization**: Displays key financial metrics such as revenue growth, gross margins, valuation, and more.

---

## 🛠️ **Tech Stack**

- **Python**: Core language for implementation.
- **Hugging Face Transformers**: Used for generating embeddings with the `sentence-transformers/all-mpnet-base-v2` model.
- **Pinecone**: A vector database to manage and query the vector database.
- **Streamlit**: Frontend framework to provide an interactive and user-friendly UI.
- **OpenAI LLMs**: For generating accurate, context-aware responses.
- **YFinance**: For fetching real-time financial data.

---

## ⚙️ **How it Works**

### 1. Get Stock info using Yahoo Finance API

Using parallel processing, stock info is collected from Yahoo Finance and stored in Pinecone.

### 2. User Input

Users describe the type of stocks they are looking for in natural language via the Streamlit interface.
Optional filters like Market Cap, Volume, Recommendation Keys can also be applied for more tailored results.

### 3. Embedding Generation

The query is converted into a high-dimensional vector representation using HuggingFace Sentence-Transformers. This ensures that the search captures the semantic meaning of the user's input.

### 4. Vector Search with Pinecone

The generated embedding is sent to Pinecone, where it searches a vector database containing stock information.
Filters are applied to narrow down the results based on user preferences, ensuring only the most relevant stocks are retrieved.

### 5. Retrieval-Augmented Generation (RAG)

The retrieved stock information is formatted into a context block.
This context, combined with the original query, is sent to OpenAI's language model to generate an augmented, detailed response.

### 6. Rendering Results

Top matches are displayed in an interactive format with key financial metrics like Revenue Growth, Gross Margins, Valuation, Market Cap, and more.
The app handles missing or malformed data gracefully to avoid errors while rendering.

---

## 📋 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Sruthij93/AutoFinancialAnalysis
cd stock-analysis-app
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Pinecone and Groq

- Create a Pinecone account at Pinecone.io.
- Get your Pinecone API key and index name.
- Get your Groq API as well.
- Configure your .env file with Pinecone and Groq credentials.

### 4. Run the Application

```bash
streamlit run streamlit_app.py
```

---

## 🌟 Next Steps

- Plan to process news articles in real-time and perform sentiment analysis on companies mentioned within these articles. By determining whether the sentiment is positive or negative, this will provide insights into the nature of events being reported about each company.
- Add more UI elements like stock trends.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for discussion.
