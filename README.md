# 🤖 Confluence Q&A Chatbot using LangChain, Groq & FAISS

This project is a Streamlit-based chatbot that enables users to query Confluence pages using natural language. It leverages **LangChain**, **Groq's LLaMA 3.1 model**, and **FAISS** for vector-based document retrieval.

## 🚀 Features

- Connects to Confluence using API credentials
- Loads and splits Confluence documents into chunks
- Embeds documents using HuggingFace embeddings
- Stores and retrieves documents using FAISS vector store
- Uses Groq's LLaMA 3.1 model for answering questions
- Interactive Streamlit UI for querying and viewing results

## 🧠 Technologies Used

- **LangChain**
- **Groq API (LLaMA 3.1)**
- **FAISS**
- **HuggingFace Embeddings**
- **Streamlit**
- **Confluence API**
- **Python**

## 📁 File Structure

- `app.py`: Main Streamlit app
- `faiss_index/`: Saved vector database
- `.env`: Environment variables for API keys

## 🔐 Environment Variables

Create a `.env` file with the following keys:

**ATLASSIAN_API_KEY**=your_confluence_api_key
**GROQ_API_KEY**=your_groq_api_key

## 🛠️ How to Run


1. Clone the repository:
```
git clone https://github.com/yourusername/confluence-chatbot.gitcd confluence-chatbotShow more lines
```

2. Install dependencies:
```
install -r requirements.txtShow more lines
```

3. Add your .env file with API keys.

4. Run the Streamlit app:
```
streamlit run app.py
```

