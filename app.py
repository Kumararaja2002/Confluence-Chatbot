
import time
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import ConfluenceLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()
os.environ['ATLASSIAN_API_KEY'] = os.getenv('ATLASSIAN_API_KEY')
api_key = os.getenv('ATLASSIAN_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['HUGGINGFACE_API_TOKEN']=os.getenv("HUGGINGFACE_API_TOKEN")

# Embeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# LLM
llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")


# Prompt Template
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.

<context>
{context}
<context>

Question: {input}
"""
)

# Function: Create Vectore Store
def create_vector_embeddings():
    if "vectorstore_db" not in st.session_state:
        # Confluence loader
        loader = ConfluenceLoader(
                url="https://your-url.atlassian.net/wiki",  # Base wiki URL
                username="your-username.com",               # Atlassian account email
                api_key=api_key,                            # API token
                # page_ids=["834306143"]
                )
        docs= loader.load()
        
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        splits = text_splitter.split_documents(docs)
        
        # Create FAISS vectorstore DB
        vectorstore_db = FAISS.from_documents(documents=splits, embedding=embeddings)

        # Save to disk
        vectorstore_db.save_local("faiss_index")

        st.session_state.vectorstore_db = vectorstore_db
        st.success("✅ Vector Database created & saved!")

# Load the existing FAISS INdex if available
if os.path.exists("faiss_index"):
    st.session_state.vectorstore_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )

# Streamlit App
st.set_page_config(page_title="Confluence Chatbot", page_icon="💬")
st.title("Confluence Q&A")

user_prompt = st.text_input('Enter your query:')

if st.button("Create Vector Embeddings"):
    create_vector_embeddings()

if user_prompt:
    if "vectorstore_db" not in st.session_state:
        st.warning("Please create the vector databasse first!")
    else:
        # Create retrieval chain
        document_chain= create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectorstore_db.as_retriever(
            search_kwargs={"k":4}
        )

        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start_time = time.process_time()
        response = retrieval_chain.invoke({"input":user_prompt})
        response_time = time.process_time() - start_time
        st.write(f"Response Time:{response_time:.2f} seconds")

        # Display answer
        st.subheader("Answer")
        st.write(response['answer'])

        # Show the retrieved content chunks
        with st.expander("Document Chunks Used"):
            for i, doc in enumerate(response['context']):
                st.markdown(f"**Chunk {i+1}:**\n\n{doc.page_content}")
                st.write(f"Source:{doc.metadata}")