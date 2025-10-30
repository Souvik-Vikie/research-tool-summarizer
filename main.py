import os
import streamlit as st
import pickle
import time
import requests
from bs4 import BeautifulSoup
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from dotenv import load_dotenv
load_dotenv()

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Initialize Hugging Face models
@st.cache_resource
def initialize_hf_models():
    # Initialize tokenizer and model for text generation (using FLAN-T5 which is free and good for Q&A)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    
    # Create pipeline
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=500,
        temperature=0.9,
    )
    
    # Create LangChain wrapper
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

llm = initialize_hf_models()

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_hf.pkl"

main_placeholder = st.empty()

def extract_text_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'ads']):
            tag.decompose()
        
        # Extract text from paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        return text
    except Exception as e:
        st.error(f"Error processing {url}: {str(e)}")
        return None

if process_url_clicked:
    # Validate URLs
    valid_urls = [url for url in urls if url.strip() != ""]
    if not valid_urls:
        st.error("Please enter at least one valid URL")
    else:
        try:
            # load data
            main_placeholder.text("Data Loading...Started...✅✅✅")
            documents = []
            
            for url in valid_urls:
                text = extract_text_from_url(url)
                if text:
                    doc = Document(page_content=text, metadata={"source": url})
                    documents.append(doc)
            
            if not documents:
                st.error("Could not fetch content from any of the URLs. Please check if the URLs are accessible.")
                st.stop()
                
            # split data
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(documents)
            
            if not docs:
                st.error("No text content could be extracted from the URLs.")
                st.stop()
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.stop()
    # create embeddings and save it to FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)




