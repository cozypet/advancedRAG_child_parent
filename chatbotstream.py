import streamlit as st
import os
from pymongo import MongoClient
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.llms import OpenAI as LangChainOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.prompts import PromptTemplate
from openai import OpenAI
import time
import tempfile

# Initialize the OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# Set up MongoDB connection details
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MONGO_URI = os.environ["MONGO_URI"]
DB_NAME = "pdfchatbot"
COLLECTION_NAME = "advancedRAGParentChild"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

# Initialize OpenAIEmbeddings with the API key
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Define field names
EMBEDDING_FIELD_NAME = "embedding"
TEXT_FIELD_NAME = "text"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Define Parent Child splitters
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

# Initialize InMemoryStore for parent documents
parent_store = InMemoryStore()

# Function to process PDF document
def process_pdf(uploaded_file):
    st.write("Starting PDF processing...")

    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Write the content of the uploaded file to the temporary file
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Use the temporary file path with PyPDFLoader
        loader = PyPDFLoader(tmp_file_path)  # Use tmp_file_path instead of uploaded_file.name
        docs = loader.load()

        parent_docs = parent_splitter.split_documents(docs)
        for parent_doc in parent_docs:
            parent_doc_content = parent_doc.page_content  # Extract the text content
            parent_doc_content = parent_doc_content.replace('\n', ' ')
                #parent_embedding = embeddings.embed_documents([parent_doc_content])[0]
            parent_id = collection.insert_one({
                'document_type': 'parent', 
                'content': parent_doc_content,  # Store the text content
                #'embedding': parent_embedding
            }).inserted_id

            child_docs = child_splitter.split_documents([parent_doc])
            for child_doc in child_docs:
                child_doc_content = child_doc.page_content  # Extract text content from child_doc
                child_doc_content = child_doc_content.replace('\n', ' ')
                child_embedding = embeddings.embed_documents([child_doc_content])[0]
                collection.insert_one({
                    'document_type': 'child', 
                    'content': child_doc_content,  # Store text content
                    'embedding': child_embedding,
                    'parent_ref': parent_id
                })
        os.remove(tmp_file_path)  # Use tmp_file_path for the file path
        st.write("PDF processing complete")
        return "PDF uploaded"
    else:
        st.write("No file uploaded.")

def query_and_display(query):

    if not isinstance(query, str):
        query = str(query)
    try:
        query_embedding = embeddings.embed_documents([query])[0]
    except Exception as e:
        print(f"Error during embedding: {e}")
        return "An error occurred during processing."

    # Retrieve relevant child documents based on query
    child_docs = collection.aggregate([{
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 10,
            "limit": 1,
            "filter": {"document_type": "child"}
            }
        },
        {"$project": {"embedding": 0}}
    ])
    child_docs_list = list(child_docs)
    
    # Fetch corresponding parent documents
    parent_docs = []

    for doc in child_docs_list:
        #print(doc['parent_ref'])
        parent_doc = collection.find_one({"_id": doc['parent_ref']})
        #print(parent_doc)
        parent_docs.append(parent_doc)
    #print(parent_docs)
    # Concatenate parent document content for response generation
    
    response_content = " ".join([doc['content'] for doc in parent_docs if doc])
    
    print(response_content)
    # Add the instruction to the response content
    additional_instruction = "You are a helpful AI retrieval augmented generation chatbot, please use only provided information to generate response. If the question is not relevant, summarize only the provided text."
    response_content = f"{response_content} {additional_instruction}"

    # Use the OpenAI client for LLM requests
    chat_completion = openai_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": query,
            },
            {
                "role": "system",
                "content": response_content,  # Response content generated from your database
            }
        ],
        model="gpt-3.5-turbo",  # Specify the model here
        temperature=0,
    )
    response_text=chat_completion.choices[0].message.content
    
    # Prepare the display content
    parent_docs_content = "\n\n".join([f"Parent Doc: {doc['content']}" for doc in parent_docs if doc])
    
    child_docs_content = "\n\n".join([f"Child Doc: {doc['content']}" for doc in child_docs_list])
    display_content = f"LLM Response: {response_text}\n\n{child_docs_content}\n\n{parent_docs_content}"
    #display_content = f"LLM Response: {response_text}\n\n{child_docs_content}\n"

    return display_content


# Streamlit UI setup
st.title("Generative AI Chatbot")

# Upload PDF
st.subheader("Upload PDF")
pdf_file = st.file_uploader("Choose a PDF file", type="pdf")
if pdf_file is not None:
    process_pdf(pdf_file)

# Ask question
st.subheader("Ask a question")
question = st.text_input("Your Question")
if st.button("Ask"):
    response = query_and_display(question)
    st.text_area("Response", response, height=250)