# advancedRAG_child_parent

https://ai.gopubby.com/byebye-basic-rag-embracing-advanced-retrieval-with-mongodb-vector-search-47b550be2c59
---

Byebye Basic RAG: Embracing Advanced Retrieval with MongoDB Vector Search
Parent-Child Relationships and Vectorization
# Introduction
Imagine you're a detective in the enormous world of information, trying to find that one vital clue hidden in a mountain of data. This is where Retrieval Augmented Generation (RAG) comes into play, acting like your reliable assistant in the world of AI and language models. But as we've seen in my previous article, even the best assistants have their limits. So, let's set off on an adventure to explore the need for an advanced RAG approach, focusing on precision and context in large-scale document retrieval.
# Basic RAG: A Compass Needing Calibration
Imagine an encyclopedia as wide as the horizon itself. Basic RAG attempts to distill this vast knowledge into a single 'embedding' - essentially a digital essence. But when you seek wisdom on a particular subject, like the enigmatic Bermuda Triangle, basic RAG's broad strokes paint over the finer details, leaving you with an incomplete picture. This limitation is like trying to find a hidden treasure with a map that only shows continents, not the intricate paths that lead to the spot marked 'X'.
Basic RAG - by authorOn the flip side, if we possess only isolated pages with their embeddings, we might pinpoint specific facts, but we lose the story they tell when woven together. Without the narrative, the Large Language Model (LLM) struggles to craft an answer that captures the true essence of our inquiry.
Basic RAG, while a commendable guide, falls short of the mark. It's a foundation, but to traverse the distances between general knowledge and precise insights, we need more.




Refining the Path: The Advent of Parent-Child Document Dynamics
Here, our adventure takes a significant leap forward with the concept of parent-child document relationships. Instead of a single summary of the entire encyclopedia, we curate concise overviews for each page (child documents), mindful of the encompassing chapter (parent document). This approach crafts detailed 'digital essences' for each page, which act as precise signposts in our search for knowledge.
In this landscape, the parent document retriever emerges as a quiet guardian. This tool dissects the colossal encyclopedia into approachable chapters and pages, each marked with its distinct digital essence. Inquiring about the Bermuda Triangle now brings forth the most pertinent page, but crucially, it also offers the entire chapter. It's akin to a sage who not only deciphers the riddle you pose but also provides the lore and legends that give it meaning.
The Need for a More Refined Approach
Basics RAG is great for getting a general idea, but when it comes to the specifics, the details get lost in the mix. In our detective story of data exploration, this is necessary to have a map that shows you the entire country when you're looking for a specific street in a small town. The embedding created by basic RAG for large documents becomes a blurred representation, making it difficult to pinpoint the exact information needed.
Advanced RAG: A Dive into Parent-Child Relationships and Vectorization
In the expanding universe of AI, we've seen how basic RAG can sometimes get lost in the forest of data when trying to pinpoint the tree of truth. But fear not, because advanced RAG, with its parent-child relationships and vectorization, is here to turn over a new leaf. Let's explore these concepts using the diagrams provided, which illustrate the process of making sense of large volumes of information and retrieving just what we need.

Step 1: Parent-Child Document Relationships
Imagine you have a large, unwieldy book - a user manual for every appliance ever made. Now, someone asks a specific question: "Why is my washing machine displaying error code 2?" With basic RAG, either we get small chunks which lacks of context either we get big chunks the search is not accurate enough. Advanced RAG, however, takes a smarter approach.
Firstly, the manual is broken down into big chunks - these are our 'parent' documents. Each section deals withlarger infomation. Within these sections, we split 'child' documents that cover specific issues, like error codes for washing machines.
Now, to the magic of vectorization. Each child document is processed through an embedding model, which analyzes the text and converts it into a vector - a series of numbers that represents the essence of the text. It's like creating a DNA profile for each small piece of information.
Each child document's text is distilled into a vector, which is then stored in vector store and its parent is also stored in this vector store which is a general database as well. This allows us to not only retrieve the most relevant small piece of information quickly but also to retain the context provided by the parent document.

Step 2: question and answering
When the question about the washing machine comes in, it's transformed into an 'embedding' - think of it as a unique digital signature. This embeddings is then matched against similar child document using a vector search. If it aligns closely with the embedding of a child document from the 'washing machine' section, we've got our match.
With our vectors stored and ready, when a question comes in, we can swiftly find the most relevant child document. But instead of providing a narrow response, we bring in the parent document, which offers more background and context. This prepared prompt, rich in specific information and broad context, is then fed into a Large Language Model (LLM), which generates a precise and context-aware answer.

This advanced RAG process, illustrated by the diagrams, ensures that the LLM has all the context it needs to generate an accurate response, much like a detective piecing together clues to solve a mystery. With the power of MongoDB vector search, we can navigate through this process with the speed and precision of a supercomputer, ensuring that every question is met with the best possible answer.
MongoDB Vector Search: The Powerhouse Behind Advanced RAG
Moving from the intricate dance of parent-child relationships and vectorization, we land squarely in the domain of MongoDB's vector search, the engine that powers our advanced RAG process. Let's delve into how MongoDB vector search turns the laborious task of sifting through mountains of data into a streamlined, efficient process.
Vector Search: The High-Speed Chase for Answers
Vector search in MongoDB is like having a high-powered searchlight in the vast ocean of data. When our washing machine aficionado asks about that pesky error code, vector search doesn't just comb through the data - it hones in on the exact location of the information, thanks to the unique 'digital signatures' we created earlier. The best part? It does this with astonishing speed, making the search for answers as fast as flipping through a well-organized filing cabinet.
The Symbiosis of Structure and Speed
MongoDB's vector search brings structure and speed together in harmony. The storage of parent and child documents, complete with their vectorized essence, allows MongoDB to quickly identify the most relevant pieces of data without getting bogged down by the less pertinent information. It's the perfect combination of a meticulous librarian and a master detective, ensuring that no clue is missed and every answer is on point.
Contextual Richness: The Added Layer
Here's where things get even more interesting. Once the vector search pinpoints the relevant child document, it doesn't stop there. By retrieving the parent document, it ensures that the richness of context is not lost. This means that our LLM doesn't just understand the 'what' but also the 'why' and 'how,' providing answers that go beyond the surface level.
MongoDB: More Than Just a Database
MongoDB is not merely a place to store data; it's a dynamic ecosystem that supports the advanced RAG process every step of the way. It manages the complex web of parent and child documents with ease and facilitates the rapid vector search that makes advanced RAG so powerful. With MongoDB, we're not just searching for answers; we're crafting responses that are as informative as they are contextually relevant.
The Result: Informed, Accurate Responses
As a result of this powerful collaboration between advanced RAG and MongoDB vector search, the responses generated are not only accurate but also richly informative. When our user asks about the error code on their washing machine, they receive a response that is both precise and filled with useful context, akin to a comprehensive guide tailored just for them.
MongoDB vector search is the backbone of this advanced RAG process, providing the speed and precision necessary to navigate the complex landscape of data retrieval. In the next section, we'll explore the practical implementation of this process, demonstrating how the advanced RAG system can be brought to life to provide users with the best answers AI can offer. Stay tuned as we translate theory into practice and bring the full potential of advanced RAG to the fore.
Implementing Advanced RAG with MongoDB Vector Search: A Code Perspective
Integrating Advanced RAG with MongoDB Vector Search into our systems begins with the harmonious blend of several technical components and a well-orchestrated flow of data processing. Let's walk through the steps, integrating your provided code into our explanation for clarity.
Step 1: Setting the Stage with Initializations
We kick things off by setting up our environment and establishing the necessary connections. This involves loading environment variables, initializing the OpenAI and MongoDB clients, and defining our database and collection names.
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain.embeddings import OpenAIEmbeddings
# Load environment variables from .env file
load_dotenv(override=True)
# Set up MongoDB connection details
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MONGO_URI = os.environ["MONGO_URI"]
DB_NAME = "pdfchatbot"
COLLECTION_NAME = "advancedRAGParentChild"
# Initialize OpenAIEmbeddings with the API key
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
Step 2: Data Load and Chunking
Next, we focus on processing the PDF document, which serves as our data source. The document is loaded and split into 'parent' and 'child' chunks to prepare for embedding and vectorization.
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Initialize the text splitters for parent and child documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
# Function to process PDF document and split it into chunks
def process_pdf(file):
    loader = PyPDFLoader(file.name)
    docs = loader.load()
    parent_docs = parent_splitter.split_documents(docs)
    
    # Process parent documents
    for parent_doc in parent_docs:
        parent_doc_content = parent_doc.page_content.replace('\n', ' ')
        parent_id = collection.insert_one({
            'document_type': 'parent',
            'content': parent_doc_content
        }).inserted_id
        
        # Process child documents
        child_docs = child_splitter.split_documents([parent_doc])
        for child_doc in child_docs:
            child_doc_content = child_doc.page_content.replace('\n', ' ')
            child_embedding = embeddings.embed_documents([child_doc_content])[0]
            collection.insert_one({
                'document_type': 'child',
                'content': child_doc_content,
                'embedding': child_embedding,
                'parent_ref': parent_id
            })
    return "PDF processing complete"
Step 3: Query Embedding and Vector Search
When a query is submitted, we convert it into an embedding and perform a vector search to find the most relevant child documents, linking back to their parents for context.
# Function to embed a query and perform a vector search
def query_and_display(query):
    query_embedding = embeddings.embed_documents([query])[0]
    
    # Retrieve relevant child documents based on query
    child_docs = collection.aggregate([{
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 10
        }
    }])
    
    # Fetch corresponding parent documents for additional context
    parent_docs = [collection.find_one({"_id": doc['parent_ref']}) for doc in child_docs]
    return parent_docs, child_docs
Step 4: Response Generation with Contextual Awareness
With the relevant documents identified, we create a prompt for the LLM that includes both the user's query and the content from the matched documents. This ensures that the response is informative and contextually relevant.
from langchain.llms import OpenAI
# Initialize the OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)
# Function to generate a response from the LLM
def generate_response(query, parent_docs, child_docs):
    response_content = " ".join([doc['content'] for doc in parent_docs if doc])
    chat_completion = openai_client.chat.completions.create(
        messages=[{"role": "user", "content": query}],
        model="gpt-3.5-turbo"
    )
    return chat_completion.choices[0].message.content
Step 5: Bringing It All Together
Finally, we combine these elements into a coherent interface where users can upload documents and ask questions. This is brought to life using Gradio, providing a user-friendly way to interact with our advanced RAG system.
with gr.Blocks(css=".gradio-container {background-color: AliceBlue}") as demo:
    gr.Markdown("Generative AI Chatbot - Upload your file and Ask questions")

    with gr.Tab("Upload PDF"):
        with gr.Row():
            pdf_input = gr.File()
            pdf_output = gr.Textbox()
        pdf_button = gr.Button("Upload PDF")

    with gr.Tab("Ask question"):
        question_input = gr.Textbox(label="Your Question")
        answer_output = gr.Textbox(label="LLM Response and Retrieved Documents", interactive=False)
        question_button = gr.Button("Ask")

    question_button.click(query_and_display, inputs=[question_input], outputs=answer_output)
    pdf_button.click(process_pdf, inputs=pdf_input, outputs=pdf_output)

demo.launch()
Step 6: Index creation on MongoDB Atlas
{
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    },
    {
      "path": "document_type",
      "type": "filter"
    }
  ]
}
Conclusion
And so, our digital detective tale reaches its end, with our AI sidekick evolving into a data-savvy hero. Thanks to advanced RAG and MongoDB Vector Search, we've gone from rummaging through data with a flashlight to illuminating insights with a high-beam spotlight.
So, cheers to the new dynamic duo of AI - the advanced RAG and MongoDB Vector Search. Together, they turn every one of us into the sharp detectives of the information age, ready to unearth the treasures of knowledge hidden in plain sight. And who knows? In our next adventure, we might just crack the code to the ultimate question: Do we always need semantic search in Generative AI? But that, my friends, is a story for another day.

---

Before you go! 🦸🏻‍♀️
If you liked my story and you want to support me:
Clap my article 50 times, that will really really help me out.👏
Follow me on Medium and subscribe to get my latest article🫶
Follow me on my LinkedIn to get other information about data 🔭

If you are interested in the topic, there are more articles you can read.
Behind Gen AI project: A Comprehensive LLM Technologies Costs Analysis
For Business decision-makers, Enterprise Architects and Developersmedium.com
From Zero to Hero: Building a Generative AI Chatbot with MongoDB and Langchain
Your Complete Beginner guide from what to why to howai.gopubby.com
AutoGen and MongoDB Magic: Your Beginner Guide to Setup an Enterprise-level Retrieval Agent
Step by step on creating your MongoDB Retrieval agent in the AutoGen frameworkmedium.com
MongoDB and Langchain Magic: Your Beginner's Guide to Setting Up a Generative AI app with Your Own…
Introduction:medium.com
