import streamlit as st
import json
import numpy as np
from elasticsearch import Elasticsearch
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
import os
from langchain.schema import Document
from langchain.schema import BaseRetriever
from typing import Any, List
from pydantic import BaseModel
from elasticsearch.helpers import bulk

st.title("FCRA Query System")
st.write("Welcome! You can ask questions related to the FCRA.")

# Load saved embeddings and metadata
with open('src/document_embeddings.json', 'r') as f:
    saved_data = json.load(f)

# Convert the embeddings back to numpy arrays (if needed)
texts = []
embeddings = []
metadatas = []

for item in saved_data:
    texts.append(item['text'])
    embeddings.append(np.array(item['embedding']))  # Convert back to numpy array
    metadatas.append(item['metadata'])

# Load environment variables
load_dotenv()

# Initialize the embedding model with the API key
embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Elasticsearch connection
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Define the Elasticsearch index name
index_name = 'fcra_chunks'

# Ensure the Elasticsearch index exists
if not es.indices.exists(index=index_name):
    st.error("Elasticsearch index not found! Indexing the documents.")
    # Define the mapping
    mapping = {
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "dense_vector",
                    "dims": 1536
                },
                "text": {
                    "type": "text"
                },
                "metadata": {
                    "type": "object",
                    "enabled": True
                }
            }
        }
    }

    # Create the index with the mapping
    es.indices.create(index=index_name, body=mapping)

    # Prepare actions for bulk indexing
    actions = [
        {
            "_index": index_name,
            "_id": str(i),
            "_source": {
                "embedding": embedding.tolist(),  # Convert numpy array to list
                "text": text,
                "metadata": metadata
            }
        }
        for i, (embedding, text, metadata) in enumerate(zip(embeddings, texts, metadatas))
    ]

    # Bulk index the documents
    bulk(es, actions)

# Add a dropdown for selecting question type
question_type = st.selectbox("Select Question Type", ["Simple Question", "True/False"])

# Input box for user query
query = st.text_input("Enter your question related to the FCRA:", "")

# Button to trigger the query
if st.button("Get Answer"):
    if query:
        # Check question type and adjust the prompt
        if question_type == "Simple Question":
            formatted_query = query
        elif question_type == "True/False":
            formatted_query = f"{query}\nIs this statement True or False?"
        
        # Process the query
        st.write(f"Processing query: {formatted_query}")
    else:
        st.error("Please enter a query.")

# Initialize the LLM (GPT)
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0, max_tokens=256)

# Create a Custom Retriever Using Elasticsearch
class ElasticSearchRetriever(BaseRetriever, BaseModel):
    es: Any
    index_name: str
    embedding_model: Any
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Generate and normalize the query embedding
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Build the script score query
        script_query = {
            "size": self.k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {
                            "query_vector": query_embedding.tolist()
                        }
                    }
                }
            },
            "_source": ["text", "metadata"]
        }

        # Execute the search
        response = self.es.search(index=self.index_name, body=script_query)

        # Convert hits to Documents
        docs = []
        for hit in response['hits']['hits']:
            doc = Document(
                page_content=hit['_source']['text'],
                metadata=hit['_source']['metadata']
            )
            docs.append(doc)
        return docs

# Initialize the Retriever
retriever = ElasticSearchRetriever(
    es=es,
    index_name=index_name,
    embedding_model=embedding_model,
    k=5  # Number of documents to retrieve
)

# Define a custom prompt template that encourages the model to use only the retrieved documents
if question_type == "Simple Question":
    prompt_template = """
    You are a helpful assistant that answers questions based on the following documents:

    {context}

    If the answer is not in the documents, respond with "I don't know based on the information provided."
    Question: {question}
    """
else:
    prompt_template = """
    You are a helpful assistant answering only based on the following documents:

    {context}

    Statement: {question}

    Answer "True" or "False" **and provide a brief explanation** based only on the provided documents. Do not use any knowledge outside the documents. 
    If the answer cannot be determined based on these documents alone, respond with "I don't know based on the information provided."
    """

# Create a prompt object using the custom prompt template
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

if query:
    with st.spinner("Retrieving answer..."):
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(formatted_query)
        
        # Check if relevant documents were retrieved
        if not docs:
            st.success("Answer: I don't know based on the information provided.")
        else:
            # Run the query using the LLM chain
            answer = qa_chain.run(formatted_query)

            # Post-process the answer
            if "I don't know based on the information provided" in answer:
                st.success("Answer: I don't know based on the information provided.")
            else:
                # Split the answer into True/False and explanation
                answer_lines = answer.strip().split('\n', 1)
                if len(answer_lines) == 2:
                    tf_answer, explanation = answer_lines
                else:
                    tf_answer = answer.strip()
                    explanation = ""
                
                # Display the answer and explanation
                st.success(f"Answer: {tf_answer}")
                if explanation:
                    st.write(f"Explanation: {explanation}")

# # If query is entered, process it and get the answer
# if query:
#     with st.spinner("Retrieving answer..."):
#         answer = qa_chain.run(formatted_query)

#         # Check if the answer contains "I don't know" and format the output accordingly
#         if "I don't know based on the information provided" in answer:
#             st.success("Answer: I don't know based on the information provided.")
#         elif answer.strip().lower() in ["true", "false"]:
#             st.success(f"Answer: {answer}")
#         else:
#             st.success("Answer: I don't know based on the information provided.")

st.sidebar.title("About")
st.sidebar.info("This app allows you to query the Fair Credit Reporting Act (FCRA) content and get answers using GPT-3 and Elasticsearch.")

st.markdown("""
### How to use:
1. Select the question type (Simple Question or True/False).
2. Enter your question related to the FCRA in the input box.
3. Click 'Get Answer' to retrieve the response.
""")