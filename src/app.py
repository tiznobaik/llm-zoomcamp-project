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
from langchain.schema import BaseRetriever
from typing import Any, List
from pydantic import BaseModel
from elasticsearch.helpers import bulk

st.title("FCRA Query System")
st.write("Welcome! You can ask questions related to the FCRA.")

# Initialize session state for generate_answer
if 'generate_answer' not in st.session_state:
    st.session_state['generate_answer'] = False

# Load saved embeddings and metadata
with open('src/document_embeddings.json', 'r') as f:
    saved_data = json.load(f)

# Convert the embeddings back to numpy arrays
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
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

embedding_model = OpenAIEmbeddings(openai_api_key=api_key)

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
question_type = st.selectbox("Select Question Type", ["Simple Question", "True/False", "Multiple Choice"])

# Initialize session state for options
if 'options' not in st.session_state:
    st.session_state.options = ['']

# Initialize session state for formatted_query
if 'formatted_query' not in st.session_state:
    st.session_state['formatted_query'] = None

# Input box for user query
if question_type == "Multiple Choice":
    query = st.text_input("Enter your multiple-choice question related to the FCRA:", "")

    # Function to add a new option
    def add_option():
        st.session_state.options.append('')

    st.write("Enter your options:")
    for i in range(len(st.session_state.options)):
        st.session_state.options[i] = st.text_input(f"Option {i + 1}:", st.session_state.options[i], key=f"option_{i}")

    # Buttons to add or remove options
    col1, col2 = st.columns(2)
    with col1:
        st.button("Add Option", on_click=add_option)
    with col2:
        if len(st.session_state.options) > 1:
            def remove_option():
                st.session_state.options.pop()
            st.button("Remove Last Option", on_click=remove_option)
else:
    query = st.text_input("Enter your question related to the FCRA:", "")

# Button to trigger the query
if st.button("Get Answer"):
    if query:
        # Check question type and adjust the prompt
        if question_type == "Simple Question":
            formatted_query = query
        elif question_type == "True/False":
            formatted_query = query  # No need to modify the query here
        elif question_type == "Multiple Choice":
            # Include the options in the query
            options_text = ""
            for idx, option_text in enumerate(st.session_state.options):
                if option_text.strip() != "":
                    options_text += f"{idx + 1}. {option_text}\n"
            formatted_query = f"{query}\nOptions:\n{options_text}"
        # Store formatted_query in session state
        st.session_state['formatted_query'] = formatted_query
        st.session_state['generate_answer'] = True  # Set flag to True
        # Process the query
        st.write(f"Processing query: {st.session_state['formatted_query']}")
    else:
        st.error("Please enter a query.")


# Function to add a new option
def add_option():
    st.session_state.options.append('')
    st.session_state['generate_answer'] = False  # Reset flag
    st.session_state['formatted_query'] = None  # Reset formatted_query

# Function to remove the last option
def remove_option():
    st.session_state.options.pop()
    st.session_state['generate_answer'] = False  # Reset flag
    st.session_state['formatted_query'] = None  # Reset formatted_query

# Initialize the LLM (GPT)
llm = OpenAI(openai_api_key=api_key, temperature=0, max_tokens=256)

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

    Question: {question}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
elif question_type == "True/False":
    prompt_template = """
    You are a helpful assistant answering only based on the following documents:

    {context}

    Statement: {question}

    Answer "True" or "False" and provide a brief explanation based only on the provided documents.
    If the answer cannot be determined based on these documents alone, respond with "I don't know based on the information provided."
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
elif question_type == "Multiple Choice":
    prompt_template = """
    You are a helpful assistant that answers questions strictly based on the following documents:

    {context}

    Question:
    {question}

    Instructions:
    - Choose the most appropriate option (e.g., 1, 2, 3, ...) based only on the information in the documents.
    - If the answer cannot be determined from the documents, respond with "I don't know based on the information provided."
    - Do not use any prior knowledge or external information.
    - Provide a brief explanation citing the relevant parts of the documents that support your answer.

    Your answer should be in the following format:

    Answer: [Option Number]
    Explanation: [Your brief explanation based on the documents.]

    Remember, if the answer cannot be determined from the documents, you must respond with "I don't know based on the information provided."
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Load the combine documents chain with your custom prompt
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

combine_docs_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

# Initialize the RetrievalQA chain manually
qa_chain = RetrievalQA(
    retriever=retriever,
    combine_documents_chain=combine_docs_chain,
    input_key="question",
    output_key="result"
    # verbose=True  # Commented out verbose mode to hide debugging information
)

# Check if formatted_query exists in session state and generate_answer is True
if st.session_state['formatted_query'] and st.session_state['generate_answer']:
    with st.spinner("Retrieving answer..."):
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(st.session_state['formatted_query'])

        # Prepare the inputs for the chain
        chain_inputs = {"question": st.session_state['formatted_query'], "context": "\n\n".join([doc.page_content for doc in docs])}

        # Run the chain
        try:
            result = qa_chain(chain_inputs)
            answer = result['result']
            st.session_state['generate_answer'] = False  # Reset the flag after generating the answer
        except Exception as e:
            st.error(f"Error during chain execution: {e}")
            st.stop()

        # Function to parse the LLM's answer
        def parse_answer(answer):
            if "I don't know based on the information provided" in answer:
                return "I don't know based on the information provided.", ""
            else:
                # Split the answer into Answer and Explanation
                answer_parts = answer.strip().split('Explanation:', 1)
                answer_line = answer_parts[0].replace('Answer:', '').strip()
                explanation = answer_parts[1].strip() if len(answer_parts) > 1 else ""

                # Extract option number from answer_line
                selected_option = answer_line.strip()
                # Ensure the selected option is a valid number
                if selected_option.isdigit():
                    return selected_option, explanation
                else:
                    return selected_option, explanation

        # Post-process the answer
        selected_option, explanation = parse_answer(answer)
        if selected_option == "I don't know based on the information provided.":
            st.success("Answer: I don't know based on the information provided.")
        else:
            if question_type == "Multiple Choice":
                option_numbers = [str(i + 1) for i in range(len(st.session_state.options))]
                if selected_option in option_numbers:
                    st.success(f"Answer: Option {selected_option}")
                    st.write(f"Explanation: {explanation}")
                else:
                    st.success("Answer: I don't know based on the information provided.")
            else:
                st.success(f"Answer: {selected_option}")
                if explanation:
                    st.write(f"Explanation: {explanation}")

st.sidebar.title("About")
st.sidebar.info("This app allows you to query the Fair Credit Reporting Act (FCRA) content and get answers using GPT-3 and Elasticsearch.")

st.markdown("""
### How to use:
1. Select the question type (Simple Question, True/False, or Multiple Choice).
2. Enter your question related to the FCRA in the input box.
3. For Multiple Choice questions, add options using the 'Add Option' button and enter the option texts.
4. Click 'Get Answer' to retrieve the response.
""")