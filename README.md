# LLM-RAG System for FCRA Compliance

## Overview

This project develops a Retrieval-Augmented Generation (RAG) system to enhance the accessibility and understanding of the Fair Credit Reporting Act (FCRA) for users needing to comply with consumer reporting requirements. By leveraging advanced natural language processing techniques within a structured knowledge base of legal documents, the system provides precise, contextually relevant information to facilitate compliance.

## Problem Description

Understanding and implementing FCRA regulations can be challenging due to the complexity and detailed nature of the legal text involved. This project addresses the need for a more accessible and efficient method to navigate these documents, using a RAG system that improves the retrieval and contextualization of legal information, thus aiding compliance efforts.

## Project Components

- Text Files: Original FCRA documents used as the primary data source for the project, converted into textual format for processing.
- JSON Files: These files represent chunked and embedded versions of the text documents, optimized for efficient information retrieval within the RAG system.
- Notebook (01.preprocessing & rag.ipynb, 02.reused_embeded_data_rag.ipynb): Contains the preprocessing steps to transform text data into a searchable format, preparing it for use in the retrieval system.
- Source Code (app.py): The final application script that integrates the RAG system, enabling user interaction with the FCRA knowledge base through queries.

## Setup and Installation

1. Dependencies: Python dependencies are managed using Poetry. Install the necessary packages by running:

```bash
poetry install
```
	
2. Elasticsearch Setup: The project utilizes Elasticsearch for handling vector-based search operations. Set up the Elasticsearch service using Docker with the following command:
    
```bash
    docker run -d \
    --name elasticsearch \
    -m 4GB \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    -v esdata:/usr/share/elasticsearch/data \
    --restart unless-stopped \
    docker.elastic.co/elasticsearch/elasticsearch:8.5.1```
```
    This setup ensures that Elasticsearch operates in a single-node configuration with security features disabled for simplicity, and it restarts automatically unless manually stopped.

3. Running the Notebook: Process the data through the preprocessing notebook:

```bash
poetry run jupyter notebook 01.preprocessing.ipynb
```

4. Running the Application: The Streamlit-based application can be started with:

```bash
poetry run streamlit run app.py
```
    This command initializes the user interface, allowing for interactive query processing.

5. Deployment: The application is deployed directly on a GCP VM, ensuring robust performance and availability. Access the live application via: http://34.71.24.6:8501


## Functionality

The application provides a Streamlit interface where users can submit queries regarding FCRA regulations. It retrieves pertinent information from the knowledge base using Elasticsearch and generates responses through the RAG system, ensuring that the outputs are not only accurate but also contextually tailored to the user’s inquiries.

## Features

- Advanced Search: Incorporates vector-based searching for high precision and relevance in response generation.
- User Interface: Designed with Streamlit, the interface offers a clean, intuitive way for users to interact with the model, enhancing user experience and engagement.
- Deployment: Hosted on a GCP VM, the system is set up for high reliability and continuous access, facilitating ongoing user interaction.

## Conclusion

This LLM-RAG project represents a significant advancement in the application of NLP techniques to legal compliance. By simplifying the interpretation and application of FCRA regulations, the system serves as a crucial tool for users ranging from legal experts to financial institutions, ensuring they can more easily meet regulatory requirements.