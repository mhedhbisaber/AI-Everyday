# Azure Cognitive Search with OpenAI Integration

This project demonstrates how to integrate Azure Cognitive Search with OpenAI to create a powerful search and retrieval system. The code generates embeddings using OpenAI, creates an index in Azure Cognitive Search, and loads data into the search service. It also configures a GPT-4 model to generate responses based on search results.

## Table of Contents
- Prerequisites
- Installation
- Usage
- Functions
- License

## Prerequisites
- Python 3.7 or higher
- Azure subscription
- OpenAI API key
- Azure Cognitive Search service

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Prepare your JSON file with the documents to be indexed.
2. Run the script with the necessary arguments:
    ```sh
    python main.py --json_file path/to/your/json_file.json \
                   --openai_api_key your_openai_api_key \
                   --azure_endpoint your_azure_endpoint \
                   --search_endpoint your_search_endpoint \
                   --search_key your_search_key \
                   --index_name your_index_name \
                   --query your_query \
                   --openai_embeddings_deployment your_openai_embeddings_deployment \
                   --openai_embeddings_dimensions your_openai_embeddings_dimensions
    ```

## Functions
### `generate_embeddings(openai_client, text, EMBEDDING_MODEL_DEPLOYMENT_NAME)`
Generates embeddings from a string of text using OpenAI.

### `read_json_and_generate_embeddings(json_file, openai_client, openai_embeddings_deployment)`
Reads a JSON file and generates embeddings for each document.

### `save_json_with_embeddings(documents, output_file)`
Saves the documents with embeddings to a JSON file.

### `create_index(service_endpoint, key, index_name, dimensions=3072)`
Creates an index in Azure Cognitive Search.

### `load_data_to_search_service(endpoint, key, index_name, documents)`
Loads data into Azure Cognitive Search.

### `configure_llm(openai_api_key, azure_endpoint)`
Configures the GPT-4 model with the provided OpenAI API key.

### `create_runnable_chain(llm, retriever)`
Creates a runnable chain for processing search results and generating responses.

### `generate_response(runnable_chain, query)`
Generates a response using the runnable chain.

### `test_retriever(search_client, query, EMBEDDING_MODEL_DEPLOYMENT_NAME)`
Tests the retriever by searching for a query in Azure Cognitive Search.
