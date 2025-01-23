import json
import openai
import argparse
import tempfile
import os
import logging
import time
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration
)
from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, SimpleJsonOutputParser
from langchain.output_parsers import RetryOutputParser, YamlOutputParser
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts.chat import HumanMessage


# Configure logging
logging.basicConfig(level=logging.INFO)

# Create the OpenAI client
def generate_embeddings(openai_client, text, EMBEDDING_MODEL_DEPLOYMENT_NAME):
    '''
    Generate embeddings from string of text.
    This will be used to vectorize data and user input for interactions with Azure OpenAI.
    '''
    response = openai_client.embeddings.create(input=text, model=EMBEDDING_MODEL_DEPLOYMENT_NAME)
    embeddings = response.model_dump()
    time.sleep(0.5) 
    return embeddings['data'][0]['embedding']

def read_json_and_generate_embeddings(json_file, openai_client, openai_embeddings_deployment):
    with open(json_file, 'r') as f:
        documents = json.load(f)
    
    texts = [doc['content'] for doc in documents]
    embeddings = [generate_embeddings(openai_client, text, openai_embeddings_deployment) for text in texts]
    
    for i, doc in enumerate(documents):
        doc['content_vector'] = embeddings[i]
    
    return documents

def save_json_with_embeddings(documents, output_file):
    with open(output_file, 'w') as f:
        json.dump(documents, f)

def create_index(service_endpoint, key, index_name, dimensions=3072):
    credential = AzureKeyCredential(key)
    client = SearchIndexClient(service_endpoint, credential)
    
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
        SimpleField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dimensions,
            vector_search_profile_name="my-vector-config"
        )
    ]
    
    vector_search = VectorSearch(
        profiles=[VectorSearchProfile(name="my-vector-config", algorithm_configuration_name="my-algorithms-config")],
        algorithms=[HnswAlgorithmConfiguration(name="my-algorithms-config")]
    )
    
    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    client.create_index(index)
    print(f"Index '{index_name}' created successfully.")

def load_data_to_search_service(endpoint, key, index_name, documents):
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(key))
    
    # Split documents into smaller batches if necessary
    batch_size = 1000  # Adjust batch size as needed
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        try:
            results = search_client.upload_documents(batch)
            print(f"Batch {i // batch_size + 1} uploaded successfully.")
        except Exception as e:
            print(f"Error uploading batch {i // batch_size + 1}: {e}")

    print("Data loaded successfully.")

def configure_llm(openai_api_key, azure_endpoint):
    """Configure the GPT-4 model with the provided OpenAI API key."""
    return AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2024-08-01-preview",
        api_key=openai_api_key,
        azure_endpoint=azure_endpoint,
        model="gpt-4o"
    )


def create_runnable_chain(llm, retriever):
    prompt_template = PromptTemplate.from_template(
        'Use the following context to answer the question: {context}\n\nQuestion: {question}'
    )

def create_runnable_chain(llm, retriever):
    prompt_template = PromptTemplate.from_template(
        'Use the following context to answer the question: {context}\n\nQuestion: {question}'
    )

    def process_search(inputs):
        search_results = retriever.search(inputs["query"], search_fields=["content"])
        print("Search Results:", search_results)  # Debugging line
        search_results_list = list(search_results)
        if not search_results_list:
            return {"context": "No relevant context found.", "question": inputs["query"]}
        
        # Sort the search results by score in descending order
        search_results_list.sort(key=lambda x: x["@search.score"], reverse=True)
        
        # Select the top result
        top_result = search_results_list[0]
        
        return {"context": top_result["content"], "question": inputs["query"]}
    
    RunnableSequenceT = RunnableLambda(process_search) | prompt_template | llm | StrOutputParser()
    return RunnableSequenceT

def generate_response(runnable_chain, query):
    response = runnable_chain.invoke({"query": query})
    return response

def test_retriever(search_client, query, EMBEDDING_MODEL_DEPLOYMENT_NAME):
    results = search_client.search(query, search_fields=["content"])
    
    # Convert results to a list of dictionaries
    results_list = [result for result in results]
    
    print("Retrieved Results:", json.dumps(results_list, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings and load data into Azure Cognitive Search.")
    parser.add_argument('--json_file', type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument('--openai_api_key', type=str, required=True, help="OpenAI API key.")
    parser.add_argument('--azure_endpoint', type=str, required=True, help="Azure OpenAI endpoint.")
    parser.add_argument('--search_endpoint', type=str, required=True, help="Azure Cognitive Search endpoint.")
    parser.add_argument('--search_key', type=str, required=True, help="Azure Cognitive Search key.")
    parser.add_argument('--index_name', type=str, required=True, help="Azure Cognitive Search index name.")
    parser.add_argument('--query', type=str, required=True, help="Query to test the RAG approach.")
    parser.add_argument('--openai_embeddings_deployment', type=str, required=True, help="OpenAI embeddings deployment name.")
    parser.add_argument('--openai_embeddings_dimensions', type=int, required=True, help="OpenAI embeddings dimensions.")
    
    args = parser.parse_args()
    
    # Initialize OpenAI client
    openai_client = AzureOpenAI(
        azure_endpoint=args.azure_endpoint,
        api_key=args.openai_api_key,
        api_version="2024-08-01-preview"
    )
    
    # Create the index
    create_index(args.search_endpoint, args.search_key, args.index_name, args.openai_embeddings_dimensions)
    
    # Generate embeddings and save to a temporary file
    documents_with_embeddings = read_json_and_generate_embeddings(args.json_file, openai_client, args.openai_embeddings_deployment)
    
    #with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_file_path = temp_file.name
        save_json_with_embeddings(documents_with_embeddings, temp_file_path)
    
    try:
        # Load data into Azure Cognitive Search
        load_data_to_search_service(args.search_endpoint, args.search_key, args.index_name, documents_with_embeddings)
        
        # Configure LLM model
        llm = configure_llm(args.openai_api_key, args.azure_endpoint)
        
        # Initialize Search Client
        search_client = SearchClient(endpoint=args.search_endpoint, index_name=args.index_name, credential=AzureKeyCredential(args.search_key))
        
        # Test the retriever
        test_retriever(search_client, args.query, args.openai_embeddings_deployment)

        runnable_chain = create_runnable_chain(llm, search_client)
        
        # Generate response
        response = generate_response(runnable_chain, args.query)
        print("Generated response:", response)

    finally:
        print("End")
        # Delete the temporary file
        os.remove(temp_file_path)
        print(f"Temporary file {temp_file_path} deleted successfully.")
