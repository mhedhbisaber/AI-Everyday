import json
import openai
import argparse
import tempfile
import time  # Import the time module
import os
import logging
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, PartitionKey
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores.azure_cosmos_db_no_sql import AzureCosmosDBNoSqlVectorSearch
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the OpenAI client

def generate_embeddings(text, openai_client, openai_embeddings_deployment):
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=openai_embeddings_deployment,
            #dimensions=openai_embeddings_dimensions
        )
        embeddings = response.model_dump()
        return embeddings['data'][0]['embedding']
    except Exception as e:
        logging.error("An error occurred while generating embeddings.", exc_info=True)
        raise

def read_json_and_generate_embeddings(json_file, openai_client, openai_embeddings_deployment):
    with open(json_file, 'r') as f:
        documents = json.load(f)
    
    texts = [doc['text'] for doc in documents]
    embeddings = [generate_embeddings(text, openai_client, openai_embeddings_deployment) for text in texts]
    
    for i, doc in enumerate(documents):
        doc['vector'] = embeddings[i]
    return documents

def save_json_with_embeddings(documents, output_file):
    with open(output_file, 'w') as f:
        json.dump(documents, f)

def load_data_to_cosmosdb(endpoint, key, database_name, container_name, json_file):
    client = CosmosClient(url=endpoint, credential=key)
    database = client.create_database_if_not_exists(id=database_name)
    container = database.create_container_if_not_exists(
        id=container_name,
        partition_key=PartitionKey(path="/id"),
        offer_throughput=400
    )
    
    with open(json_file, 'r') as f:
        documents = json.load(f)
    
    for doc in documents:
        print("writing item ", doc)
        container.upsert_item(doc)
    
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
    """
    Create a runnable chain that processes a CX query into an embedding vector before retrieving documents.
    """
    prompt_template = PromptTemplate.from_template(
        'Use the following context to answer the question: {context}\n\nQuestion: {question}'
    )
    
    def process_search_with_cx_query(inputs):
        logger.info("Processing CX query: %s", inputs["query"])
        # Perform similarity search
        search_results = retriever.similarity_search(inputs["query"])
        if not search_results:
            logger.warning("No results found for CX query: %s", inputs["query"])
            return {"context": "No relevant context found.", "question": inputs["query"]} 
        # Concatenate the text of search results
        context = "\n\n".join([doc.page_content for doc in search_results if hasattr(doc, 'page_content')])
        logger.info("Generated context: %s", context)
        return {"context": context, "question": inputs["query"]}
    
    return RunnableLambda(process_search_with_cx_query) | prompt_template | llm | StrOutputParser()

def generate_response(runnable_chain, query):
    response = runnable_chain.invoke({"query": query})
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings and load data into Cosmos DB.")
    parser.add_argument('--json_file', type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument('--openai_api_key', type=str, required=True, help="OpenAI API key.")
    parser.add_argument('--azure_endpoint', type=str, required=True, help="Azure OpenAI endpoint.")
    parser.add_argument('--cosmos_endpoint', type=str, required=True, help="Cosmos DB endpoint.")
    parser.add_argument('--cosmos_key', type=str, required=True, help="Cosmos DB key.")
    parser.add_argument('--database_name', type=str, required=True, help="Cosmos DB database name.")
    parser.add_argument('--container_name', type=str, required=True, help="Cosmos DB container name.")
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
    
    # Generate embeddings and save to a temporary file
    documents_with_embeddings = read_json_and_generate_embeddings(args.json_file, openai_client, args.openai_embeddings_deployment)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_file_path = temp_file.name
        save_json_with_embeddings(documents_with_embeddings, temp_file_path)
    
    try:
        # Load data into Cosmos DB
#        load_data_to_cosmosdb(args.cosmos_endpoint, args.cosmos_key, args.database_name, args.container_name, temp_file_path)
        
        # Configure LLM model
        llm = configure_llm(args.openai_api_key, args.azure_endpoint)
                # Define indexing policy and vector embedding policy
        indexing_policy = {
            "indexingMode": "consistent",
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [{"path": '/"_etag"/?'}],
            "vectorIndexes": [{"path": "/vector", "type": "quantizedFlat"}],
        }

        vector_embedding_policy = {
            "vectorEmbeddings": [
                {
                    "path": "/vector",
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": 3072,
                }
            ]
        }
        
        partition_key = PartitionKey(path="/id")
        cosmos_container_properties_test = {"partition_key": partition_key}
        cosmos_database_properties_test = {}
        
        # Convert documents to Document objects
        documents = [Document(page_content=doc['text'], metadata=doc) for doc in documents_with_embeddings]
        # Create LangChain instance with AzureCosmosDBNoSqlVectorSearch
        # Create LangChain instance with AzureCosmosDBNoSqlVectorSearch
        cosmos_client = CosmosClient(url=args.cosmos_endpoint, credential=args.cosmos_key)
        retriever = AzureCosmosDBNoSqlVectorSearch.from_documents(
            documents=documents,
            embedding=AzureOpenAIEmbeddings(
                azure_endpoint=args.azure_endpoint,
                api_key=args.openai_api_key,
                api_version="2024-08-01-preview",
                model=args.openai_embeddings_deployment,
                azure_deployment=args.openai_embeddings_deployment
            ),
            cosmos_client=cosmos_client,
            database_name=args.database_name,
            container_name=args.container_name,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties_test,
            cosmos_database_properties=cosmos_database_properties_test
        )

        runnable_chain = create_runnable_chain(llm, retriever)
        
        # Generate response
        response = generate_response(runnable_chain, args.query)
        print("Generated response:", response)

    finally:
        # Delete the temporary file
        os.remove(temp_file_path)
        print(f"Temporary file {temp_file_path} deleted successfully.")
