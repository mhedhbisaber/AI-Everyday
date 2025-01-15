import argparse
import httpx
from langchain_openai import AzureChatOpenAI

def configure_llm(openai_api_key):
    """Configure the GPT-4 model with the provided OpenAI API key."""
    return AzureChatOpenAI(
        azure_deployment="<deployment-name>",
        api_version="2024-08-01-preview",
        api_key=openai_api_key,
        azure_endpoint="https://XXXX.openai.azure.com/",
        model="gpt-4o"
    )

# Tool: Bing Search
def bing_search_tool(query, bing_api_key):
    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
    params = {"q": query, "count": 3, "mkt": "en-US", "freshness": "Week"}
    response = httpx.get(url, headers=headers, params=params)
    results = response.json()
    # Return relevant details (titles, URLs, snippets)
    if "webPages" in results:
        return "\n".join([f"Title: {item['name']}\nURL: {item['url']}\nSnippet: {item['snippet']}" for item in results["webPages"].get("value", [])])
    return "No results found."

def ask_llm_with_bing_results(llm, search_results, question):
    """Use LLM to answer the question using Bing search results."""
    # Create a prompt that includes the search results and the question
    prompt = f"Using the following search results, answer the question:\n\n{search_results}\n\nQuestion: {question}\n\nAnswer:"
    
    # Get the answer from GPT-4
    response = llm(prompt)
    # Extract the content from the response
    answer = response.content if hasattr(response, 'content') else response
    return answer
    
def verify_content_with_bing_and_llm(content, openai_api_key, bing_api_key):
    """
    Verify the factual accuracy of the content using Bing search and answer the question with GPT-4.
    """
    # Configure LLM
    llm = configure_llm(openai_api_key)

    # Step 1: Use Bing search for relevant information
    search_results = bing_search_tool(content, bing_api_key)

    # Step 2: Use LLM to answer the question using the search results
    answer = ask_llm_with_bing_results(llm, search_results, content)

    # Return the search results and the answer
    return {
        "search_results": search_results,
        "answer": answer
    }

def parse_arguments():
    """Parse command-line arguments for API keys and content."""
    parser = argparse.ArgumentParser(description="Verify content using Bing search and GPT-4.")
    parser.add_argument("--openai_api_key", required=True, help="Your OpenAI API key")
    parser.add_argument("--bing_api_key", required=True, help="Your Bing API key")
    parser.add_argument("--content", required=True, help="The content to verify")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Call the main function with provided inputs
    result = verify_content_with_bing_and_llm(args.content, args.openai_api_key, args.bing_api_key)
    
    # Print results
    if result["search_results"]:
        print("\nSearch Results:\n", result["search_results"])
    if result["answer"]:
        print("\nAnswer:\n", result["answer"])
