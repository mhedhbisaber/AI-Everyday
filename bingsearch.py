import argparse
import httpx

# Tool: Bing Search
def bing_search_tool(query, bing_api_key):
    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
    params = {"q": query, "count": 3, "mkt": "en-US", "freshness": "Week"}  # Limit to one result
    response = httpx.get(url, headers=headers, params=params)
    results = response.json()
    print(results)
    # Return relevant details (titles, URLs, snippets)
    if "webPages" in results:
        return "\n".join([f"Title: {item['name']}\nURL: {item['url']}\nSnippet: {item['snippet']}" for item in results["webPages"].get("value", [])])
    return "No results found."

search_tool = Tool(
    name="Bing Search",
    func=bing_search_tool,
    description="Use this tool to search the web content"
)

def verify_content_with_bing(content, bing_api_key):
    """
    Verify the factual accuracy of the content using Bing search.
    """
    # Step 1: Use Bing search for relevant information
    search_query = f"Verify: {content}"
    search_results = bing_search_tool(search_query, bing_api_key)

    # Return the search results
    return {
        "search_results": search_results
    }

def parse_arguments():
    """Parse command-line arguments for API keys and content."""
    parser = argparse.ArgumentParser(description="Verify content using Bing search.")
    parser.add_argument("--bing_api_key", required=True, help="Your Bing API key")
    parser.add_argument("--content", required=True, help="The content to verify")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Call the main function with provided inputs
    result = verify_content_with_bing(args.content, args.bing_api_key)
    
    # Print results
    if result["search_results"]:
        print("\nSearch Results:\n", result["search_results"])
