# Step-by-Step Guide to Using Microsoft Bing Search API in Python

This guide provides a step-by-step explanation of how to perform web searches using the Microsoft Bing Search API in Python. By the end of this tutorial, you will be able to query Bing for web search results and verify content programmatically.

---

## Prerequisites

Before you begin, ensure you have the following:

- A valid Microsoft Bing Search API key. You can obtain one from the [Azure Portal](https://portal.azure.com/).
- Python 3.7 or later installed on your system.
- The `httpx` library installed. You can install it using:

  ```bash
  pip install httpx
  ```
---

## Script Overview

The provided Python script performs the following tasks:

1. Accepts user input for a Bing API key and a content query.
2. Performs a web search using the Bing Search API.
3. Processes and displays relevant search results.
4. 
---

## Code

Here is the Python script:

```python

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
    description="Using Microsoft Bing Search API in Python"
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
```

---

## How to Use

### 1. Clone the Repository

Clone this repository or create a new Python file and copy the script above.

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Run the Script

Run the script using the following command-line arguments:

- `--bing_api_key`: Your Microsoft Bing Search API key.
- `--content`: The content you want to verify.

Example:

```bash
python script.py --bing_api_key "your-bing-api-key" --content "Python is a programming language."
```

### 3. Review the Output

The script will display search results, including titles, URLs, and snippets from Bing, which can help verify your content's accuracy.

---

## Key Features

- Perform web searches using Bing Search API.
- Retrieve relevant results, including titles, URLs, and snippets.
- Verify factual accuracy of content programmatically.

---

## Notes

- Ensure your API key has the necessary permissions and quotas for Bing Search API usage.
- Modify the `params` dictionary in the `bing_search_tool` function to customize the search parameters.

Happy coding! 🚀
