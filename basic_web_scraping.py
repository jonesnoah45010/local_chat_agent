import requests
from bs4 import BeautifulSoup
import re

def basic_web_scrape(url):
    """
    Fetches the visible text from a given URL.
    
    Args:
        url (str): The URL of the webpage to extract text from.
    
    Returns:
        str: Extracted visible text from the webpage.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}  # Set a user-agent to avoid blocks
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an error if the request fails
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove script, style, and other non-visible elements
    for element in soup(['script', 'style', 'meta', 'noscript', 'iframe', 'head', 'title', 'link']):
        element.extract()
    
    # Extract visible text
    text = soup.get_text(separator=' ')
    
    # Clean up extra whitespace and newlines
    return '\n'.join(line.strip() for line in text.splitlines() if line.strip())

# Example usage:
# url = "https://example.com"
# print(get_visible_text(url))



def contains_url(text):
    """
    Checks if the given text contains at least one URL.
    
    Args:
        text (str): The text to check.
    
    Returns:
        bool: True if the text contains a URL, False otherwise.
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return bool(url_pattern.search(text))




def extract_urls(text):
    """
    Extracts all URLs from a given text.
    
    Args:
        text (str): The text to extract URLs from.
    
    Returns:
        list or None: A list of URLs if found, otherwise None.
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    urls = url_pattern.findall(text)
    return urls if urls else None




























