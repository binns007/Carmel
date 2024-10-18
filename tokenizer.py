import re
from urllib.parse import urlparse

# Improved Tokenizer
def makeTokens(url):
    # Parse the URL
    url_parsed = urlparse(url)
    
    # Tokenize by domain, path, and query
    domain_tokens = re.split(r'[.-]', url_parsed.netloc)  # Split by dot and dash in domain
    path_tokens = re.split(r'[/-]', url_parsed.path)  # Split by slash and dash in path
    query_tokens = re.split(r'[&=]', url_parsed.query)  # Split query params by & and =

    # Combine all tokens
    all_tokens = domain_tokens + path_tokens + query_tokens
    
    # Filter out empty strings
    all_tokens = [token for token in all_tokens if token]
    
    return all_tokens
