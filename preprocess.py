"""
Preprocess Module for News Headline Classification
===================================================
This module handles URL parsing and data preparation for the news classifier.

Key Functions:
- extract_label_from_url(): Extracts 'foxnews' or 'nbcnews' from URL domain
- extract_text_from_url(): Extracts readable text from URL path
- prepare_data(): Main function called by the evaluation backend

Author: Chih Yu Tsai, Aditya Pratap Singh, Xinjie Hu
Course: CIS 5190 Applied Machine Learning Fall 2025
"""

import re
import pandas as pd
from typing import Tuple, List
from urllib.parse import urlparse, unquote


def extract_label_from_url(url: str) -> str:
    """
    Extract the news source label from URL domain.
    
    Args:
        url: Full URL string (e.g., 'https://www.foxnews.com/world/...')
    
    Returns:
        str: 'foxnews' or 'nbcnews'
    
    Example:
        >>> extract_label_from_url('https://www.foxnews.com/world/article')
        'foxnews'
        >>> extract_label_from_url('https://www.nbcnews.com/politics/story')
        'nbcnews'
    """
    url_lower = url.lower()
    
    if 'foxnews.com' in url_lower:
        return 'foxnews'
    elif 'nbcnews.com' in url_lower:
        return 'nbcnews'
    else:
        # Default fallback - try to extract from domain
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if 'fox' in domain:
            return 'foxnews'
        elif 'nbc' in domain:
            return 'nbcnews'
        # If unable to determine, return empty string (will be filtered)
        return ''


def extract_text_from_url(url: str) -> str:
    """
    Extract readable headline text from URL path.
    
    This function parses the URL and converts the slug portion
    (typically the last segment) into readable text by:
    1. Extracting the path from URL
    2. Taking the last meaningful segment
    3. Replacing hyphens with spaces
    4. Removing special characters and numbers-only segments
    5. Cleaning up extra whitespace
    
    Args:
        url: Full URL string
    
    Returns:
        str: Cleaned headline text extracted from URL
    
    Example:
        >>> extract_text_from_url('https://www.foxnews.com/world/australian-senator-wears-burqa')
        'australian senator wears burqa'
    """
    try:
        # Parse the URL to get the path
        parsed = urlparse(url)
        path = unquote(parsed.path)  # Decode URL-encoded characters
        
        # Split path into segments and filter empty ones
        segments = [s for s in path.split('/') if s]
        
        if not segments:
            return ''
        
        # Usually the last segment contains the article title/slug
        # But sometimes there are trailing segments like 'rcna240477'
        # We want the longest meaningful segment
        
        # Start from the last segment and work backwards
        slug = ''
        for segment in reversed(segments):
            # Skip segments that are mostly numbers (like 'rcna240477')
            if 'rcna' in segment.lower():
                continue
            if re.match(r'^[a-z]*\d+$', segment.lower()):
                continue
            # Skip very short segments (likely category codes)
            if len(segment) < 5:
                continue
            slug = segment
            break
        
        # If no good segment found, use the last one
        if not slug and segments:
            slug = segments[-1]
            
        # Clean the slug
        text = slug.replace('-', ' ').replace('_', ' ')
        text = re.sub(r'\brcna\b', '', text, flags=re.IGNORECASE)  # remove independent rcna
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip().lower()

        return text
        
    except Exception as e:
        # If any parsing error occurs, return empty string
        return ''


def prepare_data(path: str) -> Tuple[List[str], List[str]]:
    """
    Main preprocessing function called by the evaluation backend.
    
    This function reads the CSV/Excel file containing URLs, extracts
    the headline text and labels, and returns them as lists.
    
    Args:
        path: Path to the data file (CSV or Excel)
    
    Returns:
        Tuple[List[str], List[str]]: 
            - X: List of extracted headline texts
            - y: List of labels ('foxnews' or 'nbcnews')
    
    Note:
        The backend expects X to be iterable where each element
        can be passed to model.predict() in batches.
    """
    # Read the data file (support both CSV and Excel formats)
    if path.endswith('.xlsx') or path.endswith('.xls'):
        df = pd.read_excel(path)
    else:
        # Try CSV with different encodings
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin-1')
    
    # Find the URL column (case-insensitive search)
    url_column = None
    for col in df.columns:
        if col.lower() in ['url', 'urls', 'link', 'links']:
            url_column = col
            break
    
    # If no URL column found, assume first column is URL
    if url_column is None:
        url_column = df.columns[0]
    
    # Extract texts and labels from URLs
    texts = []
    labels = []
    
    for url in df[url_column]:
        if pd.isna(url):
            continue
            
        url = str(url).strip()
        
        # Extract label from domain
        label = extract_label_from_url(url)
        if not label:
            continue  # Skip URLs that don't match foxnews or nbcnews
        
        # Extract text from URL path
        text = extract_text_from_url(url)
        if not text:
            continue  # Skip if no meaningful text extracted
        
        texts.append(text)
        labels.append(label)
    
    return texts, labels


# ============================================================================
# Testing / Debug Section (not used by backend)
# ============================================================================
if __name__ == '__main__':
    # Test the extraction functions
    test_urls = [
        'https://www.foxnews.com/world/australian-senator-wears-burqa-after-move-block-her-face-covering-ban-bill',
        'https://www.nbcnews.com/world/australia/studying-wrong-caesar-gets-high-school-seniors-history-exam-rcna240477',
        'https://www.foxnews.com/politics/trump-announces-new-policy',
        'https://www.nbcnews.com/business/markets/stock-market-rally-continues',
    ]
    
    print("Testing URL extraction:")
    print("-" * 60)
    
    for url in test_urls:
        label = extract_label_from_url(url)
        text = extract_text_from_url(url)
        print(f"URL: {url[:50]}...")
        print(f"  Label: {label}")
        print(f"  Text:  {text}")
        print()
