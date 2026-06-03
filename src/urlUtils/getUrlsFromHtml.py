from bs4 import BeautifulSoup
import json
from pathlib import Path
import os

def generate_topic_mapping(html_file):
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')

        new_topic_mapping = {}
        current_lektion = None

        # Find all anchor tags in the provided HTML
        links = soup.find_all('a', href=True)

        for a in links:
            url = a['href']
            
            # Clean the URL (remove domain if present, and query params)
            # This ensures we get '/themen/xyz/' instead of 'https://site.com/themen/xyz/?v=1'
            path = url.split('?')[0].replace('https://gogymi.ch', '') 
            
            if '/lektionen/' in path:
                current_lektion = path
                new_topic_mapping[current_lektion] = []
            
            elif ('/themen/' in path or '/tests/' in path) and current_lektion:
                # Avoid duplicates if the same link appears twice in the same section
                if path not in new_topic_mapping[current_lektion]:
                    new_topic_mapping[current_lektion].append(path)

        return new_topic_mapping

    except FileNotFoundError:
        print(f"Error: {html_file} not found.")
        return None

# --- Execution ---

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH   = os.path.join(ROOT_DIR, "data/LangzetitMatheKurs.html")
mapping = generate_topic_mapping(RESULTS_PATH)

if mapping:
    print("NEW_TOPIC_MAPPING = {")
    for lektion, sub_links in mapping.items():
        print(f"    '{lektion}': [")
        for link in sub_links:
            print(f"        '{link}',")
        print("    ],")
    print("}")