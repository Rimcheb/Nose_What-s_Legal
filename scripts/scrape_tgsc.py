#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import string
import csv
import re
import time

base_url = "https://www.thegoodscentscompany.com/fragonly-{}.html"

results = []

print("Scraping Good Scents Co. Fragrance Ingredients...")

for letter in string.ascii_lowercase:
    print(f"Scraping letter: {letter}")
    url = base_url.format(letter)
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        continue
        
    soup = BeautifulSoup(res.text, 'html.parser')
    
    # Each item appears in a row with a link inside td
    for tr in soup.find_all('tr'):
        td = tr.find('td')
        if td:
            a_tag = td.find('a')
            if a_tag and a_tag.has_attr('onclick') and 'openMainWindow' in str(a_tag['onclick']):
                name = a_tag.text.strip()
                
                # Extract CAS if present
                cas_match = re.search(r'CAS:\s*([0-9\-]+)', td.text)
                cas_number = cas_match.group(1) if cas_match else ""
                
                # We only really care if we have a CAS or a very precise name, but we'll collect all.
                if cas_number:
                    results.append({'name': name, 'cas_number': cas_number})
                
    time.sleep(1) # Be polite
    
print(f"Total entries found with CAS numbers: {len(results)}")

with open('tgsc_unregulated_fragrances.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['name', 'cas_number'])
    writer.writeheader()
    writer.writerows(results)

print("Saved to tgsc_unregulated_fragrances.csv")
