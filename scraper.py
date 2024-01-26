import pandas as pd
import requests

from bs4 import BeautifulSoup
from multiprocessing import Pool
from tqdm import tqdm

def get_abstract_links(link: str):
  response = requests.get(link)
  soup = BeautifulSoup(response.text, "html.parser")
  links = soup.find_all("a")

  abstract_links = []
  for link in links:
    if "-Abstract-Conference.html" in link["href"]:  # Filter the abstracts
      abstract_links.append("https://papers.nips.cc" + link["href"])
  print(f"{len(abstract_links)} abstracts found")

  return abstract_links

def parse_paper_page(url):
  response = requests.get(url)
  soup = BeautifulSoup(response.text, "html.parser")

  info = {}
  info["title"] = soup.findAll("h4")[0].text
  info["authors"] = soup.findAll("i")[-1].text
  info["abstract"] = soup.findAll("p")[2].text
  info["url"] = url
  return info

def run_parallel():
  abstract_links = get_abstract_links("https://papers.nips.cc/paper/2022")

  results = []
  with Pool(16) as pool:  # Execute commands in parallel to speed things up
    for result in tqdm(pool.imap_unordered(parse_paper_page, abstract_links)):
      results.append(result)
  df = pd.DataFrame(results)
  df.to_csv('results.csv', index=False)

if __name__ == "__main__":
    run_parallel()