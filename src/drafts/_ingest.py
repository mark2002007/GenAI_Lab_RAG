import json
import os
import time
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

pbar = None

def init_driver(headless: bool = False) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=options)
    return driver

def get_page_url(page_num: int) -> str:
    return f"https://www.deeplearning.ai/the-batch/page/{page_num}/"

def scroll_to_bottom(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)  # Wait for new content to load
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        
# Step 1
def parse_article_tile(article, load_images=False) -> dict:
    date_str, title, summary = article.text.split("\n")
    date = time.strptime(date_str, "%b %d, %Y")
    img = article.find_elements(By.CSS_SELECTOR, "img")
    assert len(img) == 1, "Expected at most one image in the article tile"
    img_url = img[0].get_attribute("src") if img else None
    
    a = [el for el in article.find_elements(By.TAG_NAME, "a") if "issue" in el.get_attribute("href")]
    assert len(a) == 1, "Expected exactly one link with 'issue' in href"
    url = a[0].get_attribute("href")
    return {
        "title": title,
        "summary": summary,
        "date_str": date_str,
        "date": date,
        "img_url": img_url,
        'url': url
    }
    
# Step 2
def remove_duplicates(lst):
    return list(dict.fromkeys(lst))

def remove_empty_strings(lst):
    return [item for item in lst if item.strip() != ""]

def break_strings(lst) -> list:
    return [line for t in lst for line in t.split('\n')]

def get_article_text(url, driver, max_retries=10, delay=2) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            driver.get(url)
            time.sleep(2)

            content_div = driver.find_element(
                By.CSS_SELECTOR, 
                "div.prose--styled.justify-self-center.post_postContent__wGZtc"
            )

            texts = []
            for tag in content_div.find_elements(By.XPATH, ".//*"):
                if tag.tag_name == 'a': continue
                if tag.tag_name == 'hr': break
                texts.append(tag.text)

            texts = remove_empty_strings(texts)
            texts = break_strings(texts)
            texts = remove_duplicates(texts)
            return "\n".join(texts)

        except Exception as e:
            print(f"[Attempt {attempt}/{max_retries}] Error fetching {url}")
            if attempt < max_retries:
                time.sleep(delay)
            else:
                return None


def load_image(url: str) -> np.ndarray:
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

def chunkify(lst, n_chunks):
    """Split lst into n_chunks, as evenly as possible."""
    k, r = divmod(len(lst), n_chunks)
    chunks = []
    start = 0
    for i in range(n_chunks):
        size = k + (1 if i < r else 0)
        chunks.append(lst[start:start+size])
        start += size
    return chunks

def proc_chunk_articles(article_dicts: list) -> dict:
    """Worker: spin up one driver, scrape all URLs in this chunk, then quit."""
    global pbar
    driver = init_driver(headless=True)
    results = {}
    for art in article_dicts:
        art['text'] = get_article_text(art['url'], driver)
        art['img']  = load_image(art['img_url']) if art['img_url'] else None

        if pbar:
            pbar.update(1)
            pbar.set_postfix_str(art['url'], refresh=False)

        results[art['url']] = art
    driver.quit()
    return results

def parallel_articles_proc(all_article_dicts, n_proc):
    global pbar

    pbar = tqdm(total=len(all_article_dicts), desc="Scraping articles", unit="art")

    results = {}
    chunks = chunkify(all_article_dicts, n_proc)
    with ThreadPoolExecutor(max_workers=n_proc) as exe:
        futures = [exe.submit(proc_chunk_articles, chunk) for chunk in chunks]
        for f in as_completed(futures):
            results.update(f.result())

    pbar.close()
    return results
    
def save_articles(articles, filename):
    os.makedirs('images', exist_ok=True)
    for article in articles:
        issue_name = article['url'].split('/')[-2]
        # Save the image as images/issue_name.jpg
        if 'img' in article and article['img'] is not None:
            img_path = f"images/{issue_name}.jpg"
            Image.fromarray(np.array(article['img'])).save(img_path)
            del article['img']
            article['img_path'] = img_path
    with open(filename, 'w') as f:
        json.dump(articles, f, indent=4)
    return articles
    
def load_articles(filename):
    articles = None
    with open(filename, 'r') as f:
        articles = json.load(f)
        
    for article in articles:
        if 'img_path' in article:
            article['img'] = Image.open(article['img_path'])
        else:
            article['img'] = None
    return articles
    
def main():
    articles_file = "thebatch_articles.json"
    articles_full_file = "thebatch_articles_full.json"
    
    if not os.path.exists(articles_file):
        driver = init_driver(headless=True)
        article_dicts = []
        for page_number in tqdm(range(22)):
            url = get_page_url(page_number+1)
            driver.get(url)
            scroll_to_bottom(driver)

            articles = driver.find_elements(By.CSS_SELECTOR, "article")[-15:]
            
            for article in articles:
                article_dict = parse_article_tile(article, load_images=True)
                article_dicts.append(article_dict)

        save_articles(article_dicts, articles_file)
        print(f"Saved {len(article_dicts)} articles to {articles_file}.") 
    
    if not os.path.exists(articles_full_file):
        article_dicts = load_articles(articles_file)
        
        article_dicts = parallel_articles_proc(article_dicts, n_proc=9)
        article_dicts = list(article_dicts.values())
        
        save_articles(article_dicts, articles_full_file)
        print(f"Saved full articles to {articles_full_file}.")

        
if __name__ == "__main__":
    main()