#!/usr/bin/env python3
"""
thebatch_scraper.py

Scrapes articles from "The Batch" site, downloads images,
and saves article metadata to JSON.
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict

import requests
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from webdriver_manager.chrome import ChromeDriverManager


@dataclass
class Article:
    title: str
    summary: str
    date_str: str
    url: str
    img_url: Optional[str] = None
    text: Optional[str] = None
    img_path: Optional[str] = None


class TheBatchScraper:
    def __init__(
        self,
        headless: bool = True,
        max_workers: int = 4,
        output_dir: Path = Path("output"),
        images_dir: Path = Path("images"),
        timeout: int = 10,
    ):
        self.headless = headless
        self.max_workers = max_workers
        self.output_dir = output_dir
        self.images_dir = images_dir
        self.timeout = timeout
        self.session = requests.Session()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def _init_driver(self) -> webdriver.Chrome:
        options = Options()
        if self.headless:
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(
            service=service,
            options=options
        )
        driver.set_page_load_timeout(self.timeout)
        return driver

    @staticmethod
    def _get_page_url(page_num: int) -> str:
        return f"https://www.deeplearning.ai/the-batch/page/{page_num}/"

    @staticmethod
    def _scroll_to_bottom(driver: webdriver.Chrome, pause: float = 1.0) -> None:
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(pause)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                return
            last_height = new_height

    def collect_article_tiles(self, pages: int) -> List[Article]:
        driver = self._init_driver()
        articles: List[Article] = []

        for page in range(1, pages + 1):
            url = self._get_page_url(page)
            logging.info(f"Loading page {page}: {url}")
            driver.get(url)
            self._scroll_to_bottom(driver)

            tiles = driver.find_elements(By.CSS_SELECTOR, "article")[-15:]
            for tile in tiles:
                try:
                    date_str, title, summary = tile.text.split("\n")
                    link = next(
                        el.get_attribute("href")
                        for el in tile.find_elements(By.TAG_NAME, "a")
                        if "issue" in el.get_attribute("href")
                    )
                    img_el = tile.find_element(By.TAG_NAME, "img")
                    img_url = img_el.get_attribute("src") if img_el else None

                    articles.append(Article(
                        title=title.strip(),
                        summary=summary.strip(),
                        date_str=date_str.strip(),
                        url=link,
                        img_url=img_url
                    ))
                except Exception as e:
                    logging.warning(f"Failed to parse tile: {e}")

        driver.quit()
        logging.info(f"Collected {len(articles)} article tiles.")
        return articles

    def _fetch_article_text(
        self, article: Article, driver: webdriver.Chrome
    ) -> Optional[str]:
        for attempt in range(1, 4):
            try:
                driver.get(article.url)
                WebDriverWait(driver, self.timeout).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "div.prose--styled")
                    )
                )

                content = driver.find_element(
                    By.CSS_SELECTOR, "div.prose--styled"
                )
                paragraphs = []
                for el in content.find_elements(By.XPATH, ".//*"):
                    if el.tag_name == "hr":
                        break
                    if el.text:
                        paragraphs.append(el.text.strip())

                unique = list(dict.fromkeys(paragraphs))
                return "\n".join(unique)

            except Exception as e:
                logging.error(f"[Attempt {attempt}] Error fetching {article.url}: {e}")
                time.sleep(2)
        return None

    def _download_image(self, img_url: str, issue_name: str) -> Optional[str]:
        try:
            resp = self.session.get(img_url, timeout=self.timeout)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content))
            if img.mode != "RGB":
                img = img.convert("RGB")
            path = self.images_dir / f"{issue_name}.jpg"
            img.save(path, format="JPEG")
            return str(path)
        except Exception as e:
            logging.error(f"Image download failed for {img_url}: {e}")
            return None

    def enrich_articles(self, articles: List[Article]) -> List[Article]:
        pbar = tqdm(total=len(articles), desc="Fetching details", unit="art")

        def worker(chunk: List[Article]) -> Dict[str, Article]:
            driver = self._init_driver()
            out = {}
            for art in chunk:
                art.text = self._fetch_article_text(art, driver)
                if art.img_url:
                    issue = art.url.rstrip("/").split("/")[-1]
                    art.img_path = self._download_image(art.img_url, issue)
                out[art.url] = art
                pbar.update(1)
            driver.quit()
            return out

        n = min(self.max_workers, len(articles))
        chunks = [articles[i::n] for i in range(n)]

        enriched: Dict[str, Article] = {}
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(worker, chunk) for chunk in chunks]
            for f in as_completed(futures):
                enriched.update(f.result())

        pbar.close()
        return list(enriched.values())

    def save_to_json(self, articles: List[Article], filename: str) -> None:
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(a) for a in articles], f, indent=2, ensure_ascii=False)
        logging.info(f"Saved data to {path}")

    def run(self, pages: int) -> None:
        tiles_file = "thebatch_tiles.json"
        full_file = "thebatch_full.json"

        tiles_path = self.output_dir / tiles_file
        full_path = self.output_dir / full_file

        if not tiles_path.exists():
            tiles = self.collect_article_tiles(pages)
            self.save_to_json(tiles, tiles_file)
        else:
            with open(tiles_path, "r", encoding="utf-8") as f:
                tiles = [Article(**d) for d in json.load(f)]

        if not full_path.exists():
            full = self.enrich_articles(tiles)
            self.save_to_json(full, full_file)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Deeplearning.ai 'The Batch' articles."
    )
    parser.add_argument(
        "--pages", type=int, default=22,
        help="Number of paginated listing pages to scrape."
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of concurrent worker threads."
    )
    parser.add_argument(
        "--headed", dest="headless", action="store_false", default=True,
        help="Run browsers in headed (non-headless) mode."
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("output"),
        help="Directory where JSON files will be saved."
    )
    parser.add_argument(
        "--imagedir", type=Path, default=Path("output/images"),
        help="Directory where images will be saved."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s"
    )

    scraper = TheBatchScraper(
        headless=args.headless,
        max_workers=args.workers,
        output_dir=args.outdir,
        images_dir=args.imagedir
    )
    scraper.run(args.pages)


if __name__ == "__main__":
    main()
