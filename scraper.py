"""
Advanced Web Scraper with Production Features
- Dynamic content handling with Selenium
- Advanced error handling and retry mechanisms
- Concurrent processing with rate limiting
- Comprehensive data extraction and validation
"""

import asyncio
import aiohttp
import logging
import time
import random
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from bs4 import BeautifulSoup
import requests
from fake_useragent import UserAgent
from tenacity import retry, stop_after_attempt, wait_exponential
from asyncio_throttle import Throttler
import psutil

from config import CONFIG

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class ProductData:
    """Structured product data model"""
    url: str
    title: str = ""
    brand: str = ""
    price: str = ""
    original_price: str = ""
    discount_percentage: str = ""
    rating: float = 0.0
    review_count: int = 0
    description: str = ""
    specifications: Dict[str, str] = None
    availability: str = ""
    seller: str = ""
    shipping_info: str = ""
    images: List[str] = None
    scraped_at: str = ""
    
    def __post_init__(self):
        if self.specifications is None:
            self.specifications = {}
        if self.images is None:
            self.images = []
        if not self.scraped_at:
            self.scraped_at = datetime.now().isoformat()

@dataclass
class ReviewData:
    """Structured review data model"""
    reviewer_name: str = ""
    reviewer_id: str = ""
    rating: int = 0
    review_title: str = ""
    review_body: str = ""
    review_date: str = ""
    verified_buyer: bool = False
    helpful_votes: int = 0
    total_votes: int = 0
    reviewer_location: str = ""
    reviewer_rank: str = ""
    product_variant: str = ""
    scraped_at: str = ""
    
    def __post_init__(self):
        if not self.scraped_at:
            self.scraped_at = datetime.now().isoformat()

class AdvancedWebScraper:
    """Production-ready web scraper with advanced features"""
    
    def __init__(self):
        self.config = CONFIG['scraper']
        self.ua = UserAgent()
        self.session = None
        self.driver = None
        self.throttler = Throttler(rate_limit=self.config.MAX_CONCURRENT_REQUESTS)
        self.stats = {
            'requests_made': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'start_time': time.time()
        }
        
    def setup_selenium_driver(self) -> webdriver.Chrome:
        """Setup Selenium WebDriver with optimized options"""
        try:
            chrome_options = Options()
            
            if self.config.HEADLESS_BROWSER:
                chrome_options.add_argument('--headless')
            
            chrome_options.add_argument(f'--window-size={self.config.BROWSER_WINDOW_SIZE}')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-logging')
            chrome_options.add_argument('--silent')
            chrome_options.add_argument('--log-level=3')
            chrome_options.add_argument(f'--user-agent={self.ua.random}')
            
            # Performance optimizations
            prefs = {
                "profile.managed_default_content_settings.images": 2,  # Block images
                "profile.default_content_setting_values.notifications": 2,  # Block notifications
            }
            chrome_options.add_experimental_option("prefs", prefs)
            
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.set_page_load_timeout(self.config.TIMEOUT)
            
            logger.info("Selenium WebDriver initialized successfully")
            return driver
            
        except Exception as e:
            logger.error(f"Failed to setup Selenium driver: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_page_async(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Async page fetching with retry logic"""
        try:
            async with self.throttler:
                headers = {'User-Agent': self.ua.random}
                async with session.get(url, headers=headers, timeout=self.config.TIMEOUT) as response:
                    if response.status == 200:
                        self.stats['requests_made'] += 1
                        return await response.text()
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            raise
    
    def wait_for_dynamic_content(self, driver: webdriver.Chrome, timeout: int = 10) -> bool:
        """Wait for dynamic content to load"""
        try:
            # Wait for reviews section to load
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".review-item, .customer-review, [data-testid='review']"))
            )
            
            # Additional wait for JavaScript to finish
            WebDriverWait(driver, timeout).until(
                lambda driver: driver.execute_script("return jQuery.active == 0") if driver.execute_script("return typeof jQuery != 'undefined'") else True
            )
            
            return True
        except TimeoutException:
            logger.warning("Dynamic content loading timeout - proceeding with available content")
            return False
    
    def extract_product_specifications(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract detailed product specifications"""
        specs = {}
        
        # Multiple selector strategies for specifications
        spec_selectors = [
            '.product-specs table tr',
            '.specifications-table tr',
            '.spec-table tr',
            '[data-testid="specifications"] tr'
        ]
        
        for selector in spec_selectors:
            spec_rows = soup.select(selector)
            if spec_rows:
                for row in spec_rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        key = cells[0].get_text(strip=True)
                        value = cells[1].get_text(strip=True)
                        if key and value:
                            specs[key] = value
                break
        
        return specs
    
    def extract_product_images(self, soup: BeautifulSoup) -> List[str]:
        """Extract product image URLs"""
        images = []
        
        # Multiple strategies for finding images
        image_selectors = [
            '.product-images img',
            '.gallery img',
            '.product-gallery img',
            '[data-testid="product-image"] img'
        ]
        
        for selector in image_selectors:
            img_elements = soup.select(selector)
            for img in img_elements:
                src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                if src and src.startswith(('http', '//')):
                    images.append(src)
        
        return list(set(images))  # Remove duplicates
    
    def extract_advanced_product_info(self, soup: BeautifulSoup, url: str) -> ProductData:
        """Extract comprehensive product information"""
        try:
            product = ProductData(url=url)
            
            # Title extraction with multiple fallbacks - Enhanced for Newegg
            title_selectors = [
                'h1.product-title',
                '.product-title h1',
                'h1[data-testid="product-title"]', 
                '.product-wrap h1',
                'h1.page-title',
                'h1',  # Generic fallback
                '.product-title',
                '[data-key="product-title"]',
                '.item-title',
                '.product-name'
            ]
            
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    product.title = title_elem.get_text(strip=True)
                    logger.info(f"Found title with selector '{selector}': {product.title[:50]}...")
                    break
            
            if not product.title:
                logger.warning("No title found - page may not have loaded properly")
            
            # Brand extraction - Enhanced for Newegg
            brand_selectors = [
                '.product-brand',
                '[data-testid="product-brand"]',
                '.brand-name',
                '.manufacturer',
                '.product-brand a',
                'a[href*="/Brand/"]',
                '.brand',
                '.item-brand'
            ]
            
            for selector in brand_selectors:
                brand_elem = soup.select_one(selector)
                if brand_elem:
                    product.brand = brand_elem.get_text(strip=True)
                    break
            
            # Price extraction - Enhanced for Newegg  
            price_selectors = [
                '.price-current strong',
                '.price-current',
                '[data-testid="price-current"]',
                '.current-price',
                '.price .price-current',
                '.product-price .price-current',
                '.price-group .price-current',
                '.price-current-label + .price-current',
                '.price strong',
                '.item-price'
            ]
            
            for selector in price_selectors:
                price_elem = soup.select_one(selector)
                if price_elem:
                    product.price = price_elem.get_text(strip=True)
                    break
            
            # Original price (if on sale)
            original_price_selectors = [
                '.price-was',
                '.original-price',
                '.list-price',
                '[data-testid="original-price"]'
            ]
            
            for selector in original_price_selectors:
                orig_price_elem = soup.select_one(selector)
                if orig_price_elem:
                    product.original_price = orig_price_elem.get_text(strip=True)
                    break
            
            # Rating extraction
            rating_selectors = [
                '.rating-stars',
                '[data-testid="rating-stars"]',
                '.egg-rating',
                '.product-rating'
            ]
            
            for selector in rating_selectors:
                rating_elem = soup.select_one(selector)
                if rating_elem:
                    rating_text = rating_elem.get('title', '') or rating_elem.get_text(strip=True)
                    try:
                        if 'out of 5' in rating_text.lower():
                            product.rating = float(rating_text.split()[0])
                        elif len([c for c in rating_text if c == '★']) > 0:
                            product.rating = float(len([c for c in rating_text if c == '★']))
                    except:
                        pass
                    break
            
            # Review count extraction
            review_count_selectors = [
                '.item-rating-num',
                '[data-testid="review-count"]',
                '.rating-count',
                '.review-summary'
            ]
            
            for selector in review_count_selectors:
                count_elem = soup.select_one(selector)
                if count_elem:
                    count_text = count_elem.get_text(strip=True)
                    try:
                        product.review_count = int(''.join(filter(str.isdigit, count_text)))
                    except:
                        pass
                    break
            
            # Description extraction
            desc_selectors = [
                '.product-bullets ul',
                '.product-overview',
                '.product-description',
                '[data-testid="product-description"]',
                '.product-details'
            ]
            
            for selector in desc_selectors:
                desc_elem = soup.select_one(selector)
                if desc_elem:
                    product.description = desc_elem.get_text(strip=True)[:2000]  # Limit length
                    break
            
            # Availability
            availability_selectors = [
                '.product-inventory',
                '.availability',
                '.stock-status',
                '[data-testid="availability"]'
            ]
            
            for selector in availability_selectors:
                avail_elem = soup.select_one(selector)
                if avail_elem:
                    product.availability = avail_elem.get_text(strip=True)
                    break
            
            # Extract specifications and images
            product.specifications = self.extract_product_specifications(soup)
            product.images = self.extract_product_images(soup)
            
            logger.info(f"Successfully extracted product: {product.title[:50]}...")
            self.stats['successful_extractions'] += 1
            
            return product
            
        except Exception as e:
            logger.error(f"Error extracting product info: {e}")
            self.stats['failed_extractions'] += 1
            return ProductData(url=url)
    
    def extract_advanced_review(self, review_elem, product_url: str) -> ReviewData:
        """Extract comprehensive review information"""
        try:
            review = ReviewData()
            
            # Reviewer name
            name_selectors = [
                '.review-item-author',
                '.reviewer-name',
                '.review-author',
                '[data-testid="reviewer-name"]'
            ]
            
            for selector in name_selectors:
                name_elem = review_elem.select_one(selector)
                if name_elem:
                    review.reviewer_name = name_elem.get_text(strip=True)
                    break
            
            # Rating
            rating_selectors = [
                '.review-rating',
                '.rating-stars',
                '.review-item-rating',
                '[data-testid="review-rating"]'
            ]
            
            for selector in rating_selectors:
                rating_elem = review_elem.select_one(selector)
                if rating_elem:
                    rating_text = rating_elem.get('title', '') or rating_elem.get_text(strip=True)
                    try:
                        if 'out of 5' in rating_text.lower():
                            review.rating = int(float(rating_text.split()[0]))
                        elif len([c for c in rating_text if c == '★']) > 0:
                            review.rating = len([c for c in rating_text if c == '★'])
                    except:
                        pass
                    break
            
            # Review title
            title_selectors = [
                '.review-item-title',
                '.review-title',
                '.review-summary',
                '[data-testid="review-title"]'
            ]
            
            for selector in title_selectors:
                title_elem = review_elem.select_one(selector)
                if title_elem:
                    review.review_title = title_elem.get_text(strip=True)
                    break
            
            # Review body
            body_selectors = [
                '.review-item-content',
                '.review-body',
                '.review-text',
                '[data-testid="review-body"]'
            ]
            
            for selector in body_selectors:
                body_elem = review_elem.select_one(selector)
                if body_elem:
                    review.review_body = body_elem.get_text(strip=True)
                    break
            
            # Review date
            date_selectors = [
                '.review-item-date',
                '.review-date',
                '.date',
                '[data-testid="review-date"]'
            ]
            
            for selector in date_selectors:
                date_elem = review_elem.select_one(selector)
                if date_elem:
                    review.review_date = date_elem.get_text(strip=True)
                    break
            
            # Verified buyer status
            verified_selectors = [
                '.verified-purchase',
                '.verified-buyer',
                '.verified',
                '[data-testid="verified-purchase"]'
            ]
            
            for selector in verified_selectors:
                verified_elem = review_elem.select_one(selector)
                if verified_elem:
                    review.verified_buyer = True
                    break
            
            # Helpful votes
            helpful_selectors = [
                '.helpful-votes',
                '.vote-helpful',
                '[data-testid="helpful-count"]'
            ]
            
            for selector in helpful_selectors:
                helpful_elem = review_elem.select_one(selector)
                if helpful_elem:
                    helpful_text = helpful_elem.get_text(strip=True)
                    try:
                        review.helpful_votes = int(''.join(filter(str.isdigit, helpful_text)))
                    except:
                        pass
                    break
            
            return review
            
        except Exception as e:
            logger.error(f"Error extracting review: {e}")
            return ReviewData()
    
    def extract_reviews_with_pagination(self, driver: webdriver.Chrome, max_pages: int = 5) -> List[ReviewData]:
        """Extract reviews with pagination support"""
        all_reviews = []
        current_page = 1
        
        try:
            while current_page <= max_pages:
                logger.info(f"Extracting reviews from page {current_page}")
                
                # Wait for reviews to load
                self.wait_for_dynamic_content(driver)
                
                # Get page source and parse
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                
                # Extract reviews from current page
                review_selectors = [
                    '.review-item',
                    '.customer-review', 
                    '[data-testid="review-item"]',
                    '.review-container',
                    '.review',
                    '.item-review',
                    '.user-review',
                    '.review-wrapper',
                    '.reviews .review',
                    '[class*="review"]'
                ]
                
                reviews_found = False
                for selector in review_selectors:
                    review_elements = soup.select(selector)
                    if review_elements:
                        reviews_found = True
                        logger.info(f"Found {len(review_elements)} reviews with selector '{selector}'")
                        
                        # Process reviews concurrently
                        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
                            future_to_review = {
                                executor.submit(self.extract_advanced_review, review_elem, driver.current_url): review_elem
                                for review_elem in review_elements
                            }
                            
                            for future in as_completed(future_to_review):
                                review_data = future.result()
                                if review_data and review_data.reviewer_name:
                                    all_reviews.append(review_data)
                        
                        break
                
                if not reviews_found:
                    logger.warning(f"No reviews found on page {current_page}")
                    break
                
                # Try to navigate to next page
                next_button_selectors = [
                    '.pagination-next',
                    '.next-page',
                    '[data-testid="next-page"]',
                    '.pagination .next'
                ]
                
                next_clicked = False
                for selector in next_button_selectors:
                    try:
                        next_button = driver.find_element(By.CSS_SELECTOR, selector)
                        if next_button.is_enabled():
                            driver.execute_script("arguments[0].click();", next_button)
                            time.sleep(random.uniform(2, 4))  # Wait for page load
                            next_clicked = True
                            break
                    except NoSuchElementException:
                        continue
                
                if not next_clicked:
                    logger.info("No more pages available")
                    break
                
                current_page += 1
            
            logger.info(f"Extracted {len(all_reviews)} total reviews across {current_page-1} pages")
            return all_reviews
            
        except Exception as e:
            logger.error(f"Error in pagination: {e}")
            return all_reviews
    
    def export_extraction_data(self, product: ProductData, reviews: List[ReviewData], 
                             filename_prefix: str = "extraction") -> Dict[str, str]:
        """Export extracted data to multiple formats"""
        try:
            os.makedirs(self.config.EXPORT_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exported_files = {}
            
            # Prepare data for export
            export_data = {
                'product': asdict(product),
                'reviews': [asdict(review) for review in reviews],
                'extraction_stats': {
                    **self.stats,
                    'total_reviews_extracted': len(reviews),
                    'extraction_duration': time.time() - self.stats['start_time']
                }
            }
            
            # Export to JSON
            json_file = f"{self.config.EXPORT_DIR}/{filename_prefix}_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            exported_files['json'] = json_file
            
            # Export to CSV (separate files for product and reviews)
            import pandas as pd
            
            # Product CSV
            product_csv = f"{self.config.EXPORT_DIR}/{filename_prefix}_product_{timestamp}.csv"
            pd.DataFrame([asdict(product)]).to_csv(product_csv, index=False)
            exported_files['product_csv'] = product_csv
            
            # Reviews CSV
            if reviews:
                reviews_csv = f"{self.config.EXPORT_DIR}/{filename_prefix}_reviews_{timestamp}.csv"
                pd.DataFrame([asdict(review) for review in reviews]).to_csv(reviews_csv, index=False)
                exported_files['reviews_csv'] = reviews_csv
            
            # Export to Excel
            excel_file = f"{self.config.EXPORT_DIR}/{filename_prefix}_{timestamp}.xlsx"
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                pd.DataFrame([asdict(product)]).to_excel(writer, sheet_name='Product', index=False)
                if reviews:
                    pd.DataFrame([asdict(review) for review in reviews]).to_excel(writer, sheet_name='Reviews', index=False)
                pd.DataFrame([self.stats]).to_excel(writer, sheet_name='Stats', index=False)
            exported_files['excel'] = excel_file
            
            logger.info(f"Exported extraction data to {len(exported_files)} files")
            return exported_files
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return {}
    
    def scrape_product_comprehensive(self, url: str) -> Dict[str, Any]:
        """Main method for comprehensive product scraping"""
        logger.info(f"Starting comprehensive scraping of: {url}")
        self.stats['start_time'] = time.time()
        
        try:
            # Setup Selenium driver
            self.driver = self.setup_selenium_driver()
            
            # Navigate to product page
            self.driver.get(url)
            
            # Wait for page to load completely
            time.sleep(random.uniform(3, 5))
            self.wait_for_dynamic_content(self.driver)
            
            # Get page source for BeautifulSoup parsing
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Extract comprehensive product data
            product_data = self.extract_advanced_product_info(soup, url)
            
            # Extract reviews with pagination
            reviews_data = self.extract_reviews_with_pagination(self.driver, max_pages=3)
            
            # Export extracted data
            exported_files = self.export_extraction_data(product_data, reviews_data)
            
            # Calculate final stats
            duration = time.time() - self.stats['start_time']
            
            result = {
                'product': product_data,
                'reviews': reviews_data,
                'stats': {
                    **self.stats,
                    'total_reviews': len(reviews_data),
                    'scraping_duration': duration,
                    'success_rate': self.stats['successful_extractions'] / max(self.stats['requests_made'], 1)
                },
                'exported_files': exported_files
            }
            
            logger.info(f"Scraping completed successfully in {duration:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive scraping: {e}")
            return {}
            
        finally:
            if self.driver:
                self.driver.quit()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return {
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_percent': psutil.cpu_percent(),
            'requests_per_second': self.stats['requests_made'] / max(time.time() - self.stats['start_time'], 1),
            'success_rate': self.stats['successful_extractions'] / max(self.stats['requests_made'], 1),
            'total_runtime': time.time() - self.stats['start_time']
        }
