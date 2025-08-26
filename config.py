"""
Configuration management for the Enhanced Web Scraper
Production-ready configuration with environment variables support
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

class ScraperConfig:
    """Configuration class for web scraper"""
    
    # Target URLs
    TARGET_URL = "https://www.newegg.com/amd-ryzen-7-9000-series-ryzen-7-9800x3d-granite-ridge-zen-5-socket-am5-desktop-cpu-processor/p/N82E16819113877"
    
    # Scraping settings
    REQUEST_DELAY_MIN = float(os.getenv('REQUEST_DELAY_MIN', '1.0'))
    REQUEST_DELAY_MAX = float(os.getenv('REQUEST_DELAY_MAX', '3.0'))
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
    TIMEOUT = int(os.getenv('TIMEOUT', '30'))
    
    # Concurrency settings
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '5'))
    MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '10'))
    
    # Browser settings
    HEADLESS_BROWSER = os.getenv('HEADLESS_BROWSER', 'True').lower() == 'true'
    BROWSER_WINDOW_SIZE = os.getenv('BROWSER_WINDOW_SIZE', '1920,1080')
    
    # Database settings
    SQLITE_DB_PATH = os.getenv('SQLITE_DB_PATH', 'data/scraper_data.db')
    DUCKDB_PATH = os.getenv('DUCKDB_PATH', 'data/analysis.duckdb')
    
    # Export settings
    EXPORT_DIR = os.getenv('EXPORT_DIR', 'exports')
    EXPORT_FORMATS = ['json', 'csv', 'xlsx']
    
    # User agents pool
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0'
    ]

class AnalyzerConfig:
    """Configuration class for data analyzer"""
    
    # Dataset settings
    KAGGLE_DATASET = "asaniczka/amazon-uk-products-dataset-2023"
    DATASET_FILE = "data/amazon_uk_products.csv"
    SAMPLE_SIZE = int(os.getenv('SAMPLE_SIZE', '100000'))  # For testing, use smaller sample
    
    # Analysis settings
    MIN_SAMPLE_SIZE = int(os.getenv('MIN_SAMPLE_SIZE', '30'))
    CONFIDENCE_LEVEL = float(os.getenv('CONFIDENCE_LEVEL', '0.95'))
    Z_SCORE_THRESHOLD = float(os.getenv('Z_SCORE_THRESHOLD', '1.96'))
    
    # Export settings
    RESULTS_FILE = os.getenv('RESULTS_FILE', 'data/analysis_results.json')
    CHARTS_DIR = os.getenv('CHARTS_DIR', 'exports/charts')
    
    # Performance settings
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '10000'))
    MAX_MEMORY_GB = float(os.getenv('MAX_MEMORY_GB', '4.0'))

class LoggingConfig:
    """Logging configuration"""
    
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'data/scraper.log')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_MAX_BYTES = int(os.getenv('LOG_MAX_BYTES', '10485760'))  # 10MB
    LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '5'))

# Global configuration
CONFIG = {
    'scraper': ScraperConfig,
    'analyzer': AnalyzerConfig,
    'logging': LoggingConfig
}
