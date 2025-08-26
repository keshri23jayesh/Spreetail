import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any

class DatabaseManager:
    def __init__(self, db_path: str = "data/scraper_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create products table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    brand TEXT,
                    price TEXT,
                    rating REAL,
                    review_count INTEGER,
                    description TEXT,
                    url TEXT,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create reviews table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id INTEGER,
                    reviewer_name TEXT,
                    rating INTEGER,
                    review_title TEXT,
                    review_body TEXT,
                    review_date TEXT,
                    verified_buyer BOOLEAN,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (product_id) REFERENCES products (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logging.info("Database initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing database: {e}")
    
    def save_product(self, product_data: Dict[str, Any]) -> int:
        """Save product data and return product ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO products (title, brand, price, rating, review_count, description, url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                product_data.get('title'),
                product_data.get('brand'),
                product_data.get('price'),
                product_data.get('rating'),
                product_data.get('review_count'),
                product_data.get('description'),
                product_data.get('url')
            ))
            
            product_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logging.info(f"Product saved with ID: {product_id}")
            return product_id
            
        except Exception as e:
            logging.error(f"Error saving product: {e}")
            return None
    
    def save_reviews(self, reviews_data: List[Dict[str, Any]], product_id: int):
        """Save multiple reviews for a product"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for review in reviews_data:
                cursor.execute('''
                    INSERT INTO reviews (product_id, reviewer_name, rating, review_title, 
                                       review_body, review_date, verified_buyer)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    product_id,
                    review.get('reviewer_name'),
                    review.get('rating'),
                    review.get('review_title'),
                    review.get('review_body'),
                    review.get('review_date'),
                    review.get('verified_buyer')
                ))
            
            conn.commit()
            conn.close()
            
            logging.info(f"Saved {len(reviews_data)} reviews for product {product_id}")
            
        except Exception as e:
            logging.error(f"Error saving reviews: {e}")
    
    def get_product_summary(self) -> Dict[str, Any]:
        """Get summary of scraped data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM products")
            product_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM reviews")
            review_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(rating) FROM reviews WHERE rating IS NOT NULL")
            avg_rating = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_products': product_count,
                'total_reviews': review_count,
                'average_rating': round(avg_rating, 2) if avg_rating else 0
            }
            
        except Exception as e:
            logging.error(f"Error getting summary: {e}")
            return {}
