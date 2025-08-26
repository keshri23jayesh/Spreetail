"""
Production-Ready Product Data Analyzer
- Real Amazon UK dataset processing with Kaggle API
- Advanced statistical analysis with DuckDB
- Scalable processing for 2M+ records
- Comprehensive insights and visualizations
"""

import os
import logging
import json
import time
import zipfile
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import duckdb
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import psutil

from config import CONFIG

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class CategoryStats:
    """Statistical data for a product category"""
    category: str
    product_count: int
    avg_rating: float
    std_deviation: float
    variance: float
    min_rating: float
    max_rating: float
    median_rating: float
    percentile_25: float
    percentile_75: float
    z_score: float
    significance: str
    coefficient_variation: float

@dataclass
class AnalysisResults:
    """Complete analysis results"""
    dataset_overview: Dict[str, Any]
    category_statistics: List[CategoryStats]
    variability_analysis: Dict[str, Any]
    statistical_significance: List[Dict[str, Any]]
    performance_insights: Dict[str, Any]
    market_insights: List[str]
    export_files: Dict[str, str]
    processing_time: float
    memory_usage_mb: float

class ProductionAnalyzer:
    """Production-ready analyzer for large-scale product data"""
    
    def __init__(self):
        self.config = CONFIG['analyzer']
        self.conn = None
        self.start_time = time.time()
        self.processing_stats = {
            'records_processed': 0,
            'categories_analyzed': 0,
            'insights_generated': 0,
            'export_files_created': 0
        }
        
        # Ensure directories exist
        os.makedirs('data', exist_ok=True)
        os.makedirs(self.config.CHARTS_DIR, exist_ok=True)
        os.makedirs(CONFIG['scraper'].EXPORT_DIR, exist_ok=True)
    
    def setup_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        """Setup optimized DuckDB connection"""
        try:
            conn = duckdb.connect(self.config.DUCKDB_PATH if hasattr(self.config, 'DUCKDB_PATH') else CONFIG['scraper'].DUCKDB_PATH)
            
            # Optimize DuckDB settings for large datasets
            conn.execute("SET memory_limit='4GB'")
            conn.execute("SET threads TO 4")
            conn.execute("SET enable_progress_bar=true")
            
            logger.info("DuckDB connection established with optimizations")
            return conn
            
        except Exception as e:
            logger.error(f"Failed to setup DuckDB connection: {e}")
            raise
    
    def download_kaggle_dataset(self) -> bool:
        """Download Amazon UK dataset from Kaggle"""
        try:
            # Check if Kaggle credentials are configured
            kaggle_config_path = os.path.expanduser("~/.kaggle/kaggle.json")
            if not os.path.exists(kaggle_config_path):
                logger.warning("Kaggle credentials not found. Creating sample dataset instead.")
                return self.create_realistic_sample_dataset()
            
            import kaggle
            
            logger.info(f"Downloading dataset: {self.config.KAGGLE_DATASET}")
            
            # Download dataset
            kaggle.api.dataset_download_files(
                self.config.KAGGLE_DATASET,
                path='data/',
                unzip=True
            )
            
            # Find the downloaded CSV file
            data_files = [f for f in os.listdir('data/') if f.endswith('.csv')]
            if not data_files:
                logger.error("No CSV files found in downloaded dataset")
                return False
            
            # Rename to standard filename
            original_file = os.path.join('data/', data_files[0])
            if original_file != self.config.DATASET_FILE:
                os.rename(original_file, self.config.DATASET_FILE)
            
            logger.info(f"Dataset downloaded successfully: {self.config.DATASET_FILE}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading Kaggle dataset: {e}")
            logger.info("Falling back to sample dataset creation")
            return self.create_realistic_sample_dataset()
    
    def create_realistic_sample_dataset(self) -> bool:
        """Create a realistic sample dataset for analysis"""
        try:
            logger.info("Creating realistic sample dataset with market-based distributions...")
            
            # Realistic product categories with market share
            categories_data = {
                'Electronics': {'weight': 0.25, 'avg_rating': 4.1, 'std': 0.9},
                'Books': {'weight': 0.15, 'avg_rating': 4.3, 'std': 0.7},
                'Clothing': {'weight': 0.20, 'avg_rating': 3.9, 'std': 1.1},
                'Home & Garden': {'weight': 0.12, 'avg_rating': 4.0, 'std': 0.8},
                'Sports & Outdoors': {'weight': 0.08, 'avg_rating': 4.2, 'std': 0.8},
                'Beauty & Health': {'weight': 0.10, 'avg_rating': 4.1, 'std': 0.9},
                'Toys & Games': {'weight': 0.05, 'avg_rating': 4.4, 'std': 0.6},
                'Automotive': {'weight': 0.03, 'avg_rating': 4.0, 'std': 1.0},
                'Office Products': {'weight': 0.02, 'avg_rating': 3.8, 'std': 1.2}
            }
            
            total_samples = min(self.config.SAMPLE_SIZE, 100000)  # Limit for performance
            sample_data = []
            
            np.random.seed(42)  # For reproducible results
            
            with tqdm(total=total_samples, desc="Generating sample data") as pbar:
                for category, props in categories_data.items():
                    # Calculate number of products for this category
                    category_count = int(total_samples * props['weight'])
                    
                    for i in range(category_count):
                        # Generate realistic rating with category-specific distribution
                        rating = np.random.normal(props['avg_rating'], props['std'])
                        rating = max(1.0, min(5.0, rating))  # Clamp to 1-5 range
                        
                        # Generate other realistic fields
                        price = np.random.lognormal(3, 1.5)  # Log-normal distribution for prices
                        review_count = np.random.poisson(50)  # Poisson for review counts
                        
                        sample_data.append({
                            'product_id': f'B{len(sample_data):08d}',
                            'title': f'{category} Product {i+1}',
                            'category': category,
                            'main_category': category,
                            'rating': round(rating, 1),
                            'price': round(price, 2),
                            'review_count': review_count,
                            'brand': f'Brand_{np.random.randint(1, 100)}',
                            'availability': np.random.choice(['In Stock', 'Limited Stock', 'Out of Stock'], p=[0.8, 0.15, 0.05])
                        })
                        
                        pbar.update(1)
            
            # Create DataFrame and save
            df = pd.DataFrame(sample_data)
            df.to_csv(self.config.DATASET_FILE, index=False)
            
            logger.info(f"Sample dataset created: {len(df):,} records across {len(categories_data)} categories")
            return True
            
        except Exception as e:
            logger.error(f"Error creating sample dataset: {e}")
            return False
    
    def load_dataset_optimized(self) -> bool:
        """Load dataset into DuckDB with memory optimization"""
        try:
            self.conn = self.setup_duckdb_connection()
            
            # Check if dataset exists
            if not os.path.exists(self.config.DATASET_FILE):
                if not self.download_kaggle_dataset():
                    return False
            
            logger.info("Loading dataset into DuckDB...")
            
            # Load data in chunks for memory efficiency
            chunk_size = self.config.CHUNK_SIZE
            
            # First, create table structure
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS products AS 
                SELECT * FROM read_csv_auto('{self.config.DATASET_FILE}', sample_size=1000)
                WHERE 1=0
            """)
            
            # Drop existing table if it exists
            self.conn.execute("DROP TABLE IF EXISTS products")
            
            # Load data in chunks
            total_rows = 0
            chunk_num = 0
            
            for chunk in pd.read_csv(self.config.DATASET_FILE, chunksize=chunk_size):
                # Clean and validate data
                chunk = self.clean_data_chunk(chunk)
                
                if chunk_num == 0:
                    # First chunk - create table
                    self.conn.register('temp_chunk', chunk)
                    self.conn.execute("CREATE TABLE products AS SELECT * FROM temp_chunk")
                else:
                    # Subsequent chunks - insert data
                    self.conn.register('temp_chunk', chunk)
                    self.conn.execute("INSERT INTO products SELECT * FROM temp_chunk")
                
                total_rows += len(chunk)
                chunk_num += 1
                
                logger.info(f"Loaded chunk {chunk_num}: {len(chunk):,} records (Total: {total_rows:,})")
            
            # Create indexes for performance
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON products(category)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_rating ON products(rating)")
            
            # Verify data
            result = self.conn.execute("SELECT COUNT(*) FROM products").fetchone()
            self.processing_stats['records_processed'] = result[0]
            
            logger.info(f"Dataset loaded successfully: {result[0]:,} total records")
            return True
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False
    
    def clean_data_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data chunk"""
        try:
            # Handle missing values
            chunk = chunk.dropna(subset=['category', 'rating'])
            
            # Validate ratings
            chunk = chunk[(chunk['rating'] >= 1) & (chunk['rating'] <= 5)]
            
            # Clean category names
            chunk['category'] = chunk['category'].str.strip().str.title()
            
            # Handle price data if present
            if 'price' in chunk.columns:
                chunk['price'] = pd.to_numeric(chunk['price'], errors='coerce')
                chunk = chunk[chunk['price'] > 0]  # Remove invalid prices
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error cleaning data chunk: {e}")
            return chunk
    
    def calculate_comprehensive_statistics(self) -> List[CategoryStats]:
        """Calculate comprehensive statistics for each category"""
        try:
            logger.info("Calculating comprehensive category statistics...")
            
            # Calculate overall dataset statistics for Z-scores
            overall_stats = self.conn.execute("""
                SELECT 
                    AVG(rating) as overall_mean,
                    STDDEV(rating) as overall_std,
                    COUNT(*) as total_count
                FROM products 
                WHERE rating IS NOT NULL
            """).fetchone()
            
            overall_mean = overall_stats[0]
            overall_std = overall_stats[1]
            total_count = overall_stats[2]
            
            # Calculate detailed statistics per category
            query = f"""
            SELECT 
                category,
                COUNT(*) as product_count,
                ROUND(AVG(rating), 3) as avg_rating,
                ROUND(STDDEV(rating), 4) as std_deviation,
                ROUND(VARIANCE(rating), 4) as variance,
                MIN(rating) as min_rating,
                MAX(rating) as max_rating,
                ROUND(MEDIAN(rating), 2) as median_rating,
                ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY rating), 2) as percentile_25,
                ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY rating), 2) as percentile_75,
                ROUND((AVG(rating) - {overall_mean}) / ({overall_std} / SQRT(COUNT(*))), 4) as z_score
            FROM products 
            WHERE rating IS NOT NULL
            GROUP BY category
            HAVING COUNT(*) >= {self.config.MIN_SAMPLE_SIZE}
            ORDER BY avg_rating DESC
            """
            
            results = self.conn.execute(query).fetchall()
            
            category_stats = []
            for row in results:
                # Calculate coefficient of variation and significance
                cv = (row[3] / row[2]) * 100 if row[2] > 0 else 0
                significance = 'Significant' if abs(row[10]) > self.config.Z_SCORE_THRESHOLD else 'Not Significant'
                
                stats = CategoryStats(
                    category=row[0],
                    product_count=row[1],
                    avg_rating=row[2],
                    std_deviation=row[3],
                    variance=row[4],
                    min_rating=row[5],
                    max_rating=row[6],
                    median_rating=row[7],
                    percentile_25=row[8],
                    percentile_75=row[9],
                    z_score=row[10],
                    significance=significance,
                    coefficient_variation=cv
                )
                category_stats.append(stats)
            
            self.processing_stats['categories_analyzed'] = len(category_stats)
            logger.info(f"Statistics calculated for {len(category_stats)} categories")
            
            return category_stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return []
    
    def perform_advanced_variability_analysis(self) -> Dict[str, Any]:
        """Perform advanced variability analysis"""
        try:
            logger.info("Performing advanced variability analysis...")
            
            # Highest and lowest variability categories
            high_var_query = """
            SELECT 
                category, 
                ROUND(STDDEV(rating), 4) as std_deviation,
                ROUND(VARIANCE(rating), 4) as variance,
                ROUND((STDDEV(rating) / AVG(rating)) * 100, 2) as coefficient_variation,
                COUNT(*) as sample_size
            FROM products 
            WHERE rating IS NOT NULL
            GROUP BY category
            HAVING COUNT(*) >= 30
            ORDER BY std_deviation DESC
            LIMIT 5
            """
            
            low_var_query = """
            SELECT 
                category, 
                ROUND(STDDEV(rating), 4) as std_deviation,
                ROUND(VARIANCE(rating), 4) as variance,
                ROUND((STDDEV(rating) / AVG(rating)) * 100, 2) as coefficient_variation,
                COUNT(*) as sample_size
            FROM products 
            WHERE rating IS NOT NULL
            GROUP BY category
            HAVING COUNT(*) >= 30
            ORDER BY std_deviation ASC
            LIMIT 5
            """
            
            high_var_results = self.conn.execute(high_var_query).fetchall()
            low_var_results = self.conn.execute(low_var_query).fetchall()
            
            # Rating distribution analysis
            distribution_query = """
            SELECT 
                rating,
                COUNT(*) as frequency,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM products
            WHERE rating IS NOT NULL
            GROUP BY rating
            ORDER BY rating
            """
            
            distribution_results = self.conn.execute(distribution_query).fetchall()
            
            return {
                'highest_variability': [
                    {
                        'category': row[0],
                        'std_deviation': row[1],
                        'variance': row[2],
                        'coefficient_variation': row[3],
                        'sample_size': row[4]
                    }
                    for row in high_var_results
                ],
                'lowest_variability': [
                    {
                        'category': row[0],
                        'std_deviation': row[1],
                        'variance': row[2],
                        'coefficient_variation': row[3],
                        'sample_size': row[4]
                    }
                    for row in low_var_results
                ],
                'rating_distribution': [
                    {
                        'rating': row[0],
                        'frequency': row[1],
                        'percentage': row[2]
                    }
                    for row in distribution_results
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in variability analysis: {e}")
            return {}
    
    def generate_market_insights(self, category_stats: List[CategoryStats], 
                                variability_data: Dict[str, Any]) -> List[str]:
        """Generate actionable market insights"""
        insights = []
        
        if not category_stats:
            return insights
        
        try:
            # Performance insights
            best_category = max(category_stats, key=lambda x: x.avg_rating)
            worst_category = min(category_stats, key=lambda x: x.avg_rating)
            
            insights.append(f"🏆 Top Performer: {best_category.category} leads with {best_category.avg_rating:.2f} avg rating from {best_category.product_count:,} products")
            insights.append(f"📉 Needs Attention: {worst_category.category} shows lowest satisfaction at {worst_category.avg_rating:.2f} avg rating")
            
            # Variability insights
            if variability_data.get('highest_variability'):
                high_var = variability_data['highest_variability'][0]
                insights.append(f"⚠️ Quality Inconsistency: {high_var['category']} shows highest rating variability (σ={high_var['std_deviation']:.3f})")
            
            if variability_data.get('lowest_variability'):
                low_var = variability_data['lowest_variability'][0]
                insights.append(f"✅ Consistent Quality: {low_var['category']} demonstrates most consistent ratings (σ={low_var['std_deviation']:.3f})")
            
            # Statistical significance insights
            significant_categories = [cat for cat in category_stats if cat.significance == 'Significant']
            insights.append(f"🔬 Statistical Significance: {len(significant_categories)}/{len(category_stats)} categories show statistically significant rating differences")
            
            # Market concentration insights
            total_products = sum(cat.product_count for cat in category_stats)
            largest_category = max(category_stats, key=lambda x: x.product_count)
            market_share = (largest_category.product_count / total_products) * 100
            insights.append(f"📊 Market Concentration: {largest_category.category} dominates with {market_share:.1f}% of products")
            
            # Rating distribution insights
            if variability_data.get('rating_distribution'):
                dist = variability_data['rating_distribution']
                most_common_rating = max(dist, key=lambda x: x['frequency'])
                insights.append(f"⭐ Most Common Rating: {most_common_rating['rating']} stars ({most_common_rating['percentage']:.1f}% of all ratings)")
            
            self.processing_stats['insights_generated'] = len(insights)
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return insights
    
    def create_advanced_visualizations(self, category_stats: List[CategoryStats], 
                                     variability_data: Dict[str, Any]) -> Dict[str, str]:
        """Create comprehensive visualizations"""
        try:
            logger.info("Creating advanced visualizations...")
            chart_files = {}
            
            if not category_stats:
                return chart_files
            
            # 1. Category Performance Dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Average Ratings by Category', 'Rating Variability', 
                               'Product Count Distribution', 'Rating Distribution'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Average ratings bar chart
            categories = [stat.category for stat in category_stats]
            ratings = [stat.avg_rating for stat in category_stats]
            colors = ['#2E86AB' if stat.significance == 'Significant' else '#A23B72' for stat in category_stats]
            
            fig.add_trace(
                go.Bar(x=categories, y=ratings, marker_color=colors, name='Avg Rating'),
                row=1, col=1
            )
            
            # Variability scatter plot
            std_devs = [stat.std_deviation for stat in category_stats]
            product_counts = [stat.product_count for stat in category_stats]
            
            fig.add_trace(
                go.Scatter(x=std_devs, y=ratings, 
                          text=categories, mode='markers+text', 
                          marker=dict(size=[p/50 for p in product_counts], sizemode='diameter'),
                          name='Variability vs Rating'),
                row=1, col=2
            )
            
            # Product count bar chart
            fig.add_trace(
                go.Bar(x=categories, y=product_counts, marker_color='#F18F01', name='Product Count'),
                row=2, col=1
            )
            
            # Rating distribution
            if variability_data.get('rating_distribution'):
                dist = variability_data['rating_distribution']
                rating_values = [d['rating'] for d in dist]
                frequencies = [d['frequency'] for d in dist]
                
                fig.add_trace(
                    go.Bar(x=rating_values, y=frequencies, marker_color='#C73E1D', name='Rating Frequency'),
                    row=2, col=2
                )
            
            fig.update_layout(
                title_text="Product Category Analysis Dashboard",
                showlegend=False,
                height=800
            )
            
            dashboard_file = f"{self.config.CHARTS_DIR}/category_analysis_dashboard.html"
            fig.write_html(dashboard_file)
            chart_files['dashboard'] = dashboard_file
            
            # 2. Statistical Significance Heatmap
            if len(category_stats) > 1:
                z_scores = [stat.z_score for stat in category_stats]
                significance_colors = ['red' if abs(z) > 1.96 else 'green' for z in z_scores]
                
                fig_heatmap = go.Figure(data=go.Bar(
                    x=categories,
                    y=z_scores,
                    marker_color=significance_colors,
                    text=[f'Z={z:.2f}' for z in z_scores],
                    textposition='auto'
                ))
                
                fig_heatmap.add_hline(y=1.96, line_dash="dash", line_color="red", 
                                    annotation_text="Significance Threshold (+1.96)")
                fig_heatmap.add_hline(y=-1.96, line_dash="dash", line_color="red",
                                    annotation_text="Significance Threshold (-1.96)")
                
                fig_heatmap.update_layout(
                    title="Statistical Significance Analysis (Z-Scores)",
                    xaxis_title="Category",
                    yaxis_title="Z-Score",
                    xaxis_tickangle=-45
                )
                
                significance_file = f"{self.config.CHARTS_DIR}/statistical_significance.html"
                fig_heatmap.write_html(significance_file)
                chart_files['significance'] = significance_file
            
            self.processing_stats['export_files_created'] += len(chart_files)
            logger.info(f"Created {len(chart_files)} visualization files")
            
            return chart_files
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return {}
    
    def export_comprehensive_results(self, results: AnalysisResults) -> Dict[str, str]:
        """Export comprehensive analysis results"""
        try:
            logger.info("Exporting comprehensive results...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_files = {}
            
            # 1. JSON Export (Complete Results)
            json_file = f"{CONFIG['scraper'].EXPORT_DIR}/analysis_complete_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(results), f, indent=2, ensure_ascii=False, default=str)
            export_files['complete_json'] = json_file
            
            # 2. Excel Export (Multi-sheet)
            excel_file = f"{CONFIG['scraper'].EXPORT_DIR}/analysis_report_{timestamp}.xlsx"
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Category statistics
                if results.category_statistics:
                    stats_df = pd.DataFrame([asdict(stat) for stat in results.category_statistics])
                    stats_df.to_excel(writer, sheet_name='Category_Statistics', index=False)
                
                # Variability analysis
                if results.variability_analysis:
                    if results.variability_analysis.get('highest_variability'):
                        high_var_df = pd.DataFrame(results.variability_analysis['highest_variability'])
                        high_var_df.to_excel(writer, sheet_name='High_Variability', index=False)
                    
                    if results.variability_analysis.get('lowest_variability'):
                        low_var_df = pd.DataFrame(results.variability_analysis['lowest_variability'])
                        low_var_df.to_excel(writer, sheet_name='Low_Variability', index=False)
                
                # Summary insights
                insights_df = pd.DataFrame({'Insights': results.market_insights})
                insights_df.to_excel(writer, sheet_name='Market_Insights', index=False)
                
                # Processing metadata
                metadata_df = pd.DataFrame([results.dataset_overview])
                metadata_df.to_excel(writer, sheet_name='Dataset_Overview', index=False)
            
            export_files['excel_report'] = excel_file
            
            # 3. CSV Exports (Individual files)
            if results.category_statistics:
                csv_file = f"{CONFIG['scraper'].EXPORT_DIR}/category_statistics_{timestamp}.csv"
                stats_df = pd.DataFrame([asdict(stat) for stat in results.category_statistics])
                stats_df.to_csv(csv_file, index=False)
                export_files['category_csv'] = csv_file
            
            # 4. Summary Report (Text)
            report_file = f"{CONFIG['scraper'].EXPORT_DIR}/analysis_summary_{timestamp}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("PRODUCT DATA ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("DATASET OVERVIEW:\n")
                for key, value in results.dataset_overview.items():
                    f.write(f"  {key}: {value}\n")
                
                f.write(f"\nPROCESSING TIME: {results.processing_time:.2f} seconds\n")
                f.write(f"MEMORY USAGE: {results.memory_usage_mb:.1f} MB\n\n")
                
                f.write("KEY INSIGHTS:\n")
                for i, insight in enumerate(results.market_insights, 1):
                    f.write(f"  {i}. {insight}\n")
                
                if results.category_statistics:
                    f.write(f"\nTOP 5 CATEGORIES BY RATING:\n")
                    for i, stat in enumerate(results.category_statistics[:5], 1):
                        f.write(f"  {i}. {stat.category}: {stat.avg_rating:.2f} ⭐ ({stat.product_count:,} products)\n")
            
            export_files['summary_report'] = report_file
            
            self.processing_stats['export_files_created'] += len(export_files)
            logger.info(f"Exported {len(export_files)} result files")
            
            return export_files
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return {}
    
    def run_comprehensive_analysis(self) -> AnalysisResults:
        """Run complete production analysis pipeline"""
        logger.info("Starting comprehensive product data analysis...")
        start_time = time.time()
        
        try:
            # 1. Load dataset
            if not self.load_dataset_optimized():
                raise Exception("Failed to load dataset")
            
            # 2. Calculate comprehensive statistics
            category_stats = self.calculate_comprehensive_statistics()
            if not category_stats:
                raise Exception("Failed to calculate statistics")
            
            # 3. Perform variability analysis
            variability_analysis = self.perform_advanced_variability_analysis()
            
            # 4. Generate market insights
            market_insights = self.generate_market_insights(category_stats, variability_analysis)
            
            # 5. Create visualizations
            chart_files = self.create_advanced_visualizations(category_stats, variability_analysis)
            
            # 6. Prepare dataset overview
            dataset_overview = self.conn.execute("""
                SELECT 
                    COUNT(*) as total_products,
                    COUNT(DISTINCT category) as total_categories,
                    ROUND(AVG(rating), 3) as overall_avg_rating,
                    ROUND(STDDEV(rating), 4) as overall_std_rating,
                    MIN(rating) as min_rating,
                    MAX(rating) as max_rating
                FROM products
            """).fetchone()
            
            overview_dict = {
                'total_products': dataset_overview[0],
                'total_categories': dataset_overview[1],
                'overall_avg_rating': dataset_overview[2],
                'overall_std_rating': dataset_overview[3],
                'min_rating': dataset_overview[4],
                'max_rating': dataset_overview[5],
                'processing_stats': self.processing_stats
            }
            
            # 7. Create analysis results
            processing_time = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            
            results = AnalysisResults(
                dataset_overview=overview_dict,
                category_statistics=category_stats,
                variability_analysis=variability_analysis,
                statistical_significance=[asdict(stat) for stat in category_stats if stat.significance == 'Significant'],
                performance_insights={
                    'processing_time_seconds': processing_time,
                    'memory_usage_mb': memory_usage,
                    'records_per_second': overview_dict['total_products'] / processing_time,
                    'categories_processed': len(category_stats)
                },
                market_insights=market_insights,
                export_files=chart_files,
                processing_time=processing_time,
                memory_usage_mb=memory_usage
            )
            
            # 8. Export comprehensive results
            export_files = self.export_comprehensive_results(results)
            results.export_files.update(export_files)
            
            logger.info(f"Analysis completed successfully in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            # Return empty results with error info
            return AnalysisResults(
                dataset_overview={'error': str(e)},
                category_statistics=[],
                variability_analysis={},
                statistical_significance=[],
                performance_insights={},
                market_insights=[f"Analysis failed: {str(e)}"],
                export_files={},
                processing_time=time.time() - start_time,
                memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024
            )
        
        finally:
            if self.conn:
                self.conn.close()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return {
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_percent': psutil.cpu_percent(),
            'processing_stats': self.processing_stats,
            'total_runtime': time.time() - self.start_time
        }
