#!/usr/bin/env python3
"""
Production-Ready Enhanced Web Scraper & Product Data Analysis
Main application with advanced features, comprehensive logging, and scalable architecture
"""

import os
import sys
import logging
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional
import psutil
import json

# Import production modules
from scraper import AdvancedWebScraper
from analyzer import ProductionAnalyzer
from database import DatabaseManager
from config import CONFIG

class ProductionApplication:
    """Main production application class"""
    
    def __init__(self):
        self.setup_logging()
        self.setup_directories()
        self.start_time = time.time()
        self.performance_metrics = {
            'scraper_success': False,
            'analyzer_success': False,
            'total_execution_time': 0,
            'peak_memory_usage': 0,
            'files_generated': 0
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Production application initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        log_config = CONFIG['logging']
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_config.LOG_LEVEL))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_config.LOG_FILE,
            maxBytes=log_config.LOG_MAX_BYTES,
            backupCount=log_config.LOG_BACKUP_COUNT
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(log_config.LOG_FORMAT)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Suppress noisy third-party loggers
        logging.getLogger('selenium').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
    
    def setup_directories(self):
        """Setup required directories"""
        directories = [
            'data',
            'logs',
            'exports',
            'exports/charts',
            CONFIG['scraper'].EXPORT_DIR,
            CONFIG['analyzer'].CHARTS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def monitor_system_resources(self) -> Dict[str, float]:
        """Monitor system resource usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        metrics = {
            'memory_rss_mb': memory_info.rss / 1024 / 1024,
            'memory_vms_mb': memory_info.vms / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'system_memory_percent': psutil.virtual_memory().percent,
            'system_cpu_percent': psutil.cpu_percent()
        }
        
        # Update peak memory usage
        if metrics['memory_rss_mb'] > self.performance_metrics['peak_memory_usage']:
            self.performance_metrics['peak_memory_usage'] = metrics['memory_rss_mb']
        
        return metrics
    
    def run_advanced_web_scraper(self) -> Dict[str, Any]:
        """Run the advanced web scraper with comprehensive features"""
        self.logger.info("=" * 80)
        self.logger.info("🕷️  STARTING ADVANCED WEB SCRAPER")
        self.logger.info("=" * 80)
        
        try:
            # Initialize scraper
            scraper = AdvancedWebScraper()
            target_url = CONFIG['scraper'].TARGET_URL
            
            self.logger.info(f"🎯 Target URL: {target_url}")
            self.logger.info("📊 Starting comprehensive product and review extraction...")
            
            # Monitor resources before scraping
            initial_resources = self.monitor_system_resources()
            self.logger.info(f"Initial memory usage: {initial_resources['memory_rss_mb']:.1f} MB")
            
            # Perform comprehensive scraping
            scraped_data = scraper.scrape_product_comprehensive(target_url)
            
            if not scraped_data:
                self.logger.error("❌ Scraping failed - no data extracted")
                return {'success': False, 'error': 'No data extracted'}
            
            # Extract results
            product_data = scraped_data.get('product')
            reviews_data = scraped_data.get('reviews', [])
            scraper_stats = scraped_data.get('stats', {})
            exported_files = scraped_data.get('exported_files', {})
            
            self.logger.info(f"✅ Product extracted: {product_data.title[:60]}...")
            self.logger.info(f"✅ Reviews extracted: {len(reviews_data)}")
            self.logger.info(f"✅ Scraping duration: {scraper_stats.get('scraping_duration', 0):.2f}s")
            
            # Save to database
            db = DatabaseManager()
            
            # Convert dataclass to dict for database storage
            from dataclasses import asdict
            product_dict = asdict(product_data)
            reviews_dict = [asdict(review) for review in reviews_data]
            
            product_id = db.save_product(product_dict)
            if product_id and reviews_dict:
                db.save_reviews(reviews_dict, product_id)
            
            # Get database summary
            db_summary = db.get_product_summary()
            
            # Monitor resources after scraping
            final_resources = self.monitor_system_resources()
            
            # Performance metrics
            performance_metrics = scraper.get_performance_metrics()
            
            # Display comprehensive summary
            self.logger.info("\n📈 SCRAPING SUMMARY:")
            self.logger.info(f"   • Product Title: {product_data.title}")
            self.logger.info(f"   • Brand: {product_data.brand}")
            self.logger.info(f"   • Price: {product_data.price}")
            self.logger.info(f"   • Rating: {product_data.rating}/5.0")
            self.logger.info(f"   • Total Reviews Extracted: {len(reviews_data)}")
            self.logger.info(f"   • Database Products: {db_summary.get('total_products', 0)}")
            self.logger.info(f"   • Database Reviews: {db_summary.get('total_reviews', 0)}")
            self.logger.info(f"   • Export Files: {len(exported_files)}")
            
            self.logger.info("\n🔧 PERFORMANCE METRICS:")
            self.logger.info(f"   • Scraping Duration: {scraper_stats.get('scraping_duration', 0):.2f}s")
            self.logger.info(f"   • Success Rate: {scraper_stats.get('success_rate', 0):.1%}")
            self.logger.info(f"   • Memory Usage: {final_resources['memory_rss_mb']:.1f} MB")
            self.logger.info(f"   • CPU Usage: {final_resources['cpu_percent']:.1f}%")
            
            self.logger.info("\n📁 EXPORTED FILES:")
            for file_type, file_path in exported_files.items():
                self.logger.info(f"   • {file_type.upper()}: {file_path}")
            
            self.performance_metrics['scraper_success'] = True
            self.performance_metrics['files_generated'] += len(exported_files)
            
            return {
                'success': True,
                'product': product_data,
                'reviews': reviews_data,
                'stats': scraper_stats,
                'exported_files': exported_files,
                'database_summary': db_summary,
                'performance': performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"❌ Scraper error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def run_production_analyzer(self) -> Dict[str, Any]:
        """Run the production data analyzer with comprehensive features"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 STARTING PRODUCTION DATA ANALYSIS")
        self.logger.info("=" * 80)
        
        try:
            # Initialize analyzer
            analyzer = ProductionAnalyzer()
            
            self.logger.info("🔄 Loading and analyzing large-scale product dataset...")
            
            # Monitor resources before analysis
            initial_resources = self.monitor_system_resources()
            self.logger.info(f"Initial memory usage: {initial_resources['memory_rss_mb']:.1f} MB")
            
            # Run comprehensive analysis
            analysis_results = analyzer.run_comprehensive_analysis()
            
            if not analysis_results or not analysis_results.category_statistics:
                self.logger.error("❌ Analysis failed - no results generated")
                return {'success': False, 'error': 'Analysis failed'}
            
            # Monitor resources after analysis
            final_resources = self.monitor_system_resources()
            
            # Display comprehensive results
            self.logger.info("\n📈 ANALYSIS RESULTS:")
            self.logger.info("=" * 60)
            
            # Dataset overview
            overview = analysis_results.dataset_overview
            self.logger.info("📦 Dataset Overview:")
            self.logger.info(f"   • Total Products: {overview.get('total_products', 0):,}")
            self.logger.info(f"   • Categories Analyzed: {overview.get('total_categories', 0)}")
            self.logger.info(f"   • Overall Avg Rating: {overview.get('overall_avg_rating', 0):.3f}")
            self.logger.info(f"   • Rating Std Dev: {overview.get('overall_std_rating', 0):.4f}")
            self.logger.info(f"   • Rating Range: {overview.get('min_rating', 0):.1f} - {overview.get('max_rating', 0):.1f}")
            
            # Top performing categories
            if analysis_results.category_statistics:
                self.logger.info("\n🏆 Top 5 Categories by Rating:")
                for i, stat in enumerate(analysis_results.category_statistics[:5], 1):
                    significance = "📊" if stat.significance == "Significant" else "⚪"
                    self.logger.info(f"   {i}. {stat.category}: {stat.avg_rating:.3f} ⭐ "
                                   f"({stat.product_count:,} products) {significance}")
                
                self.logger.info("\n📉 Bottom 3 Categories by Rating:")
                for i, stat in enumerate(analysis_results.category_statistics[-3:], 1):
                    significance = "📊" if stat.significance == "Significant" else "⚪"
                    self.logger.info(f"   {i}. {stat.category}: {stat.avg_rating:.3f} ⭐ "
                                   f"({stat.product_count:,} products) {significance}")
            
            # Variability analysis
            if analysis_results.variability_analysis:
                var_analysis = analysis_results.variability_analysis
                
                self.logger.info("\n📊 Rating Variability Analysis:")
                
                if var_analysis.get('highest_variability'):
                    self.logger.info("   🔥 Highest Variability Categories:")
                    for cat in var_analysis['highest_variability'][:3]:
                        self.logger.info(f"      • {cat['category']}: σ={cat['std_deviation']:.4f}, "
                                       f"CV={cat['coefficient_variation']:.1f}%")
                
                if var_analysis.get('lowest_variability'):
                    self.logger.info("   ✅ Most Consistent Categories:")
                    for cat in var_analysis['lowest_variability'][:3]:
                        self.logger.info(f"      • {cat['category']}: σ={cat['std_deviation']:.4f}, "
                                       f"CV={cat['coefficient_variation']:.1f}%")
            
            # Statistical significance
            significant_count = len(analysis_results.statistical_significance)
            total_categories = len(analysis_results.category_statistics)
            self.logger.info(f"\n🔬 Statistical Significance:")
            self.logger.info(f"   • Significant Categories: {significant_count}/{total_categories}")
            
            if analysis_results.statistical_significance:
                self.logger.info("   • Top Significant Deviations:")
                for cat in analysis_results.statistical_significance[:3]:
                    z_score = cat.get('z_score', 0)
                    self.logger.info(f"      • {cat['category']}: Z={z_score:.3f}, "
                                   f"Rating={cat.get('avg_rating', 0):.3f}")
            
            # Market insights
            if analysis_results.market_insights:
                self.logger.info("\n🎯 Key Market Insights:")
                for insight in analysis_results.market_insights:
                    self.logger.info(f"   • {insight}")
            
            # Performance metrics
            perf = analysis_results.performance_insights
            self.logger.info(f"\n🔧 PERFORMANCE METRICS:")
            self.logger.info(f"   • Processing Time: {analysis_results.processing_time:.2f}s")
            self.logger.info(f"   • Memory Usage: {analysis_results.memory_usage_mb:.1f} MB")
            self.logger.info(f"   • Records/Second: {perf.get('records_per_second', 0):.0f}")
            self.logger.info(f"   • Categories Processed: {perf.get('categories_processed', 0)}")
            
            # Export files
            if analysis_results.export_files:
                self.logger.info("\n📁 GENERATED FILES:")
                for file_type, file_path in analysis_results.export_files.items():
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path) / 1024  # KB
                        self.logger.info(f"   • {file_type.upper()}: {file_path} ({file_size:.1f} KB)")
            
            self.performance_metrics['analyzer_success'] = True
            self.performance_metrics['files_generated'] += len(analysis_results.export_files)
            
            return {
                'success': True,
                'results': analysis_results,
                'performance': analyzer.get_performance_metrics()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Analysis error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def generate_execution_report(self, scraper_result: Dict[str, Any], 
                                analyzer_result: Dict[str, Any]) -> str:
        """Generate comprehensive execution report"""
        try:
            report_data = {
                'execution_timestamp': datetime.now().isoformat(),
                'total_execution_time': time.time() - self.start_time,
                'performance_metrics': self.performance_metrics,
                'scraper_results': {
                    'success': scraper_result.get('success', False),
                    'product_extracted': bool(scraper_result.get('product')),
                    'reviews_count': len(scraper_result.get('reviews', [])),
                    'export_files': len(scraper_result.get('exported_files', {}))
                },
                'analyzer_results': {
                    'success': analyzer_result.get('success', False),
                    'categories_analyzed': 0,
                    'insights_generated': 0,
                    'export_files': 0
                },
                'system_resources': self.monitor_system_resources()
            }
            
            # Update analyzer metrics if successful
            if analyzer_result.get('success') and analyzer_result.get('results'):
                results = analyzer_result['results']
                report_data['analyzer_results'].update({
                    'categories_analyzed': len(results.category_statistics),
                    'insights_generated': len(results.market_insights),
                    'export_files': len(results.export_files)
                })
            
            # Save execution report
            report_file = f"logs/execution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"📄 Execution report saved: {report_file}")
            return report_file
            
        except Exception as e:
            self.logger.error(f"Error generating execution report: {e}")
            return ""
    
    def run_production_pipeline(self) -> bool:
        """Run the complete production pipeline"""
        self.logger.info("🚀 ENHANCED WEB SCRAPER & PRODUCT DATA ANALYSIS")
        self.logger.info("🏭 Production Pipeline Starting...")
        self.logger.info("=" * 80)
        
        success_count = 0
        
        # Run web scraper
        scraper_result = self.run_advanced_web_scraper()
        if scraper_result.get('success'):
            success_count += 1
        
        # Run data analyzer
        analyzer_result = self.run_production_analyzer()
        if analyzer_result.get('success'):
            success_count += 1
        
        # Update final metrics
        self.performance_metrics['total_execution_time'] = time.time() - self.start_time
        
        # Generate execution report
        report_file = self.generate_execution_report(scraper_result, analyzer_result)
        
        # Final summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🏁 PRODUCTION PIPELINE COMPLETE")
        self.logger.info("=" * 80)
        
        final_resources = self.monitor_system_resources()
        
        self.logger.info(f"✅ Components completed successfully: {success_count}/2")
        self.logger.info(f"⏱️  Total execution time: {self.performance_metrics['total_execution_time']:.2f}s")
        self.logger.info(f"💾 Peak memory usage: {self.performance_metrics['peak_memory_usage']:.1f} MB")
        self.logger.info(f"📁 Files generated: {self.performance_metrics['files_generated']}")
        
        if success_count == 2:
            self.logger.info("🎉 ALL COMPONENTS COMPLETED SUCCESSFULLY!")
            self.logger.info("\n📂 Key Output Files:")
            self.logger.info("   • SQLite Database: data/scraper_data.db")
            self.logger.info("   • DuckDB Analytics: data/analysis.duckdb")
            self.logger.info("   • Comprehensive Reports: exports/")
            self.logger.info("   • Interactive Charts: exports/charts/")
            self.logger.info("   • Execution Logs: logs/")
            if report_file:
                self.logger.info(f"   • Execution Report: {report_file}")
        else:
            self.logger.warning("⚠️  Some components encountered issues.")
            self.logger.info("📋 Check logs for detailed error information")
        
        return success_count == 2

def main():
    """Main application entry point"""
    try:
        # Initialize production application
        app = ProductionApplication()
        
        # Run production pipeline
        success = app.run_production_pipeline()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Application interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
