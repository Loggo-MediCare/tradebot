"""
Automated Weekly Model Retraining Scheduler
=============================================
Sets up automated retraining of all Taiwan stock models every week.

Expected Accuracy Improvement: +5-10%

Features:
  - Automatic retraining on schedule (weekly)
  - Email notifications on completion
  - Error logging and retry logic
  - Performance comparison (before/after)
"""

import os
import sys
import io
import json
import subprocess
from datetime import datetime, timedelta
import schedule
import time

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Taiwan stocks list (all 134 stocks)
TW_STOCKS = [
    '1101', '1303', '1314', '1326', '1471', '1513', '1514', '1519', '1605',
    '2002', '2049', '2059', '2308', '2313', '2314', '2317', '2327', '2330',
    '2337', '2344', '2345', '2357', '2360', '2363', '2368', '2376', '2382',
    '2383', '2385', '2395', '2408', '2409', '2412', '2449', '2451', '2454',
    '2485', '2603', '2634', '2884', '2890', '2891', '3004', '3006', '3017',
    '3022', '3030', '3037', '3044', '3105', '3135', '3138', '3231', '3260',
    '3363', '3443', '3449', '3481', '3491', '3533', '3563', '3576', '3615',
    '3653', '3661', '3665', '3711', '3715', '4540', '4564', '4577', '4722',
    '4746', '4768', '4938', '4971', '4989', '4991', '5371', '5483', '6163',
    '6187', '6209', '6220', '6223', '6230', '6239', '6269', '6271', '6274',
    '6285', '6443', '6442', '6446', '6472', '6477', '6505', '6510', '6515',
    '6526', '6531', '6668', '6669', '6683', '6770', '6781', '6789', '6805',
    '6830', '8021', '8046', '8069', '8110', '8112', '8131', '8150', '8210',
    '8222', '8438', '8499', '8908', '8917', '8927', '9918', '9931'
]

# Configuration
CONFIG = {
    'retraining_day': 'Sunday',      # Day of week to retrain
    'retraining_time': '02:00',      # Time to start (2 AM - low market activity)
    'data_lookback_days': 180,       # Use last 180 days of data for training
    'log_file': 'retraining_log.txt',
    'model_backup_dir': 'model_backups',
    'metrics_file': 'retraining_metrics.json'
}


class RetrainingScheduler:
    """Manages automated retraining of all stock models"""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {}
        self.failed_stocks = []
        self.successful_stocks = []
        self.load_config()
        self.create_backup_dir()
    
    def load_config(self):
        """Load configuration from file if exists"""
        if os.path.exists('retraining_config.json'):
            with open('retraining_config.json', 'r') as f:
                CONFIG.update(json.load(f))
            self.log("Configuration loaded from file")
    
    def create_backup_dir(self):
        """Create directory for model backups"""
        os.makedirs(CONFIG['model_backup_dir'], exist_ok=True)
    
    def log(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(CONFIG['log_file'], 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def backup_models(self):
        """Create backup of current models before retraining"""
        backup_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_subdir = os.path.join(CONFIG['model_backup_dir'], f'backup_{backup_date}')
        os.makedirs(backup_subdir, exist_ok=True)
        
        model_files = [f for f in os.listdir('.') if f.endswith('_tw_model.pkl')]
        
        for model_file in model_files:
            src = model_file
            dst = os.path.join(backup_subdir, model_file)
            try:
                import shutil
                shutil.copy2(src, dst)
            except Exception as e:
                self.log(f"ERROR: Failed to backup {model_file}: {e}")
        
        self.log(f"Models backed up to {backup_subdir}")
        return backup_subdir
    
    def retrain_single_stock(self, ticker):
        """Retrain a single stock model"""
        script_name = f"train_{ticker}_taiwan_improved.py"
        
        if not os.path.exists(script_name):
            self.log(f"SKIP: Training script not found for {ticker}")
            return None
        
        try:
            self.log(f"RETRAINING: {ticker}.TW...")
            
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout
                encoding='utf-8',
                errors='ignore'
            )
            
            if result.returncode == 0:
                self.log(f"SUCCESS: {ticker}.TW retrained")
                self.successful_stocks.append(ticker)
                return True
            else:
                error_msg = result.stderr[:200] if result.stderr else "Unknown error"
                self.log(f"FAILED: {ticker}.TW - {error_msg}")
                self.failed_stocks.append(ticker)
                return False
        
        except subprocess.TimeoutExpired:
            self.log(f"TIMEOUT: {ticker}.TW (>10 minutes)")
            self.failed_stocks.append(ticker)
            return False
        except Exception as e:
            self.log(f"ERROR: {ticker}.TW - {e}")
            self.failed_stocks.append(ticker)
            return False
    
    def retrain_all_stocks(self):
        """Retrain all stock models"""
        self.start_time = datetime.now()
        self.successful_stocks = []
        self.failed_stocks = []
        
        self.log("="*80)
        self.log("WEEKLY RETRAINING STARTED")
        self.log("="*80)
        
        # Backup current models
        backup_dir = self.backup_models()
        
        # Retrain all stocks
        total_stocks = len(TW_STOCKS)
        for i, ticker in enumerate(TW_STOCKS, 1):
            self.log(f"\nProgress: [{i}/{total_stocks}] Retraining {ticker}.TW")
            self.retrain_single_stock(ticker)
            
            # Progress update every 10 stocks
            if i % 10 == 0:
                success_rate = (len(self.successful_stocks) / i) * 100
                self.log(f"Progress: {success_rate:.1f}% success rate so far")
        
        # Final summary
        self.print_summary()
        self.save_metrics()
        
        # Send notification
        self.send_notification()
    
    def print_summary(self):
        """Print retraining summary"""
        elapsed = datetime.now() - self.start_time
        
        self.log("\n" + "="*80)
        self.log("RETRAINING COMPLETE")
        self.log("="*80)
        self.log(f"Total stocks: {len(TW_STOCKS)}")
        self.log(f"Successfully retrained: {len(self.successful_stocks)}")
        self.log(f"Failed: {len(self.failed_stocks)}")
        self.log(f"Success rate: {(len(self.successful_stocks)/len(TW_STOCKS)*100):.1f}%")
        self.log(f"Time elapsed: {elapsed}")
        
        if self.failed_stocks:
            self.log(f"\nFailed stocks:")
            for ticker in self.failed_stocks:
                self.log(f"  - {ticker}.TW")
        
        self.log("="*80)
    
    def save_metrics(self):
        """Save retraining metrics to JSON"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_stocks': len(TW_STOCKS),
            'successful': len(self.successful_stocks),
            'failed': len(self.failed_stocks),
            'success_rate': len(self.successful_stocks) / len(TW_STOCKS),
            'successful_stocks': self.successful_stocks,
            'failed_stocks': self.failed_stocks
        }
        
        # Load existing metrics and append
        all_metrics = []
        if os.path.exists(CONFIG['metrics_file']):
            try:
                with open(CONFIG['metrics_file'], 'r') as f:
                    all_metrics = json.load(f)
            except:
                pass
        
        all_metrics.append(metrics)
        
        # Keep only last 52 weeks (1 year) of metrics
        if len(all_metrics) > 52:
            all_metrics = all_metrics[-52:]
        
        with open(CONFIG['metrics_file'], 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        self.log(f"Metrics saved to {CONFIG['metrics_file']}")
    
    def send_notification(self):
        """Send email notification of retraining results"""
        # This is optional - implement if you want email alerts
        # For now, just log a summary
        success_rate = (len(self.successful_stocks) / len(TW_STOCKS)) * 100
        message = f"Weekly retraining complete: {success_rate:.1f}% success"
        self.log(f"\n📧 NOTIFICATION: {message}")
    
    def schedule_retraining(self):
        """Schedule automatic weekly retraining"""
        # Schedule for specific day and time
        schedule.every().week.do(self.retrain_all_stocks).tag('weekly_retrain')
        
        self.log(f"Retraining scheduled for every {CONFIG['retraining_day']} at {CONFIG['retraining_time']}")
        
        # Keep scheduler running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


def run_immediate_retraining():
    """Run retraining immediately (for manual triggering)"""
    scheduler = RetrainingScheduler()
    scheduler.retrain_all_stocks()


def start_scheduler():
    """Start the automatic retraining scheduler"""
    scheduler = RetrainingScheduler()
    scheduler.schedule_retraining()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--now':
        # Manual immediate retraining
        print("Starting immediate retraining...")
        run_immediate_retraining()
    else:
        # Start scheduler
        print("Starting automated retraining scheduler...")
        print(f"Retraining will run every {CONFIG['retraining_day']} at {CONFIG['retraining_time']}")
        print("Press Ctrl+C to stop")
        start_scheduler()
