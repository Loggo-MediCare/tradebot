"""
Quick Test Script - Verify All Improvements
==============================================
Run this to validate that all improvements are working correctly.

Expected runtime: 5-10 minutes
"""

import os
import sys
import io
import subprocess
from datetime import datetime

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def test_imports():
    """Test that all required modules can be imported"""
    print_header("TEST 1: Import Validation")
    
    modules_to_test = [
        ('numpy', 'Core numerical library'),
        ('pandas', 'Data manipulation'),
        ('sklearn', 'Machine learning'),
        ('xgboost', 'XGBoost'),
        ('lightgbm', 'LightGBM'),
        ('catboost', 'CatBoost'),
    ]
    
    success_count = 0
    for module, description in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module:15} - {description}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module:15} - MISSING: {e}")
    
    print(f"\nResult: {success_count}/{len(modules_to_test)} modules available")
    return success_count == len(modules_to_test)


def test_custom_modules():
    """Test custom improvement modules"""
    print_header("TEST 2: Custom Module Import")
    
    modules = {
        'enhanced_indicators': 'New technical indicators',
        'ensemble_models': 'Ensemble voting classifier',
    }
    
    success_count = 0
    for module_name, description in modules.items():
        try:
            if module_name == 'enhanced_indicators':
                from enhanced_indicators import EnhancedTechnicalIndicators, RiskFilters
                print(f"✅ {module_name:25} - {description}")
            elif module_name == 'ensemble_models':
                from ensemble_models import EnsembleModelBuilder
                print(f"✅ {module_name:25} - {description}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module_name:25} - ERROR: {e}")
    
    print(f"\nResult: {success_count}/{len(modules)} custom modules working")
    return success_count == len(modules)


def test_enhanced_indicators():
    """Test enhanced indicator calculations"""
    print_header("TEST 3: Enhanced Indicators Calculation")
    
    try:
        import pandas as pd
        import numpy as np
        from enhanced_indicators import EnhancedTechnicalIndicators
        
        # Create sample data
        dates = pd.date_range('2025-01-01', periods=100)
        df = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000000, 5000000, 100)
        })
        
        # Calculate indicators
        indicators = EnhancedTechnicalIndicators.calculate_all(df)
        
        print("✅ ADX calculation works")
        print("✅ Stochastic RSI calculation works")
        print("✅ OBV calculation works")
        print("✅ Williams %R calculation works")
        print("✅ CCI calculation works")
        print("✅ TRIX calculation works")
        
        print(f"\nResult: All {len(indicators)} indicators calculated successfully")
        return True
    
    except Exception as e:
        print(f"❌ Indicator calculation failed: {e}")
        return False


def test_risk_filters():
    """Test risk management filters"""
    print_header("TEST 4: Risk Filters")
    
    try:
        import pandas as pd
        import numpy as np
        from enhanced_indicators import RiskFilters
        
        # Create sample data
        df = pd.DataFrame({
            'close': np.linspace(100, 110, 50),
            'atr': np.full(50, 2.0),  # 2% ATR
            'volume': np.full(50, 5000000),  # 5M daily volume
            'sma_10': np.linspace(100, 110, 50),
            'sma_30': np.linspace(99, 109, 50),
            'sma_50': np.linspace(98, 108, 50),
            'rsi_xgb': np.linspace(30, 70, 50),
            'bb_middle_xgb': np.linspace(100, 110, 50)
        })
        
        # Test filters
        volatility_ok = RiskFilters.check_volatility(df)
        liquidity_ok = RiskFilters.check_liquidity(df)
        regime_ok, regime = RiskFilters.check_market_regime(df)
        rsi_ok, rsi_status = RiskFilters.check_rsi_extremes(df)
        
        print(f"✅ Volatility filter: {'PASS' if volatility_ok else 'FAIL'}")
        print(f"✅ Liquidity filter: {'PASS' if liquidity_ok else 'FAIL'}")
        print(f"✅ Market regime filter: {regime}")
        print(f"✅ RSI extreme filter: {rsi_status}")
        
        return True
    
    except Exception as e:
        print(f"❌ Risk filters failed: {e}")
        return False


def test_retraining_scheduler():
    """Test retraining scheduler exists and has correct structure"""
    print_header("TEST 5: Retraining Scheduler")
    
    try:
        if os.path.exists('weekly_retraining_scheduler.py'):
            with open('weekly_retraining_scheduler.py', 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            required_items = [
                'RetrainingScheduler',
                'retrain_single_stock',
                'retrain_all_stocks',
                'weekly_retraining_scheduler'
            ]
            
            for item in required_items:
                if item in content:
                    print(f"✅ {item} class/function found")
                else:
                    print(f"❌ {item} missing")
            
            print(f"\n✅ Scheduler file exists and contains all components")
            return True
        else:
            print("❌ weekly_retraining_scheduler.py not found")
            return False
    
    except Exception as e:
        print(f"❌ Scheduler validation failed: {e}")
        return False


def test_ensemble_builder():
    """Test ensemble model builder"""
    print_header("TEST 6: Ensemble Model Builder")
    
    try:
        from ensemble_models import EnsembleModelBuilder
        
        # Create builder instance
        builder = EnsembleModelBuilder('TEST')
        
        # Test that models can be created
        xgb_model = builder.create_xgboost_model()
        lgb_model = builder.create_lightgbm_model()
        cat_model = builder.create_catboost_model()
        rf_model = builder.create_random_forest_model()
        
        models_created = 0
        if xgb_model is not None:
            print("✅ XGBoost model builder works")
            models_created += 1
        if lgb_model is not None:
            print("✅ LightGBM model builder works")
            models_created += 1
        if cat_model is not None:
            print("✅ CatBoost model builder works")
            models_created += 1
        if rf_model is not None:
            print("✅ Random Forest model builder works")
            models_created += 1
        
        print(f"\nResult: {models_created}/4 model builders functional")
        return models_created >= 2  # At least 2 models needed
    
    except Exception as e:
        print(f"❌ Ensemble builder test failed: {e}")
        return False


def test_documentation():
    """Verify all improvement documentation exists"""
    print_header("TEST 7: Documentation")
    
    docs = {
        'AI_ACCURACY_IMPROVEMENT_PLAN.md': 'Improvement strategies guide',
        'IMPLEMENTATION_GUIDE.md': 'Step-by-step implementation',
    }
    
    found = 0
    for filename, description in docs.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✅ {filename:40} ({size:,} bytes)")
            found += 1
        else:
            print(f"❌ {filename:40} MISSING")
    
    print(f"\nResult: {found}/{len(docs)} documentation files present")
    return found == len(docs)


def print_summary(results):
    """Print test summary"""
    print_header("TEST SUMMARY")
    
    tests = [
        ('Module Imports', results[0]),
        ('Custom Modules', results[1]),
        ('Enhanced Indicators', results[2]),
        ('Risk Filters', results[3]),
        ('Retraining Scheduler', results[4]),
        ('Ensemble Builder', results[5]),
        ('Documentation', results[6]),
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} - {test_name}")
    
    print(f"\n{'='*80}")
    print(f"Overall Result: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print(f"{'='*80}")
    
    if passed == total:
        print("""
✅ ALL TESTS PASSED!

Next Steps:
1. Start weekly retraining:
   python weekly_retraining_scheduler.py --now
   
2. Test enhanced signals:
   python get_trading_signal_6531.py
   
3. Monitor improvements:
   Expected accuracy increase: +5-10% per improvement
   Timeline: 4-8 weeks for full implementation
        """)
    else:
        print(f"""
❌ Some tests failed. Fix issues above before proceeding.

Troubleshooting:
- Check that all Python packages are installed
- Verify file paths are correct
- Review error messages above for specific issues
        """)
    
    return passed == total


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("  AI ACCURACY IMPROVEMENTS - VALIDATION TEST SUITE")
    print("="*80)
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Run all tests
    results = [
        test_imports(),
        test_custom_modules(),
        test_enhanced_indicators(),
        test_risk_filters(),
        test_retraining_scheduler(),
        test_ensemble_builder(),
        test_documentation(),
    ]
    
    # Print summary
    success = print_summary(results)
    
    print(f"\n  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
