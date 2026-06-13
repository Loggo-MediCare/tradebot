"""
Enhanced Technical Indicator Module for AI Trading Signals
============================================================
Adds advanced technical indicators to improve model accuracy

New Indicators Added:
  - ADX (Average Directional Index) - Trend strength 0-100
  - Stochastic RSI - Momentum confirmation
  - OBV (On-Balance Volume) - Volume trend detection
  - Williams %R - Overbought/oversold levels
  - CCI (Commodity Channel Index) - Mean reversion
  - ATR (Average True Range) - Volatility measure
  - TRIX - Triple Exponential Moving Average momentum
"""

import pandas as pd
import numpy as np
from scipy import stats


class EnhancedTechnicalIndicators:
    """Calculate enhanced technical indicators for better predictions"""
    
    @staticmethod
    def calculate_adx(df, period=14):
        """Average Directional Index - Measures trend strength (0-100)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Directional movements
        up = high.diff()
        down = -low.diff()
        
        pos_dm = up.where((up > down) & (up > 0), 0)
        neg_dm = down.where((down > up) & (down > 0), 0)
        
        pos_di = 100 * pos_dm.rolling(period).mean() / (atr + 1e-10)
        neg_di = 100 * neg_dm.rolling(period).mean() / (atr + 1e-10)
        
        # ADX
        di_diff = abs(pos_di - neg_di)
        di_sum = pos_di + neg_di
        dx = 100 * di_diff / (di_sum + 1e-10)
        adx = dx.rolling(period).mean()
        
        return adx, atr
    
    @staticmethod
    def calculate_stochastic_rsi(df, period=14, smooth=3):
        """Stochastic RSI - RSI applied to RSI for momentum confirmation"""
        # First calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Then apply stochastic to RSI
        rsi_min = rsi.rolling(period).min()
        rsi_max = rsi.rolling(period).max()
        stoch_rsi = 100 * (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
        
        # Smooth
        stoch_rsi_k = stoch_rsi.rolling(smooth).mean()
        stoch_rsi_d = stoch_rsi_k.rolling(smooth).mean()
        
        return stoch_rsi_k, stoch_rsi_d
    
    @staticmethod
    def calculate_obv(df):
        """On-Balance Volume - Cumulative volume-based indicator"""
        obv = np.where(df['close'] > df['close'].shift(1), df['volume'],
                       np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
        obv = pd.Series(obv, index=df.index).cumsum()
        obv_ema = obv.ewm(span=20).mean()
        return obv, obv_ema
    
    @staticmethod
    def calculate_williams_r(df, period=14):
        """Williams %R - Overbought/oversold indicator (-100 to 0)"""
        highest_high = df['high'].rolling(period).max()
        lowest_low = df['low'].rolling(period).min()
        
        williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-10)
        return williams_r
    
    @staticmethod
    def calculate_cci(df, period=20):
        """Commodity Channel Index - Mean reversion indicator"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (tp - sma) / (0.015 * mad + 1e-10)
        return cci
    
    @staticmethod
    def calculate_trix(df, period=15):
        """TRIX - Triple exponential moving average momentum"""
        ema1 = df['close'].ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()
        
        trix = 100 * ema3.pct_change()
        return trix
    
    @staticmethod
    def calculate_all(df):
        """Calculate all enhanced indicators"""
        indicators = {}
        
        # ADX and ATR
        indicators['adx'], indicators['atr'] = EnhancedTechnicalIndicators.calculate_adx(df)
        
        # Stochastic RSI
        indicators['stoch_rsi_k'], indicators['stoch_rsi_d'] = EnhancedTechnicalIndicators.calculate_stochastic_rsi(df)
        
        # OBV
        indicators['obv'], indicators['obv_ema'] = EnhancedTechnicalIndicators.calculate_obv(df)
        
        # Williams %R
        indicators['williams_r'] = EnhancedTechnicalIndicators.calculate_williams_r(df)
        
        # CCI
        indicators['cci'] = EnhancedTechnicalIndicators.calculate_cci(df)
        
        # TRIX
        indicators['trix'] = EnhancedTechnicalIndicators.calculate_trix(df)
        
        return indicators


class RiskFilters:
    """Risk management filters to avoid bad trades"""
    
    @staticmethod
    def check_volatility(df, max_atr_percent=5.0):
        """Filter out extremely volatile periods (safer to avoid)"""
        if len(df) < 20:
            return True  # Not enough data, allow trade
        
        current_atr = df['atr'].iloc[-1]
        current_price = df['close'].iloc[-1]
        atr_percent = (current_atr / current_price) * 100
        
        return atr_percent < max_atr_percent
    
    @staticmethod
    def check_liquidity(df, min_volume=1000000):
        """Ensure sufficient trading volume (avoid illiquid stocks)"""
        avg_volume_20d = df['volume'].tail(20).mean()
        return avg_volume_20d > min_volume
    
    @staticmethod
    def check_technical_alignment(df, technical_score):
        """Only trade when technical indicators align"""
        # Require at least 40% technical alignment for signal
        return technical_score >= 40
    
    @staticmethod
    def check_market_regime(df):
        """Avoid trading in ranging/consolidation markets"""
        sma10 = df['sma_10'].iloc[-1]
        sma30 = df['sma_30'].iloc[-1]
        sma50 = df['sma_50'].iloc[-1]
        
        # Strong trend (avoid ranging)
        if sma10 > sma30 > sma50:
            return True, "UPTREND"  # Strong uptrend - favorable
        elif sma10 < sma30 < sma50:
            return True, "DOWNTREND"  # Strong downtrend - favorable
        else:
            return False, "RANGING"  # Ranging market - avoid
    
    @staticmethod
    def check_rsi_extremes(df):
        """Avoid RSI extremes that indicate reversals"""
        rsi = df['rsi_xgb'].iloc[-1]
        
        if rsi > 80:  # Overbought - likely reversal down
            return False, "OVERBOUGHT"
        elif rsi < 20:  # Oversold - likely reversal up
            return False, "OVERSOLD"
        else:
            return True, "RSI_NORMAL"
    
    @staticmethod
    def check_trend_confirmation(df):
        """Ensure trend is confirmed by multiple indicators"""
        close = df['close'].iloc[-1]
        sma20 = df['bb_middle_xgb'].iloc[-1] if 'bb_middle_xgb' in df.columns else close
        
        # Price above middle band = uptrend
        # Price below middle band = downtrend
        if close > sma20:
            return True, "ABOVE_MA"
        else:
            return False, "BELOW_MA"


class SignalQualityScorer:
    """Score the quality of generated signals for confidence-based trading"""
    
    @staticmethod
    def calculate_confidence_score(df, model_prediction, technical_score):
        """
        Calculate confidence score (0-100) for a signal
        
        Factors:
        - Model prediction strength (0-40 points)
        - Technical indicator alignment (0-40 points)
        - Trend confirmation (0-20 points)
        """
        confidence = 0
        
        # Factor 1: Model strength (max 40 points)
        model_strength = min(abs(model_prediction), 1.0)
        confidence += model_strength * 40
        
        # Factor 2: Technical alignment (max 40 points)
        confidence += min(technical_score, 100) * 0.4
        
        # Factor 3: Trend confirmation (max 20 points)
        sma10 = df['sma_10'].iloc[-1]
        sma30 = df['sma_30'].iloc[-1]
        
        if sma10 > sma30:  # Uptrend
            if model_prediction > 0:
                confidence += 20  # Signal aligns with trend
        elif sma10 < sma30:  # Downtrend
            if model_prediction < 0:
                confidence += 20  # Signal aligns with trend
        
        return min(confidence, 100)
    
    @staticmethod
    def position_size_from_confidence(base_size=100, confidence=50):
        """
        Adjust position size based on confidence score
        
        confidence >= 75%: 100% position size
        confidence 65-75%: 75% position size
        confidence 55-65%: 50% position size
        confidence 45-55%: 25% position size
        confidence < 45%:  0% position size (no trade)
        """
        if confidence >= 75:
            return base_size * 1.0
        elif confidence >= 65:
            return base_size * 0.75
        elif confidence >= 55:
            return base_size * 0.5
        elif confidence >= 45:
            return base_size * 0.25
        else:
            return 0  # No trade
