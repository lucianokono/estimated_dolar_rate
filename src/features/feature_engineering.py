# Features para ML - implementação inicial
# TODO: otimizar alguns cálculos que estão lentos

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import talib
from loguru import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

from ..config.settings import settings


class TechnicalIndicators:
    """Indicadores técnicos - implementação básica mas funcional"""
    
    @staticmethod
    def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
        """Média móvel simples - básico mas funciona"""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
        """Média móvel exponencial - mais responsiva que SMA"""
        return prices.ewm(span=window).mean()
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """RSI - indicador de momentum, 14 é padrão"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD - convergência/divergência de médias móveis"""
        ema_fast = TechnicalIndicators.calculate_ema(prices, fast)
        ema_slow = TechnicalIndicators.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Bandas de Bollinger"""
        sma = TechnicalIndicators.calculate_sma(prices, window)
        std = prices.rolling(window=window).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()


class TemporalFeatures:
    """Criador de features temporais"""
    
    @staticmethod
    def create_time_features(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """Cria features temporais"""
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Features básicas de tempo
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['dayofweek'] = df[date_column].dt.dayofweek
        df['dayofyear'] = df[date_column].dt.dayofyear
        df['quarter'] = df[date_column].dt.quarter
        df['week'] = df[date_column].dt.isocalendar().week
        
        # Features cíclicas
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Features de sazonalidade
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['day'] <= 7
        df['is_month_end'] = df['day'] >= 25
        
        return df
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """Cria features de lag"""
        df = df.copy()
        
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
        """Cria features de janela deslizante"""
        df = df.copy()
        
        for col in columns:
            for window in windows:
                # Estatísticas básicas
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
                
                # Percentis
                df[f'{col}_rolling_q25_{window}'] = df[col].rolling(window=window).quantile(0.25)
                df[f'{col}_rolling_q75_{window}'] = df[col].rolling(window=window).quantile(0.75)
        
        return df


class SentimentFeatures:
    """Criador de features de sentimento"""
    
    @staticmethod
    def aggregate_sentiment_features(df: pd.DataFrame, sentiment_column: str = 'sentiment') -> pd.DataFrame:
        """Agrega features de sentimento por período"""
        df = df.copy()
        
        # Converter sentimento para numérico
        sentiment_map = {'NEGATIVE': -1, 'NEUTRAL': 0, 'POSITIVE': 1}
        df['sentiment_numeric'] = df[sentiment_column].map(sentiment_map)
        
        # Features agregadas por dia
        daily_sentiment = df.groupby('date').agg({
            'sentiment_numeric': ['mean', 'std', 'count'],
            'confidence': ['mean', 'std'],
            'relevance_score': ['mean', 'std']
        }).reset_index()
        
        # Achatar colunas multi-nível
        daily_sentiment.columns = ['date', 'sentiment_mean', 'sentiment_std', 'sentiment_count',
                                 'confidence_mean', 'confidence_std', 'relevance_mean', 'relevance_std']
        
        return daily_sentiment
    
    @staticmethod
    def create_sentiment_momentum(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
        """Cria features de momentum de sentimento"""
        df = df.copy()
        
        # Momentum simples
        df['sentiment_momentum'] = df['sentiment_mean'].diff(window)
        
        # Momentum exponencial
        df['sentiment_ema'] = df['sentiment_mean'].ewm(span=window).mean()
        df['sentiment_ema_momentum'] = df['sentiment_mean'] - df['sentiment_ema']
        
        return df


class FeatureSelector:
    """Seletor de features"""
    
    def __init__(self, method: str = 'f_regression', k: int = 20):
        self.method = method
        self.k = k
        self.selector = None
        self.selected_features = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Treina o seletor de features"""
        self.selector = SelectKBest(score_func=f_regression, k=self.k)
        self.selector.fit(X, y)
        self.selected_features = X.columns[self.selector.get_support()].tolist()
        
        logger.info(f"Selecionadas {len(self.selected_features)} features")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica seleção de features"""
        if self.selector is None:
            raise ValueError("Seletor não foi treinado. Chame fit() primeiro.")
        
        return X[self.selected_features]
    
    def get_feature_scores(self) -> Dict[str, float]:
        """Retorna scores das features"""
        if self.selector is None:
            return {}
        
        scores = dict(zip(self.selected_features, self.selector.scores_))
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


class FeaturePipeline:
    """Pipeline completo de engenharia de features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = FeatureSelector()
        self.is_fitted = False
    
    def create_all_features(self, 
                          financial_df: pd.DataFrame,
                          sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Cria todas as features"""
        logger.info("Iniciando criação de features")
        
        # Features técnicas
        technical_features = self._create_technical_features(financial_df)
        
        # Features temporais
        temporal_features = self._create_temporal_features(financial_df)
        
        # Features de sentimento
        sentiment_features = self._create_sentiment_features(sentiment_df)
        
        # Combinar todas as features
        all_features = self._combine_features(
            technical_features, temporal_features, sentiment_features
        )
        
        logger.info(f"Features criadas: {all_features.shape}")
        return all_features
    
    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features técnicas - alguns indicadores básicos"""
        features_df = df.copy()
        
        # médias móveis - testei diferentes períodos
        features_df['sma_5'] = TechnicalIndicators.calculate_sma(df['fechamento'], 5)
        features_df['sma_20'] = TechnicalIndicators.calculate_sma(df['fechamento'], 20)
        features_df['ema_12'] = TechnicalIndicators.calculate_ema(df['fechamento'], 12)
        features_df['ema_26'] = TechnicalIndicators.calculate_ema(df['fechamento'], 26)
        features_df['rsi'] = TechnicalIndicators.calculate_rsi(df['fechamento'])
        
        # MACD - útil para identificar tendências
        macd = TechnicalIndicators.calculate_macd(df['fechamento'])
        features_df['macd'] = macd['macd']
        features_df['macd_signal'] = macd['signal']
        features_df['macd_histogram'] = macd['histogram']
        
        # Bollinger - volatilidade
        bb = TechnicalIndicators.calculate_bollinger_bands(df['fechamento'])
        features_df['bb_upper'] = bb['upper']
        features_df['bb_middle'] = bb['middle']
        features_df['bb_lower'] = bb['lower']
        features_df['bb_width'] = (bb['upper'] - bb['lower']) / bb['middle']
        
        # ATR - volatilidade absoluta
        features_df['atr'] = TechnicalIndicators.calculate_atr(
            df['maxima'], df['minima'], df['fechamento']
        )
        
        return features_df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features temporais"""
        features_df = TemporalFeatures.create_time_features(df)
        
        # Features de lag
        price_columns = ['fechamento', 'abertura', 'maxima', 'minima']
        features_df = TemporalFeatures.create_lag_features(
            features_df, price_columns, [1, 2, 3, 5, 10]
        )
        
        # Features de janela deslizante
        features_df = TemporalFeatures.create_rolling_features(
            features_df, ['fechamento'], [5, 10, 20]
        )
        
        return features_df
    
    def _create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features de sentimento"""
        if df.empty:
            return pd.DataFrame()
        
        # Agregar sentimento por dia
        sentiment_features = SentimentFeatures.aggregate_sentiment_features(df)
        
        # Adicionar momentum
        sentiment_features = SentimentFeatures.create_sentiment_momentum(sentiment_features)
        
        return sentiment_features
    
    def _combine_features(self, 
                         technical_df: pd.DataFrame,
                         temporal_df: pd.DataFrame,
                         sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Combina todas as features"""
        # Merge por data
        combined_df = technical_df.merge(temporal_df, on='date', how='left')
        
        if not sentiment_df.empty:
            combined_df = combined_df.merge(sentiment_df, on='date', how='left')
        
        # Preencher valores faltantes
        combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
        
        return combined_df
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Treina e transforma features"""
        # Selecionar features
        self.feature_selector.fit(X, y)
        X_selected = self.feature_selector.transform(X)
        
        # Normalizar
        X_scaled = self.scaler.fit_transform(X_selected)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_selected.columns, index=X.index)
        
        self.is_fitted = True
        return X_scaled_df
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforma features (após fit)"""
        if not self.is_fitted:
            raise ValueError("Pipeline não foi treinado. Chame fit_transform() primeiro.")
        
        X_selected = self.feature_selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_selected.columns, index=X.index)
        
        return X_scaled_df


def main():
    """Função principal para teste"""
    # Dados de exemplo
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    financial_data = pd.DataFrame({
        'date': dates,
        'fechamento': 5.0 + np.cumsum(np.random.randn(100) * 0.01),
        'abertura': 5.0 + np.cumsum(np.random.randn(100) * 0.01),
        'maxima': 5.0 + np.cumsum(np.random.randn(100) * 0.01) + 0.1,
        'minima': 5.0 + np.cumsum(np.random.randn(100) * 0.01) - 0.1
    })
    
    sentiment_data = pd.DataFrame({
        'date': dates,
        'sentiment': np.random.choice(['POSITIVE', 'NEGATIVE', 'NEUTRAL'], 100),
        'confidence': np.random.rand(100),
        'relevance_score': np.random.rand(100)
    })
    
    # Criar pipeline
    pipeline = FeaturePipeline()
    features = pipeline.create_all_features(financial_data, sentiment_data)
    
    logger.info(f"Features criadas: {features.shape}")
    logger.info(f"Colunas: {list(features.columns)}")


if __name__ == "__main__":
    main()
