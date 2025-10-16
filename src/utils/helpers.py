# Utilitários diversos - alguns helpers que acabei criando ao longo do tempo
# TODO: organizar melhor esses métodos quando tiver tempo

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import re
import json
from loguru import logger
import hashlib
import pickle
from pathlib import Path


class DataValidator:
    """Validador básico - começou simples mas foi crescendo"""
    
    @staticmethod
    def validate_financial_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Valida dados financeiros - ainda tem alguns bugs que preciso arrumar"""
        issues = []
        
        # colunas que sempre precisamos ter
        required_columns = ['fechamento', 'abertura', 'maxima', 'minima']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            issues.append(f"Colunas obrigatórias ausentes: {missing_columns}")
        
        # verificar nulos
        null_counts = df.isnull().sum()
        if null_counts.any():
            issues.append(f"Valores nulos encontrados: {null_counts[null_counts > 0].to_dict()}")
        
        # preços negativos não fazem sentido
        price_columns = ['fechamento', 'abertura', 'maxima', 'minima']
        for col in price_columns:
            if col in df.columns:
                negative_count = (df[col] <= 0).sum()
                if negative_count > 0:
                    issues.append(f"Valores negativos/zero em {col}: {negative_count}")
        
        # máxima não pode ser menor que mínima (óbvio mas já vi isso acontecer)
        if all(col in df.columns for col in ['maxima', 'minima', 'fechamento', 'abertura']):
            invalid_high_low = (df['maxima'] < df['minima']).sum()
            if invalid_high_low > 0:
                issues.append(f"Maxima < Minima em {invalid_high_low} registros")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'record_count': len(df)
        }
    
    @staticmethod
    def validate_news_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Valida dados de notícias"""
        issues = []
        
        # Verificar colunas obrigatórias
        required_columns = ['title', 'source', 'collected_at']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            issues.append(f"Colunas obrigatórias ausentes: {missing_columns}")
        
        # Verificar títulos vazios
        if 'title' in df.columns:
            empty_titles = df['title'].isnull().sum() + (df['title'] == '').sum()
            if empty_titles > 0:
                issues.append(f"Títulos vazios: {empty_titles}")
        
        # Verificar datas
        if 'collected_at' in df.columns:
            try:
                pd.to_datetime(df['collected_at'])
            except:
                issues.append("Formato de data inválido")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'record_count': len(df)
        }


class DataCleaner:
    """Limpeza de dados"""
    
    @staticmethod
    def clean_financial_data(df: pd.DataFrame) -> pd.DataFrame:
        """Limpa dados financeiros"""
        df_clean = df.copy()
        
        # Remover duplicatas
        df_clean = df_clean.drop_duplicates()
        
        # Ordenar por data
        if 'date' in df_clean.columns:
            df_clean = df_clean.sort_values('date')
        
        # Interpolar valores faltantes
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_columns] = df_clean[numeric_columns].interpolate(method='linear')
        
        # Remover outliers extremos (usando IQR)
        for col in ['fechamento', 'abertura', 'maxima', 'minima']:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Substituir outliers por valores interpolados
                outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                if outlier_mask.any():
                    df_clean.loc[outlier_mask, col] = np.nan
                    df_clean[col] = df_clean[col].interpolate(method='linear')
        
        return df_clean
    
    @staticmethod
    def clean_news_data(df: pd.DataFrame) -> pd.DataFrame:
        """Limpa dados de notícias"""
        df_clean = df.copy()
        
        # Remover duplicatas
        df_clean = df_clean.drop_duplicates(subset=['title', 'url'])
        
        # Limpar títulos
        if 'title' in df_clean.columns:
            df_clean['title'] = df_clean['title'].str.strip()
            df_clean = df_clean[df_clean['title'] != '']
        
        # Converter datas
        if 'collected_at' in df_clean.columns:
            df_clean['collected_at'] = pd.to_datetime(df_clean['collected_at'])
        
        return df_clean


class CacheManager:
    """Cache simples usando pickle - não é o mais eficiente mas funciona"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        # TODO: migrar para redis quando tiver tempo
    
    def _get_cache_path(self, key: str) -> Path:
        """Gera caminho do cache"""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """Recupera item do cache - às vezes falha mas funciona na maioria das vezes"""
        cache_path = self._get_cache_path(key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                
                # verificar se expirou
                if 'expires_at' in data and datetime.now() > data['expires_at']:
                    cache_path.unlink()
                    return None
                
                return data['value']
                
            except Exception as e:
                logger.error(f"Erro ao ler cache {key}: {e}")
                cache_path.unlink()  # deletar arquivo corrompido
        
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Armazena item no cache"""
        cache_path = self._get_cache_path(key)
        
        try:
            data = {
                'value': value,
                'expires_at': datetime.now() + timedelta(seconds=ttl_seconds),
                'created_at': datetime.now()
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            logger.error(f"Erro ao salvar cache {key}: {e}")
    
    def delete(self, key: str):
        """Remove item do cache"""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
    
    def clear_expired(self):
        """Remove itens expirados"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                if 'expires_at' in data and datetime.now() > data['expires_at']:
                    cache_file.unlink()
                    
            except Exception as e:
                logger.error(f"Erro ao verificar cache {cache_file}: {e}")
                cache_file.unlink()


class MetricsCalculator:
    """Calculadora de métricas"""
    
    @staticmethod
    def calculate_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calcula MAPE (Mean Absolute Percentage Error)"""
        mask = y_true != 0
        if not mask.any():
            return float('inf')
        
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return mape
    
    @staticmethod
    def calculate_smape(y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calcula SMAPE (Symmetric Mean Absolute Percentage Error)"""
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        # Evitar divisão por zero
        mask = denominator != 0
        if not mask.any():
            return float('inf')
        
        smape = np.mean(numerator[mask] / denominator[mask]) * 100
        return smape
    
    @staticmethod
    def calculate_directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calcula acurácia direcional"""
        true_direction = np.sign(y_true.diff())
        pred_direction = np.sign(y_pred.diff())
        
        # Remover NaN
        mask = ~(true_direction.isna() | pred_direction.isna())
        
        if not mask.any():
            return 0.0
        
        accuracy = (true_direction[mask] == pred_direction[mask]).mean()
        return accuracy
    
    @staticmethod
    def calculate_hit_ratio(y_true: pd.Series, y_pred: pd.Series, threshold: float = 0.01) -> float:
        """Calcula hit ratio (predições dentro do threshold)"""
        errors = np.abs(y_true - y_pred)
        hits = (errors <= threshold).sum()
        return hits / len(y_true)


class TextProcessor:
    """Processador de texto"""
    
    @staticmethod
    def extract_dollar_mentions(text: str) -> List[Dict[str, Any]]:
        """Extrai menções ao dólar - regex não é perfeita mas pega a maioria"""
        patterns = [
            r'dólar\s+(?:a\s+)?(?:R\$?\s*)?(\d+(?:,\d+)?(?:\.\d+)?)',
            r'dolar\s+(?:a\s+)?(?:R\$?\s*)?(\d+(?:,\d+)?(?:\.\d+)?)',  # sem acento
            r'USD\s+(?:a\s+)?(?:R\$?\s*)?(\d+(?:,\d+)?(?:\.\d+)?)',
            r'R\$\s*(\d+(?:,\d+)?(?:\.\d+)?)\s+(?:por\s+)?dólar',
            r'R\$\s*(\d+(?:,\d+)?(?:\.\d+)?)\s+(?:por\s+)?dolar'
        ]
        
        mentions = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                mentions.append({
                    'value': match.group(1),
                    'position': match.start(),
                    'context': text[max(0, match.start()-20):match.end()+20]  # contexto de 20 chars
                })
        
        return mentions
    
    @staticmethod
    def extract_economic_indicators(text: str) -> Dict[str, List[str]]:
        """Extrai indicadores econômicos do texto"""
        indicators = {
            'selic': [],
            'ipca': [],
            'pib': [],
            'inflacao': [],
            'desemprego': []
        }
        
        patterns = {
            'selic': [r'selic\s+(?:de\s+)?(\d+(?:,\d+)?(?:\.\d+)?)', r'taxa\s+selic\s+(\d+(?:,\d+)?(?:\.\d+)?)'],
            'ipca': [r'ipca\s+(?:de\s+)?(\d+(?:,\d+)?(?:\.\d+)?)', r'inflação\s+(\d+(?:,\d+)?(?:\.\d+)?)'],
            'pib': [r'pib\s+(?:de\s+)?(\d+(?:,\d+)?(?:\.\d+)?)', r'produto\s+interno\s+bruto'],
            'inflacao': [r'inflação\s+(\d+(?:,\d+)?(?:\.\d+)?)', r'inflacao\s+(\d+(?:,\d+)?(?:\.\d+)?)'],
            'desemprego': [r'desemprego\s+(\d+(?:,\d+)?(?:\.\d+)?)', r'taxa\s+de\s+desemprego']
        }
        
        for indicator, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                indicators[indicator].extend(matches)
        
        return indicators


class ConfigManager:
    """Gerenciador de configurações"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Carrega configurações de arquivo"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.json'):
                    return json.load(f)
                else:
                    # Assumir formato YAML
                    import yaml
                    return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar configuração {config_path}: {e}")
            return {}
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """Salva configurações em arquivo"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.endswith('.json'):
                    json.dump(config, f, indent=2, ensure_ascii=False)
                else:
                    import yaml
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.error(f"Erro ao salvar configuração {config_path}: {e}")


class PerformanceMonitor:
    """Monitor de performance"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Inicia timer para operação"""
        self.start_times[operation] = datetime.now()
    
    def end_timer(self, operation: str) -> float:
        """Finaliza timer e retorna duração"""
        if operation not in self.start_times:
            return 0.0
        
        duration = (datetime.now() - self.start_times[operation]).total_seconds()
        
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(duration)
        del self.start_times[operation]
        
        return duration
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Retorna estatísticas de uma operação"""
        if operation not in self.metrics:
            return {}
        
        durations = self.metrics[operation]
        
        return {
            'count': len(durations),
            'mean': np.mean(durations),
            'std': np.std(durations),
            'min': np.min(durations),
            'max': np.max(durations),
            'total': np.sum(durations)
        }


def main():
    """Teste rápido dos helpers - só para verificar se está funcionando"""
    logger.info("Testando utilitários")
    
    # dados de exemplo para testar
    sample_data = pd.DataFrame({
        'fechamento': [5.0, 5.1, 5.05],
        'abertura': [4.9, 5.0, 5.1],
        'maxima': [5.1, 5.2, 5.15],
        'minima': [4.8, 4.9, 5.0]
    })
    
    validator = DataValidator()
    result = validator.validate_financial_data(sample_data)
    logger.info(f"Validação: {result}")
    
    # testar cache
    cache = CacheManager()
    cache.set("test_key", {"data": "test"}, 60)
    cached_data = cache.get("test_key")
    logger.info(f"Cache test: {cached_data}")
    
    # testar extração de texto
    text = "O dólar subiu para R$ 5,20 hoje após decisão do Copom"
    mentions = TextProcessor.extract_dollar_mentions(text)
    logger.info(f"Mentions: {mentions}")
    
    logger.info("Testes concluídos")


if __name__ == "__main__":
    main()
