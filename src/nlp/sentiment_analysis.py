"""
Pipeline de Processamento de Linguagem Natural

Este módulo implementa análise de sentimentos e processamento de texto
específico para português brasileiro, utilizando modelos BERT fine-tuned.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
import torch
from loguru import logger
import nltk
from textblob import TextBlob
import spacy

from ..config.settings import settings


class TextPreprocessor:
    """Pré-processador de texto para português brasileiro"""
    
    def __init__(self):
        # Download de recursos NLTK se necessário
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Carregar modelo spaCy para português
        try:
            self.nlp = spacy.load("pt_core_news_sm")
        except OSError:
            logger.warning("Modelo spaCy não encontrado. Instale com: python -m spacy download pt_core_news_sm")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """Limpa e normaliza texto"""
        if not text:
            return ""
        
        # Converter para minúsculas
        text = text.lower()
        
        # Remover caracteres especiais mas manter acentos
        text = re.sub(r'[^\w\sáàâãéèêíìîóòôõúùûç]', ' ', text)
        
        # Remover espaços extras
        text = re.sub(r'\s+', ' ', text)
        
        # Remover URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remover números isolados (manter números com contexto)
        text = re.sub(r'\b\d+\b', '', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokeniza texto usando spaCy"""
        if self.nlp:
            doc = self.nlp(text)
            return [token.text for token in doc if not token.is_stop and not token.is_punct]
        else:
            # Fallback para tokenização simples
            return text.split()
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extrai entidades nomeadas"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities


class SentimentAnalyzer:
    """Analisador de sentimentos usando BERT"""
    
    def __init__(self):
        self.model_name = settings.SENTIMENT_MODEL
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Carrega modelo BERT para análise de sentimentos"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Criar pipeline de análise de sentimentos
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info(f"Modelo de sentimentos carregado: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de sentimentos: {e}")
            self.pipeline = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analisa sentimento de um texto"""
        if not self.pipeline:
            return self._fallback_sentiment(text)
        
        try:
            # Truncar texto se muito longo
            if len(text) > settings.MAX_TEXT_LENGTH:
                text = text[:settings.MAX_TEXT_LENGTH]
            
            result = self.pipeline(text)
            
            return {
                'sentiment': result[0]['label'],
                'confidence': result[0]['score'],
                'method': 'bert'
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de sentimentos: {e}")
            return self._fallback_sentiment(text)
    
    def _fallback_sentiment(self, text: str) -> Dict[str, Any]:
        """Análise de sentimento usando TextBlob como fallback"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = 'POSITIVE'
            elif polarity < -0.1:
                sentiment = 'NEGATIVE'
            else:
                sentiment = 'NEUTRAL'
            
            return {
                'sentiment': sentiment,
                'confidence': abs(polarity),
                'method': 'textblob'
            }
            
        except Exception as e:
            logger.error(f"Erro no fallback de sentimentos: {e}")
            return {
                'sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'method': 'fallback'
            }
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analisa sentimentos em lote"""
        results = []
        
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        
        return results


class FinancialRelevanceClassifier:
    """Classificador de relevância financeira"""
    
    def __init__(self):
        self.financial_keywords = {
            'dolar': ['dólar', 'dolar', 'usd', 'dólares', 'dolares'],
            'economia': ['economia', 'econômico', 'econômica', 'financeiro', 'financeira'],
            'inflacao': ['inflação', 'inflacao', 'ipca', 'preços', 'precos'],
            'juros': ['juros', 'selic', 'taxa', 'taxas'],
            'politica': ['política', 'politica', 'governo', 'ministro', 'presidente'],
            'mercado': ['mercado', 'bolsa', 'ações', 'acoes', 'investimento']
        }
    
    def calculate_relevance_score(self, text: str) -> float:
        """Calcula score de relevância financeira"""
        text_lower = text.lower()
        score = 0.0
        
        for category, keywords in self.financial_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1.0
        
        # Normalizar score
        max_possible_score = sum(len(keywords) for keywords in self.financial_keywords.values())
        normalized_score = min(score / max_possible_score, 1.0)
        
        return normalized_score
    
    def classify_relevance(self, text: str, threshold: float = 0.3) -> Dict[str, Any]:
        """Classifica relevância financeira"""
        score = self.calculate_relevance_score(text)
        
        return {
            'relevance_score': score,
            'is_relevant': score >= threshold,
            'category': self._get_dominant_category(text)
        }
    
    def _get_dominant_category(self, text: str) -> str:
        """Identifica categoria dominante no texto"""
        text_lower = text.lower()
        category_scores = {}
        
        for category, keywords in self.financial_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        return 'other'


class NLPPipeline:
    """Pipeline completo de NLP"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.relevance_classifier = FinancialRelevanceClassifier()
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Processa texto completo"""
        # Pré-processar
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Analisar sentimento
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(cleaned_text)
        
        # Classificar relevância
        relevance_result = self.relevance_classifier.classify_relevance(cleaned_text)
        
        # Extrair entidades
        entities = self.preprocessor.extract_entities(text)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'sentiment': sentiment_result,
            'relevance': relevance_result,
            'entities': entities,
            'processed_at': pd.Timestamp.now()
        }
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'title') -> pd.DataFrame:
        """Processa DataFrame completo"""
        logger.info(f"Processando {len(df)} textos")
        
        results = []
        for idx, row in df.iterrows():
            result = self.process_text(row[text_column])
            results.append(result)
        
        # Converter para DataFrame
        processed_df = pd.DataFrame(results)
        
        # Expandir colunas aninhadas
        sentiment_df = pd.json_normalize(processed_df['sentiment'])
        relevance_df = pd.json_normalize(processed_df['relevance'])
        
        # Combinar DataFrames
        final_df = pd.concat([
            processed_df[['original_text', 'cleaned_text', 'processed_at']],
            sentiment_df,
            relevance_df
        ], axis=1)
        
        logger.info(f"Processamento concluído: {len(final_df)} registros")
        return final_df


def main():
    """Função principal para teste do pipeline"""
    pipeline = NLPPipeline()
    
    # Texto de exemplo
    sample_text = "Dólar sobe para R$ 5,20 após decisão do Copom sobre taxa de juros"
    
    result = pipeline.process_text(sample_text)
    
    logger.info("Resultado do processamento:")
    logger.info(f"Sentimento: {result['sentiment']}")
    logger.info(f"Relevância: {result['relevance']}")
    logger.info(f"Entidades: {result['entities']}")


if __name__ == "__main__":
    main()
