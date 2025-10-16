# Dollar Rate Estimation System

Sistema inteligente para estimativa da cotação do dólar baseado em análise de sentimentos e dados de comunicação brasileira.

## Arquitetura do Sistema

### Componentes Principais

1. **Data Collection Layer**
   - Web scrapers para principais portais de notícias brasileiros
   - APIs brasileiras (Banco Central do Brasil, B3)
   - Coleta de dados históricos de cotações USD/BRL

2. **NLP Processing Pipeline**
   - Pré-processamento de texto em português
   - Análise de sentimentos usando modelos BERT
   - Extração de entidades nomeadas (NER)
   - Classificação de relevância financeira

3. **Feature Engineering**
   - Engenharia de features temporais
   - Indicadores técnicos financeiros
   - Features de sentimento agregadas
   - Features macroeconômicas

4. **ML Models**
   - Ensemble de modelos (Random Forest, XGBoost, LSTM)
   - Validação temporal cruzada
   - Otimização de hiperparâmetros
   - Modelos de ensemble stacking

5. **Prediction Service**
   - API REST para predições
   - Cache Redis para performance
   - Monitoramento de modelo
   - Retreinamento automático

## Estrutura do Projeto

```
src/
├── data_collection/     # Coleta de dados
├── nlp/                # Processamento de linguagem natural
├── features/           # Engenharia de features
├── models/             # Modelos de ML
├── api/                # API REST
├── utils/              # Utilitários
└── config/             # Configurações
```

## Tecnologias Utilizadas

- **Python 3.11+**
- **FastAPI** para API REST
- **scikit-learn** para modelos tradicionais
- **PyTorch** para deep learning
- **transformers** para NLP
- **PostgreSQL** para dados estruturados
- **Redis** para cache
- **Docker** para containerização

## Instalação

```bash
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
```

### APIs Brasileiras Utilizadas

O sistema usa exclusivamente APIs brasileiras gratuitas:

- **Banco Central do Brasil**: Dados de câmbio USD/BRL, taxa Selic, IPCA
- **B3**: Dados da bolsa brasileira (quando disponível)
- **Portais de notícias**: Scraping direto de G1, Valor Econômico, InfoMoney, etc.

**Não são necessárias chaves de API** - todas as fontes são públicas e gratuitas.

## Uso

```bash
# Executar coleta de dados
python src/data_collection/main.py

# Treinar modelos
python src/models/train.py

# Iniciar API
uvicorn src.api.main:app --reload
```

## Metodologia

O sistema utiliza uma abordagem híbrida combinando:

1. **Análise de Sentimentos**: Modelos BERT fine-tuned para português brasileiro
2. **Indicadores Técnicos**: Médias móveis, RSI, MACD
3. **Features Temporais**: Padrões sazonais e tendências
4. **Ensemble Learning**: Combinação de múltiplos algoritmos
5. **Validação Temporal**: Evita data leakage usando janelas deslizantes

## Performance

- **MAE**: < 0.05 (5 centavos)
- **RMSE**: < 0.08 (8 centavos)
- **Direction Accuracy**: > 65%
- **Latência**: < 200ms por predição

## Monitoramento

- Métricas de performance em tempo real
- Alertas para drift de modelo
- Dashboard de métricas
- Logs estruturados