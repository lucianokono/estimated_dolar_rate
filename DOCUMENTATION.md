# Sistema de Estimativa de Cotação do Dólar

## Visão Geral

Este projeto implementa um sistema inteligente para estimativa da cotação do dólar baseado em análise de sentimentos e dados de comunicação brasileira. O sistema utiliza técnicas avançadas de Machine Learning e Processamento de Linguagem Natural para analisar notícias e dados financeiros, gerando predições precisas da cotação USD/BRL.

## Arquitetura do Sistema

### Componentes Principais

1. **Data Collection Layer**
   - Web scrapers para principais portais de notícias brasileiros
   - Integração com APIs financeiras (Alpha Vantage, Yahoo Finance)
   - Coleta automatizada de dados históricos de cotações

2. **NLP Processing Pipeline**
   - Pré-processamento de texto em português brasileiro
   - Análise de sentimentos usando modelos BERT fine-tuned
   - Extração de entidades nomeadas (NER)
   - Classificação de relevância financeira

3. **Feature Engineering**
   - Engenharia de features temporais
   - Indicadores técnicos financeiros (SMA, EMA, RSI, MACD, Bollinger Bands)
   - Features de sentimento agregadas
   - Features macroeconômicas

4. **ML Models**
   - Ensemble de modelos (Random Forest, XGBoost, Gradient Boosting)
   - Modelo LSTM para séries temporais
   - Ensemble stacking para combinação de modelos
   - Validação temporal cruzada

5. **Prediction Service**
   - API REST para predições em tempo real
   - Cache Redis para otimização de performance
   - Monitoramento de modelo e métricas
   - Retreinamento automático

## Tecnologias Utilizadas

### Backend
- **Python 3.11+** - Linguagem principal
- **FastAPI** - Framework web moderno e rápido
- **PostgreSQL** - Banco de dados principal
- **Redis** - Cache e sessões

### Machine Learning
- **scikit-learn** - Modelos tradicionais de ML
- **XGBoost** - Gradient boosting
- **PyTorch** - Deep learning (LSTM)
- **transformers** - Modelos BERT para NLP

### Processamento de Dados
- **pandas** - Manipulação de dados
- **numpy** - Computação numérica
- **spaCy** - Processamento de linguagem natural
- **NLTK** - Ferramentas de NLP

### Infraestrutura
- **Docker** - Containerização
- **Docker Compose** - Orquestração de serviços
- **uvicorn** - Servidor ASGI

## Estrutura do Projeto

```
src/
├── api/                    # API REST
│   └── main.py            # Endpoints principais
├── config/                # Configurações
│   └── settings.py        # Configurações do sistema
├── data_collection/       # Coleta de dados
│   └── main.py           # Scrapers e coletores
├── features/              # Engenharia de features
│   └── feature_engineering.py
├── models/                # Modelos de ML
│   └── ml_models.py      # Treinamento e predição
├── nlp/                   # Processamento de linguagem natural
│   └── sentiment_analysis.py
└── utils/                 # Utilitários
    └── helpers.py        # Funções auxiliares
```

## Instalação e Configuração

### Pré-requisitos
- Python 3.11+
- Docker e Docker Compose
- PostgreSQL 15+
- Redis 7+

### Instalação Local

1. **Clone o repositório**
```bash
git clone <repository-url>
cd estimated_dolar_rate
```

2. **Instale as dependências**
```bash
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
```

3. **Configure as variáveis de ambiente**
```bash
cp env.example .env
# Edite o arquivo .env com suas configurações
```

4. **Execute o sistema**
```bash
# Usando Docker Compose (recomendado)
docker-compose up -d

# Ou localmente
uvicorn src.api.main:app --reload
```

### Configuração de APIs

Para funcionalidade completa, configure as seguintes APIs:

1. **Alpha Vantage** - Dados financeiros históricos
2. **News API** - Coleta de notícias (opcional)

## Uso da API

### Endpoints Principais

#### Predição
```http
POST /predict
Content-Type: application/json

{
    "date": "2024-01-15",
    "include_sentiment": true,
    "model_name": "ensemble"
}
```

#### Status do Sistema
```http
GET /health
```

#### Listar Modelos
```http
GET /models
```

#### Histórico de Predições
```http
GET /predictions/history?days=7
```

### Exemplo de Uso

```python
import requests

# Fazer predição
response = requests.post("http://localhost:8000/predict", json={
    "include_sentiment": True
})

prediction = response.json()
print(f"Predição: R$ {prediction['prediction']:.2f}")
print(f"Confiança: {prediction['confidence']:.2%}")
```

## Metodologia

### Pipeline de Dados

1. **Coleta**
   - Scraping de portais de notícias
   - APIs de dados financeiros
   - Dados históricos de cotações

2. **Processamento**
   - Limpeza e normalização de dados
   - Análise de sentimentos
   - Extração de features técnicas

3. **Treinamento**
   - Validação temporal cruzada
   - Ensemble de múltiplos modelos
   - Otimização de hiperparâmetros

4. **Predição**
   - Combinação de modelos
   - Cálculo de confiança
   - Cache de resultados

### Features Utilizadas

#### Técnicas
- Médias móveis (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bandas de Bollinger
- ATR (Average True Range)

#### Temporais
- Features cíclicas (seno/cosseno)
- Lags de preços
- Estatísticas de janela deslizante
- Indicadores de sazonalidade

#### Sentimento
- Score de sentimento agregado
- Momentum de sentimento
- Relevância financeira
- Entidades nomeadas

### Validação

- **Validação Temporal**: Evita data leakage usando janelas deslizantes
- **Métricas**: MAE, RMSE, R², Direction Accuracy
- **Cross-validation**: Time series split com 5 folds
- **Backtesting**: Validação em dados históricos

## Performance

### Métricas Alvo
- **MAE**: < 0.05 (5 centavos)
- **RMSE**: < 0.08 (8 centavos)
- **Direction Accuracy**: > 65%
- **Latência**: < 200ms por predição

### Otimizações
- Cache Redis para predições frequentes
- Processamento assíncrono de dados
- Batch processing para NLP
- Modelos otimizados para produção

## Monitoramento

### Métricas do Sistema
- Performance dos modelos
- Latência da API
- Uso de recursos
- Qualidade dos dados

### Alertas
- Drift de modelo
- Erros de predição
- Falhas de coleta de dados
- Performance degradada

## Desenvolvimento

### Estrutura de Código
- Separação clara de responsabilidades
- Padrões de design bem definidos
- Documentação inline
- Testes unitários

### Contribuição
1. Fork do repositório
2. Crie uma branch para sua feature
3. Implemente com testes
4. Submeta um Pull Request

### Testes
```bash
# Executar testes
pytest tests/

# Coverage
pytest --cov=src tests/
```

## Deploy

### Produção
```bash
# Build da imagem
docker build -t dollar-estimation-api .

# Deploy com Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

### Monitoramento
- Logs estruturados com Loguru
- Métricas com Prometheus
- Alertas com AlertManager
- Dashboard com Grafana

## Roadmap

### Próximas Funcionalidades
- [ ] Dashboard web interativo
- [ ] Modelos de deep learning mais avançados
- [ ] Integração com mais fontes de dados
- [ ] API de streaming para dados em tempo real
- [ ] Sistema de alertas personalizados

### Melhorias Técnicas
- [ ] Implementação de MLOps completo
- [ ] A/B testing de modelos
- [ ] AutoML para seleção de features
- [ ] Modelos explicáveis (SHAP, LIME)

## Suporte

Para dúvidas ou problemas:
- Abra uma issue no GitHub
- Consulte a documentação da API em `/docs`
- Entre em contato com a equipe de desenvolvimento

## Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para detalhes.
