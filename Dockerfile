# Arquivo de configuração Docker
FROM python:3.11-slim

# Definir variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Definir diretório de trabalho
WORKDIR /app

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baixar modelo spaCy para português
RUN python -m spacy download pt_core_news_sm

# Copiar código da aplicação
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Criar diretórios necessários
RUN mkdir -p logs cache

# Expor porta
EXPOSE 8000

# Comando para executar a aplicação
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
