"""
API REST para Sistema de Estimativa de Cotação do Dólar

Esta API fornece endpoints para predições, monitoramento e gerenciamento
do sistema de estimativa de cotação do dólar.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
import redis
import json

from ..config.settings import settings
from ..models.ml_models import ModelManager
from ..features.feature_engineering import FeaturePipeline
from ..nlp.sentiment_analysis import NLPPipeline
from ..data_collection.main import DataProcessor


# Modelos Pydantic
class PredictionRequest(BaseModel):
    """Request para predição"""
    date: Optional[str] = Field(None, description="Data para predição (YYYY-MM-DD)")
    include_sentiment: bool = Field(True, description="Incluir análise de sentimento")
    model_name: Optional[str] = Field(None, description="Nome do modelo específico")

class PredictionResponse(BaseModel):
    """Response de predição"""
    prediction: float = Field(..., description="Predição da cotação")
    confidence: float = Field(..., description="Confiança da predição")
    model_used: str = Field(..., description="Modelo utilizado")
    features_used: int = Field(..., description="Número de features utilizadas")
    timestamp: datetime = Field(..., description="Timestamp da predição")

class ModelInfo(BaseModel):
    """Informações do modelo"""
    name: str
    type: str
    performance: Dict[str, float]
    last_trained: datetime
    is_active: bool

class SystemStatus(BaseModel):
    """Status do sistema"""
    status: str
    models_loaded: int
    last_prediction: Optional[datetime]
    system_uptime: str
    memory_usage: Dict[str, Any]

# Inicializar FastAPI
app = FastAPI(
    title="Dollar Rate Estimation API",
    description="API para estimativa de cotação do dólar baseada em análise de sentimentos",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependências globais
model_manager = ModelManager()
feature_pipeline = FeaturePipeline()
nlp_pipeline = NLPPipeline()
redis_client = redis.Redis.from_url(settings.REDIS_URL)

# Cache para predições
PREDICTION_CACHE_TTL = settings.PREDICTION_CACHE_TTL


@app.on_event("startup")
async def startup_event():
    """Inicialização da aplicação"""
    logger.info("Iniciando API de Estimativa de Cotação do Dólar")
    
    try:
        # Carregar modelos salvos
        model_manager.load_models()
        logger.info("Modelos carregados com sucesso")
        
        # Testar conexão Redis
        redis_client.ping()
        logger.info("Conexão Redis estabelecida")
        
    except Exception as e:
        logger.error(f"Erro na inicialização: {e}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raiz"""
    return {
        "message": "API de Estimativa de Cotação do Dólar",
        "version": "1.0.0",
        "status": "active"
    }


@app.get("/health", response_model=SystemStatus)
async def health_check():
    """Verificação de saúde do sistema"""
    try:
        # Verificar modelos
        models_loaded = len(model_manager.trainer.models)
        
        # Verificar última predição
        last_prediction = redis_client.get("last_prediction")
        if last_prediction:
            last_prediction = datetime.fromisoformat(last_prediction.decode())
        
        # Status geral
        status = "healthy" if models_loaded > 0 else "degraded"
        
        return SystemStatus(
            status=status,
            models_loaded=models_loaded,
            last_prediction=last_prediction,
            system_uptime="N/A",  # Implementar se necessário
            memory_usage={"status": "N/A"}  # Implementar se necessário
        )
        
    except Exception as e:
        logger.error(f"Erro no health check: {e}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Endpoint principal para predições"""
    try:
        # Verificar cache
        cache_key = f"prediction:{request.date}:{request.include_sentiment}:{request.model_name}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            logger.info("Retornando predição do cache")
            return PredictionResponse(**json.loads(cached_result))
        
        # Coletar dados se necessário
        if request.include_sentiment:
            background_tasks.add_task(collect_latest_data)
        
        # Preparar features
        features = await prepare_features(request.date)
        
        if features.empty:
            raise HTTPException(status_code=400, detail="Dados insuficientes para predição")
        
        # Fazer predição
        model_name = request.model_name or model_manager.best_model
        
        if model_name not in model_manager.trainer.models:
            raise HTTPException(status_code=400, detail=f"Modelo {model_name} não encontrado")
        
        model = model_manager.trainer.models[model_name]
        prediction = model.predict(features.iloc[[-1]])[0]
        
        # Calcular confiança (simplificado)
        confidence = calculate_confidence(model, features)
        
        # Criar resposta
        response = PredictionResponse(
            prediction=float(prediction),
            confidence=float(confidence),
            model_used=model_name,
            features_used=len(features.columns),
            timestamp=datetime.now()
        )
        
        # Salvar no cache
        redis_client.setex(
            cache_key, 
            PREDICTION_CACHE_TTL, 
            json.dumps(response.dict(), default=str)
        )
        
        # Atualizar última predição
        redis_client.set("last_prediction", datetime.now().isoformat())
        
        logger.info(f"Predição realizada: {prediction:.4f} (modelo: {model_name})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """Lista todos os modelos disponíveis"""
    try:
        models_info = []
        
        for name, model in model_manager.trainer.models.items():
            # Obter métricas de performance
            performance = {}
            if name in model_manager.training_history:
                performance = model_manager.training_history[name]
            
            model_info = ModelInfo(
                name=name,
                type=type(model).__name__,
                performance=performance,
                last_trained=datetime.now(),  # Implementar timestamp real
                is_active=(name == model_manager.best_model)
            )
            models_info.append(model_info)
        
        return models_info
        
    except Exception as e:
        logger.error(f"Erro ao listar modelos: {e}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")


@app.post("/models/{model_name}/retrain")
async def retrain_model(model_name: str, background_tasks: BackgroundTasks):
    """Retreina um modelo específico"""
    try:
        if model_name not in model_manager.trainer.models:
            raise HTTPException(status_code=404, detail=f"Modelo {model_name} não encontrado")
        
        # Adicionar tarefa em background
        background_tasks.add_task(retrain_model_task, model_name)
        
        return {"message": f"Retreinamento do modelo {model_name} iniciado"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao iniciar retreinamento: {e}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")


@app.get("/predictions/history")
async def get_prediction_history(days: int = 7):
    """Histórico de predições"""
    try:
        # Buscar predições do Redis
        pattern = "prediction:*"
        keys = redis_client.keys(pattern)
        
        predictions = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for key in keys:
            data = redis_client.get(key)
            if data:
                pred_data = json.loads(data)
                pred_date = datetime.fromisoformat(pred_data['timestamp'])
                
                if pred_date >= cutoff_date:
                    predictions.append(pred_data)
        
        # Ordenar por timestamp
        predictions.sort(key=lambda x: x['timestamp'])
        
        return {"predictions": predictions, "count": len(predictions)}
        
    except Exception as e:
        logger.error(f"Erro ao buscar histórico: {e}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")


@app.get("/features/importance")
async def get_feature_importance():
    """Importância das features"""
    try:
        importance_data = {}
        
        for model_name, importance in model_manager.trainer.feature_importance.items():
            # Ordenar por importância
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            importance_data[model_name] = {
                "features": sorted_features[:20],  # Top 20
                "total_features": len(importance)
            }
        
        return importance_data
        
    except Exception as e:
        logger.error(f"Erro ao buscar importância das features: {e}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")


# Funções auxiliares
async def prepare_features(prediction_date: Optional[str]) -> pd.DataFrame:
    """Prepara features para predição"""
    try:
        # Aqui seria implementada a lógica completa de preparação de features
        # Por simplicidade, retornamos um DataFrame de exemplo
        
        # Em produção, isso incluiria:
        # 1. Coleta de dados financeiros mais recentes
        # 2. Processamento de notícias
        # 3. Cálculo de indicadores técnicos
        # 4. Aplicação do pipeline de features
        
        features = pd.DataFrame({
            'feature_1': [np.random.randn()],
            'feature_2': [np.random.randn()],
            'feature_3': [np.random.randn()]
        })
        
        return features
        
    except Exception as e:
        logger.error(f"Erro ao preparar features: {e}")
        return pd.DataFrame()


def calculate_confidence(model: Any, features: pd.DataFrame) -> float:
    """Calcula confiança da predição"""
    try:
        # Implementação simplificada
        # Em produção, isso poderia usar:
        # - Variância das predições de ensemble
        # - Distância dos dados de treino
        # - Métricas de incerteza do modelo
        
        return np.random.uniform(0.7, 0.95)
        
    except Exception as e:
        logger.error(f"Erro ao calcular confiança: {e}")
        return 0.5


async def collect_latest_data():
    """Coleta dados mais recentes em background"""
    try:
        processor = DataProcessor()
        result = processor.process_and_save()
        logger.info(f"Coleta de dados concluída: {result}")
        
    except Exception as e:
        logger.error(f"Erro na coleta de dados: {e}")


async def retrain_model_task(model_name: str):
    """Tarefa de retreinamento em background"""
    try:
        logger.info(f"Iniciando retreinamento do modelo {model_name}")
        
        # Implementar lógica de retreinamento
        # 1. Coletar novos dados
        # 2. Preparar features
        # 3. Treinar modelo
        # 4. Validar performance
        # 5. Substituir modelo se melhor
        
        logger.info(f"Retreinamento do modelo {model_name} concluído")
        
    except Exception as e:
        logger.error(f"Erro no retreinamento do modelo {model_name}: {e}")


# Middleware para logging
@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware para logging de requests"""
    start_time = datetime.now()
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
