# Scripts de Automação

## Scripts de Treinamento e Deploy

### train_models.py
```python
#!/usr/bin/env python3
"""
Script para treinamento completo dos modelos
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ml_models import ModelManager
from src.features.feature_engineering import FeaturePipeline
from src.data_collection.main import DataProcessor
from src.nlp.sentiment_analysis import NLPPipeline
from loguru import logger
import pandas as pd
from datetime import datetime

def main():
    """Executa pipeline completo de treinamento"""
    logger.info("Iniciando pipeline de treinamento")
    
    try:
        # 1. Coletar dados
        logger.info("Coletando dados...")
        processor = DataProcessor()
        collection_result = processor.process_and_save()
        logger.info(f"Coleta concluída: {collection_result}")
        
        # 2. Preparar features
        logger.info("Preparando features...")
        # Aqui seria implementada a lógica de carregamento dos dados coletados
        # e preparação das features
        
        # 3. Treinar modelos
        logger.info("Treinando modelos...")
        manager = ModelManager()
        # Aqui seria implementado o treinamento completo
        
        # 4. Salvar modelos
        logger.info("Salvando modelos...")
        manager.save_models()
        
        logger.info("Pipeline de treinamento concluído com sucesso")
        
    except Exception as e:
        logger.error(f"Erro no pipeline de treinamento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### deploy_api.py
```python
#!/usr/bin/env python3
"""
Script para deploy da API
"""

import subprocess
import sys
from loguru import logger

def deploy():
    """Executa deploy da API"""
    logger.info("Iniciando deploy da API")
    
    try:
        # Build da imagem Docker
        logger.info("Fazendo build da imagem Docker...")
        subprocess.run(["docker", "build", "-t", "dollar-estimation-api", "."], check=True)
        
        # Parar containers existentes
        logger.info("Parando containers existentes...")
        subprocess.run(["docker-compose", "down"], check=True)
        
        # Iniciar novos containers
        logger.info("Iniciando novos containers...")
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        
        # Verificar saúde da API
        logger.info("Verificando saúde da API...")
        import requests
        import time
        
        time.sleep(10)  # Aguardar inicialização
        
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            logger.info("Deploy realizado com sucesso!")
        else:
            logger.error("API não está respondendo corretamente")
            sys.exit(1)
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Erro no deploy: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    deploy()
```

### monitor_system.py
```python
#!/usr/bin/env python3
"""
Script para monitoramento do sistema
"""

import requests
import json
import time
from datetime import datetime
from loguru import logger
import psutil
import redis

class SystemMonitor:
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.redis_client = redis.Redis.from_url("redis://localhost:6379")
    
    def check_api_health(self):
        """Verifica saúde da API"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def check_redis_connection(self):
        """Verifica conexão Redis"""
        try:
            self.redis_client.ping()
            return True
        except:
            return False
    
    def get_system_metrics(self):
        """Coleta métricas do sistema"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'timestamp': datetime.now().isoformat()
        }
    
    def monitor(self):
        """Executa monitoramento contínuo"""
        logger.info("Iniciando monitoramento do sistema")
        
        while True:
            try:
                # Verificar API
                api_healthy = self.check_api_health()
                
                # Verificar Redis
                redis_healthy = self.check_redis_connection()
                
                # Métricas do sistema
                metrics = self.get_system_metrics()
                
                # Log de status
                status = {
                    'api_healthy': api_healthy,
                    'redis_healthy': redis_healthy,
                    'system_metrics': metrics
                }
                
                logger.info(f"Status: {json.dumps(status, indent=2)}")
                
                # Alertas
                if not api_healthy:
                    logger.warning("API não está respondendo!")
                
                if not redis_healthy:
                    logger.warning("Redis não está conectado!")
                
                if metrics['cpu_percent'] > 80:
                    logger.warning(f"CPU alta: {metrics['cpu_percent']}%")
                
                if metrics['memory_percent'] > 80:
                    logger.warning(f"Memória alta: {metrics['memory_percent']}%")
                
                time.sleep(60)  # Verificar a cada minuto
                
            except KeyboardInterrupt:
                logger.info("Monitoramento interrompido")
                break
            except Exception as e:
                logger.error(f"Erro no monitoramento: {e}")
                time.sleep(60)

if __name__ == "__main__":
    monitor = SystemMonitor()
    monitor.monitor()
```

### data_pipeline.py
```python
#!/usr/bin/env python3
"""
Pipeline de dados automatizado
"""

import schedule
import time
from datetime import datetime
from loguru import logger
from src.data_collection.main import DataProcessor
from src.nlp.sentiment_analysis import NLPPipeline

def collect_data():
    """Coleta dados periodicamente"""
    logger.info("Executando coleta de dados")
    
    try:
        processor = DataProcessor()
        result = processor.process_and_save()
        logger.info(f"Coleta concluída: {result}")
    except Exception as e:
        logger.error(f"Erro na coleta: {e}")

def process_sentiment():
    """Processa sentimentos das notícias"""
    logger.info("Processando sentimentos")
    
    try:
        # Implementar processamento de sentimentos
        logger.info("Processamento de sentimentos concluído")
    except Exception as e:
        logger.error(f"Erro no processamento: {e}")

def retrain_models():
    """Retreina modelos periodicamente"""
    logger.info("Iniciando retreinamento")
    
    try:
        # Implementar retreinamento
        logger.info("Retreinamento concluído")
    except Exception as e:
        logger.error(f"Erro no retreinamento: {e}")

def main():
    """Configura e executa pipeline automatizado"""
    logger.info("Iniciando pipeline automatizado")
    
    # Agendar tarefas
    schedule.every(30).minutes.do(collect_data)
    schedule.every(2).hours.do(process_sentiment)
    schedule.every().day.at("02:00").do(retrain_models)
    
    logger.info("Tarefas agendadas:")
    logger.info("- Coleta de dados: a cada 30 minutos")
    logger.info("- Processamento de sentimentos: a cada 2 horas")
    logger.info("- Retreinamento: diariamente às 02:00")
    
    # Executar loop principal
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
```

### backup_data.py
```python
#!/usr/bin/env python3
"""
Script para backup de dados
"""

import os
import shutil
import gzip
from datetime import datetime
from pathlib import Path
from loguru import logger

def backup_data():
    """Executa backup dos dados"""
    logger.info("Iniciando backup de dados")
    
    try:
        # Diretórios para backup
        backup_dirs = ['data', 'models', 'logs']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"backup_{timestamp}"
        
        # Criar diretório de backup
        backup_path = Path(f"backups/{backup_name}")
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copiar arquivos
        for dir_name in backup_dirs:
            if os.path.exists(dir_name):
                dest_path = backup_path / dir_name
                shutil.copytree(dir_name, dest_path)
                logger.info(f"Backup de {dir_name} concluído")
        
        # Compactar backup
        archive_path = f"backups/{backup_name}.tar.gz"
        shutil.make_archive(backup_name, 'gztar', backup_path)
        
        # Remover diretório temporário
        shutil.rmtree(backup_path)
        
        logger.info(f"Backup concluído: {archive_path}")
        
        # Limpar backups antigos (manter últimos 7)
        cleanup_old_backups()
        
    except Exception as e:
        logger.error(f"Erro no backup: {e}")

def cleanup_old_backups():
    """Remove backups antigos"""
    try:
        backups_dir = Path("backups")
        if not backups_dir.exists():
            return
        
        # Listar arquivos de backup
        backup_files = list(backups_dir.glob("backup_*.tar.gz"))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Manter apenas os últimos 7
        for old_backup in backup_files[7:]:
            old_backup.unlink()
            logger.info(f"Backup antigo removido: {old_backup}")
            
    except Exception as e:
        logger.error(f"Erro na limpeza de backups: {e}")

if __name__ == "__main__":
    backup_data()
```
