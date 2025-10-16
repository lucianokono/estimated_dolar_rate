# Modelos ML para predição do dólar
# Implementação inicial - alguns modelos ainda estão em desenvolvimento

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import joblib
from datetime import datetime
from loguru import logger

# Modelos tradicionais
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# XGBoost
import xgboost as xgb

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensemble
from sklearn.ensemble import VotingRegressor, StackingRegressor

from ..config.settings import settings


class LSTMModel(nn.Module):
    """LSTM básico - ainda testando diferentes arquiteturas"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # experimentei com diferentes tamanhos, 64 parece funcionar bem
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # pegar só a última saída (padrão para predição)
        last_output = lstm_out[:, -1, :]
        
        # dropout + camada final
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output


class ModelTrainer:
    """Treinador - começou simples mas foi ficando complexo"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.training_history = {}
    
    def create_models(self) -> Dict[str, Any]:
        """Cria modelos - alguns hiperparâmetros foram ajustados manualmente"""
        self.models = {
            # random forest funciona bem para este tipo de problema
            'random_forest': RandomForestRegressor(
                n_estimators=100,  # testei com 50, 100, 200 - 100 é um bom meio termo
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            # xgboost geralmente é o melhor
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,  # não muito profundo para evitar overfitting
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            
            # gradient boosting como backup
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            
            # modelos lineares para baseline
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale')  # svr é lento mas às vezes funciona
        }
        
        logger.info(f"Criados {len(self.models)} modelos")
        return self.models
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Dict[str, float]]:
        """Treina modelos - processo demorado mas necessário"""
        logger.info("Iniciando treinamento dos modelos")
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Treinando {name}")
            
            try:
                # fit do modelo
                model.fit(X_train, y_train)
                
                # fazer predições
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                # calcular métricas básicas
                train_mae = mean_absolute_error(y_train, train_pred)
                val_mae = mean_absolute_error(y_val, val_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                train_r2 = r2_score(y_train, train_pred)
                val_r2 = r2_score(y_val, val_pred)
                
                results[name] = {
                    'train_mae': train_mae,
                    'val_mae': val_mae,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'train_r2': train_r2,
                    'val_r2': val_r2
                }
                
                # salvar importância das features se disponível
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(
                        X_train.columns, model.feature_importances_
                    ))
                
                logger.info(f"{name} - Val MAE: {val_mae:.4f}, Val R²: {val_r2:.4f}")
                
            except Exception as e:
                logger.error(f"Erro ao treinar {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.training_history = results
        return results
    
    def create_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Cria modelo ensemble"""
        logger.info("Criando modelo ensemble")
        
        # Selecionar melhores modelos base
        base_models = []
        for name, model in self.models.items():
            if name in self.training_history and 'val_mae' in self.training_history[name]:
                base_models.append((name, model))
        
        if len(base_models) < 2:
            logger.warning("Poucos modelos válidos para ensemble")
            return None
        
        # Criar ensemble com voting
        ensemble = VotingRegressor(
            estimators=base_models,
            weights=None  # Peso igual para todos
        )
        
        # Treinar ensemble
        ensemble.fit(X_train, y_train)
        
        self.models['ensemble'] = ensemble
        logger.info("Ensemble criado com sucesso")
        
        return ensemble
    
    def create_stacking_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Cria ensemble com stacking"""
        logger.info("Criando ensemble com stacking")
        
        # Modelos base
        base_models = [
            ('rf', self.models['random_forest']),
            ('xgb', self.models['xgboost']),
            ('gb', self.models['gradient_boosting'])
        ]
        
        # Meta-modelo
        meta_model = LinearRegression()
        
        # Criar stacking ensemble
        stacking_ensemble = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=3,
            n_jobs=-1
        )
        
        # Treinar
        stacking_ensemble.fit(X_train, y_train)
        
        self.models['stacking'] = stacking_ensemble
        logger.info("Stacking ensemble criado com sucesso")
        
        return stacking_ensemble


class LSTMTrainer:
    """Treinador específico para LSTM"""
    
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_sequences(self, data: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara sequências para LSTM"""
        # Normalizar dados
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, data.columns.get_loc(target_col)])
        
        return np.array(X), np.array(y)
    
    def train_lstm(self, X: np.ndarray, y: np.ndarray, 
                   epochs: int = 100, batch_size: int = 32) -> Dict[str, List[float]]:
        """Treina modelo LSTM"""
        logger.info("Iniciando treinamento LSTM")
        
        # Converter para tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Criar dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Criar modelo
        input_size = X.shape[2]
        self.model = LSTMModel(input_size).to(self.device)
        
        # Otimizador e loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Treinamento
        history = {'loss': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            history['loss'].append(avg_loss)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        logger.info("Treinamento LSTM concluído")
        return history
    
    def predict_lstm(self, X: np.ndarray) -> np.ndarray:
        """Faz predições com LSTM"""
        if self.model is None:
            raise ValueError("Modelo LSTM não foi treinado")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        # Desnormalizar
        predictions_np = predictions.cpu().numpy()
        # Aqui seria necessário desnormalizar adequadamente
        
        return predictions_np


class ModelEvaluator:
    """Avaliador de modelos"""
    
    @staticmethod
    def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Avalia um modelo"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Métricas específicas para câmbio
        direction_accuracy = ModelEvaluator.calculate_direction_accuracy(y_true, y_pred)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_accuracy
        }
    
    @staticmethod
    def calculate_direction_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calcula acurácia de direção"""
        true_direction = np.sign(y_true.diff())
        pred_direction = np.sign(y_pred.diff())
        
        # Remover NaN do primeiro valor
        mask = ~(true_direction.isna() | pred_direction.isna())
        
        accuracy = (true_direction[mask] == pred_direction[mask]).mean()
        return accuracy
    
    @staticmethod
    def cross_validate_time_series(model: Any, X: pd.DataFrame, y: pd.Series, 
                                 cv_folds: int = 5) -> Dict[str, List[float]]:
        """Validação cruzada temporal"""
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        scores = {
            'mae': [],
            'rmse': [],
            'r2': []
        }
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Treinar e predizer
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Calcular métricas
            scores['mae'].append(mean_absolute_error(y_val, y_pred))
            scores['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
            scores['r2'].append(r2_score(y_val, y_pred))
        
        return scores


class ModelManager:
    """Gerenciador de modelos"""
    
    def __init__(self):
        self.trainer = ModelTrainer()
        self.lstm_trainer = LSTMTrainer()
        self.evaluator = ModelEvaluator()
        self.best_model = None
        self.best_score = float('inf')
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Treina todos os modelos"""
        logger.info("Iniciando treinamento completo")
        
        # Criar modelos
        self.trainer.create_models()
        
        # Treinar modelos tradicionais
        results = self.trainer.train_models(X_train, y_train, X_val, y_val)
        
        # Criar ensembles
        self.trainer.create_ensemble(X_train, y_train)
        self.trainer.create_stacking_ensemble(X_train, y_train)
        
        # Avaliar ensemble
        if 'ensemble' in self.trainer.models:
            ensemble_pred = self.trainer.models['ensemble'].predict(X_val)
            ensemble_score = self.evaluator.evaluate_model(y_val, ensemble_pred)
            results['ensemble'] = ensemble_score
        
        # Encontrar melhor modelo
        self._find_best_model(results, X_val, y_val)
        
        logger.info(f"Melhor modelo: {self.best_model}")
        logger.info(f"Melhor score: {self.best_score:.4f}")
        
        return results
    
    def _find_best_model(self, results: Dict[str, Any], X_val: pd.DataFrame, y_val: pd.Series):
        """Encontra o melhor modelo"""
        for name, metrics in results.items():
            if isinstance(metrics, dict) and 'val_mae' in metrics:
                if metrics['val_mae'] < self.best_score:
                    self.best_score = metrics['val_mae']
                    self.best_model = name
    
    def save_models(self, path: str = "models/"):
        """Salva modelos treinados"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.trainer.models.items():
            filename = f"{path}/{name}_model.pkl"
            joblib.dump(model, filename)
            logger.info(f"Modelo {name} salvo em {filename}")
    
    def load_models(self, path: str = "models/"):
        """Carrega modelos salvos"""
        import os
        
        for filename in os.listdir(path):
            if filename.endswith('_model.pkl'):
                name = filename.replace('_model.pkl', '')
                filepath = os.path.join(path, filename)
                self.trainer.models[name] = joblib.load(filepath)
                logger.info(f"Modelo {name} carregado de {filepath}")


def main():
    """Teste básico dos modelos"""
    # dados de exemplo
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples)
    })
    
    # target com alguma relação com as features
    y = pd.Series(2 * X['feature_1'] + 1.5 * X['feature_2'] + np.random.randn(n_samples) * 0.1)
    
    # split temporal (não aleatório)
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # treinar modelos
    manager = ModelManager()
    results = manager.train_all_models(X_train, y_train, X_val, y_val)
    
    logger.info("Resultados do treinamento:")
    for name, metrics in results.items():
        if isinstance(metrics, dict) and 'val_mae' in metrics:
            logger.info(f"{name}: MAE={metrics['val_mae']:.4f}, R²={metrics['val_r2']:.4f}")


if __name__ == "__main__":
    main()
