import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from explainerdashboard import RegressionExplainer, ExplainerDashboard
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_dashboard():
    """
    Загружает обученную модель и тестовые данные из volumes
    Генерирует конфигурацию дашборда в YAML
    """
    
    try:
        model = CatBoostRegressor()
        model.load_model('models/trained_model.cbm')
        logger.info("Модель загружена")
        
        preprocessed_data = pd.read_csv('data/preprocessed_data.csv')
        preprocessed_data = preprocessed_data.dropna()
        
        X = preprocessed_data.drop(['price'], axis=1)
        y = preprocessed_data['price']

        X_test = X
        y_test = y
        
        explainer = RegressionExplainer(
            model, 
            X_test, 
            y_test,
            model_output='raw',
        )
        
        dashboard = ExplainerDashboard(
            explainer,
            title="Diamond Price Prediction Explorer",
            description="CatBoost регрессор для предсказания цены алмазов"
        )
        
        dashboard.to_yaml(
            "dashboard.yaml",
            explainerfile="explainer.joblib",
            dump_explainer=True
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при создании дашборда: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    try:
        generate_dashboard()
    except Exception as e:
        logger.error(f"Ошибка при создании дашборда: {e}")
        raise
