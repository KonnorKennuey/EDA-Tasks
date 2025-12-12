# app.py - Flask приложение для предсказания цены алмазов

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
import os
import logging
import json
import pickle

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

model = None
feature_names = None
model_metrics = None

def load_model():
    global model, feature_names, model_metrics
    
    try:
        logger.info("Загружаем модель регрессии для цены алмазов...")
        model = CatBoostRegressor()
        model.load_model('models/trained_model.cbm')
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        raise

def encode_categorical_features(data):
    data_copy = data.copy()
    cut_mapping = {
        'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5
    }
    
    clarity_mapping = {
        'I1': 7, 'SI2': 6, 'SI1': 5, 'VS2': 4, 'VS1': 3, 
        'VVS2': 2, 'VVS1': 2, 'IF': 1
    }
    
    color_mapping = {
        'D': 'color_D', 'E': 'color_E', 'F': 'color_F', 
        'G': 'color_G', 'H': 'color_H', 'I': 'color_I', 'J': 'color_J'
    }
    
    if 'cut' in data_copy:
        cut_val = data_copy['cut']
        data_copy['cut'] = cut_mapping[cut_val]

    if 'clarity' in data_copy:
        clarity_val = data_copy['clarity']
        data_copy['clarity'] = clarity_mapping[clarity_val]
    
    if 'color' in data_copy:
        color_val = data_copy['color']
        del data_copy['color']
        for col in color_mapping.values():
            data_copy[col] = 1 if col == color_mapping[color_val] else 0
    
    return data_copy

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'Diamond Price Prediction Regressor',
        'task': 'Regression'
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получаем данные из запроса
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        logger.info(f"Получен запрос: {data}")
        
        try:
            # Преобразуем категорические переменные
            logger.info(f"entered")
            encoded_data = encode_categorical_features(data)
            logger.info(encoded_data)
            # Создаем DataFrame
            df = pd.DataFrame([encoded_data])
            
            
        except (KeyError, ValueError) as e:
            return jsonify({
                'error': f'Error processing data: {str(e)}',
                'required_features': feature_names,
                'encoding_info': {
                    'cut': ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
                    'clarity': ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'],
                    'color': ['D', 'E', 'F', 'G', 'H', 'I', 'J']
                }
            }), 400
        
        predicted_price = model.predict(df)[0]
        
        result = {
            'predicted_price': float(predicted_price),
        }
        
        logger.info(f"Результат: ${predicted_price:.2f}")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Обработка 404"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/health',
            '/predict',
        ]
    }), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Загружаем модель при запуске
    load_model()
    
    # Запускаем Flask приложение
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
