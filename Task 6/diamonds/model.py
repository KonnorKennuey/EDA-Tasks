# model.py - Обучение и сохранение модели регрессии для предсказания цены алмазов

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import os
import json

def train_model():
    data = pd.read_csv('data/preprocessed_data.csv')
    
    x = data.drop(['price'], axis=1)
    y = data['price']
    
    
    loss_function = 'RMSE' 
    learning_rate = 0.05
    iterations = 500
    metric = 'RMSE'

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        train_size=0.8,
        random_state=42
    )
    
    model = CatBoostRegressor(
        iterations=iterations,
        loss_function=loss_function,
        learning_rate=learning_rate,
        verbose=100,
        random_state=42,
        use_best_model=True,
        eval_metric=metric,
        depth=6,
        l2_leaf_reg=3,
        task_type='CPU'
    )
    
    model.fit(
        x_train, y_train,
        eval_set=(x_test, y_test),
        early_stopping_rounds=30
    )
    
    y_pred = model.predict(x_test)
    
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    try:
        cv_pool = Pool(x, y)
        cv_results = cv(
            cv_pool,
            model.get_params(),
            fold_count=5,
            verbose=False
        )
        cv_rmse = cv_results['test-RMSE-mean'].iloc[-1]
        print(f"CV RMSE: ${cv_rmse:.2f}")
    except Exception as e:
        print(f"Ошибка при кросс-валидации: {e}")
    
    feature_importance = pd.DataFrame({
        'feature': x.columns,
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head())
    
    
    os.makedirs('models', exist_ok=True)
    model_path = 'models/trained_model.cbm'
    model.save_model(model_path)
    
    return model

if __name__ == '__main__':
    try:
        train_model()
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")
        raise
