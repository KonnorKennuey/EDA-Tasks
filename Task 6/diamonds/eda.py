# eda.py - Exploratory Data Analysis и подготовка данных для регрессии цены алмазов

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_and_prepare_diamonds():

    url = 'https://raw.githubusercontent.com/aiedu-courses/stepik_eda_and_dev_tools/main/datasets/diamonds_good.csv'
    df = pd.read_csv(url)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/diamonds_raw.csv', index=False)
    
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)

    df = df.rename(columns={"'x'" : 'x', "'y'" : 'y', "'z'" : 'z'})
    df.columns

    median_carat = df['carat'].median()

    df['carat'].fillna(median_carat, inplace=True)

    median_depth = df['depth'].median()

    df['depth'].fillna(median_depth, inplace=True)

    median_depth = round(df['y'].median(),2)

    df['y'].fillna(median_depth, inplace=True)

    df['cut'] = df['cut'].replace('Goood', 'Good')

    df["cut"] = df["cut"].replace({'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5})
    
    df["clarity"] = df["clarity"].replace({'I1': 7, 'SI2': 6, 'SI1': 5, 'VS2': 4, 'VS1': 3, 'VVS1': 2, 'VVS2': 2, 'IF': 1})
    
    dummies = pd.get_dummies(df["color"], prefix="color", dtype=int)
    df = pd.concat([df.drop(columns=["color"]), dummies], axis=1)
    
    # Сохраняем обработанные данные
    df.to_csv('data/preprocessed_data.csv', index=False)
    print(f"\nОбработанные данные сохранены")
    
    return df

if __name__ == '__main__':
    try:
        load_and_prepare_diamonds()
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        raise
