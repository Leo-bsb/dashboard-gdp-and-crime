# -*- coding: utf-8 -*-
"""
Script de Pré-processamento e Transformação de Dados
Aplica feature engineering e prepara dados para modelagem
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class CrimeRateCalculator(BaseEstimator, TransformerMixin):
    """
    Calcula taxas de criminalidade por 100 mil habitantes
    """
    def __init__(self):
        self.crime_columns = [
            'vitimas_feminicidio', 'vitimas_homicidio_doloso',
            'vitimas_tentativa_homicidio', 'vitimas_totais',
            'vitimas_lesao_corporal_seguida_de_morte',
            'vitimas_transito_ou_decorrencia_dele', 
            'vitimas_sem_indicio_de_crime',
            'vitimas_latrocinio', 'vitimas_suicidios'
        ]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.crime_columns:
            if col in X.columns and 'Total_Habitantes' in X.columns:
                X[f'{col}_por100mil'] = (X[col] / X['Total_Habitantes'] * 100000)
        return X

class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Trata valores ausentes
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Preenche valores ausentes numéricos com mediana
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isna().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        return X

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Seleciona features relevantes para modelagem
    """
    def __init__(self, features=None):
        self.features = features
    
    def fit(self, X, y=None):
        if self.features is None:
            # Features padrão para predição de vitimas_totais
            self.features = ['Total_Habitantes', 'vl_pib_per_capta', 
                           'vl_agropecuaria', 'vl_industria', 'vl_servicos']
        return self
    
    def transform(self, X):
        return X[self.features]

def create_preprocessing_pipeline(features=None):
    """
    Cria pipeline de pré-processamento
    
    Parameters:
    -----------
    features : list, optional
        Lista de features a serem selecionadas
        
    Returns:
    --------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline de pré-processamento
    """
    pipeline = Pipeline([
        ('missing_handler', MissingValueHandler()),
        ('crime_rate_calc', CrimeRateCalculator()),
        ('feature_selector', FeatureSelector(features=features))
    ])
    
    return pipeline

def prepare_data_for_modeling(df, target='vitimas_totais'):
    """
    Prepara dados para modelagem
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com dados brutos
    target : str
        Nome da coluna alvo
        
    Returns:
    --------
    X : pandas.DataFrame
        Features
    y : pandas.Series
        Target
    """
    # Aplicar transformações básicas
    missing_handler = MissingValueHandler()
    df_processed = missing_handler.transform(df)
    
    # Separar features e target
    feature_cols = ['Total_Habitantes', 'vl_pib_per_capta', 
                   'vl_agropecuaria', 'vl_industria', 'vl_servicos']
    
    X = df_processed[feature_cols]
    y = df_processed[target]
    
    return X, y

if __name__ == "__main__":
    # Teste do script
    from data_ingestion import load_data
    
    df = load_data()
    print("\n=== Testando Pipeline de Pré-processamento ===")
    
    X, y = prepare_data_for_modeling(df)
    print(f"\n✓ Dados preparados:")
    print(f"  - Features shape: {X.shape}")
    print(f"  - Target shape: {y.shape}")
    print(f"\nFeatures utilizadas:")
    print(X.columns.tolist())