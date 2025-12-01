# -*- coding: utf-8 -*-
"""
Script de Modelagem
Treina modelos preditivos, avalia e salva o melhor modelo
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from data_ingestion import load_data
from data_processing import prepare_data_for_modeling, MissingValueHandler

class ModelTrainer:
    """
    Classe para treinar e avaliar modelos de regressão
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def create_models(self):
        """Cria dicionário de modelos a serem treinados"""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=self.random_state,
                max_depth=10
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=5
            )
        }
        return self.models
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Treina e avalia todos os modelos
        
        Parameters:
        -----------
        X_train, X_test : pandas.DataFrame
            Features de treino e teste
        y_train, y_test : pandas.Series
            Target de treino e teste
            
        Returns:
        --------
        results : dict
            Dicionário com métricas de cada modelo
        """
        if not self.models:
            self.create_models()
        
        print("\n" + "="*60)
        print("TREINAMENTO E AVALIAÇÃO DOS MODELOS")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\n🔹 Treinando {name}...")
            
            # Treinar modelo
            model.fit(X_train, y_train)
            
            # Fazer predições
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calcular métricas
            metrics = {
                'r2_train': r2_score(y_train, y_pred_train),
                'r2_test': r2_score(y_test, y_pred_test),
                'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae_train': mean_absolute_error(y_train, y_pred_train),
                'mae_test': mean_absolute_error(y_test, y_pred_test)
            }
            
            # Validação cruzada
            cv_scores = cross_val_score(model, X_train, y_train, 
                                       cv=5, scoring='r2')
            metrics['cv_r2_mean'] = cv_scores.mean()
            metrics['cv_r2_std'] = cv_scores.std()
            
            self.results[name] = metrics
            
            # Exibir resultados
            print(f"   R² (Treino): {metrics['r2_train']:.4f}")
            print(f"   R² (Teste):  {metrics['r2_test']:.4f}")
            print(f"   RMSE (Teste): {metrics['rmse_test']:.2f}")
            print(f"   CV R² (média ± std): {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
        
        # Selecionar melhor modelo
        best_model_name = max(self.results.keys(), 
                             key=lambda k: self.results[k]['r2_test'])
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        print("\n" + "="*60)
        print(f"🏆 MELHOR MODELO: {self.best_model_name}")
        print(f"   R² (Teste): {self.results[best_model_name]['r2_test']:.4f}")
        print("="*60)
        
        return self.results
    
    def save_model(self, filepath='models/best_model.pkl'):
        """Salva o melhor modelo"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'metrics': self.results[self.best_model_name],
            'all_results': self.results,
            'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(model_data, filepath)
        print(f"\n✓ Modelo salvo em: {filepath}")
        
    def get_results_dataframe(self):
        """Retorna DataFrame com resultados comparativos"""
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        return results_df

def main():
    """Função principal para executar o pipeline de modelagem"""
    print("\n" + "="*60)
    print("PIPELINE DE MACHINE LEARNING - CRIMINALIDADE RIDE/DF")
    print("="*60)
    
    # 1. Carregar dados
    print("\n📁 Carregando dados...")
    df = load_data()
    
    # 2. Preparar dados
    print("\n🔧 Preparando dados para modelagem...")
    X, y = prepare_data_for_modeling(df, target='vitimas_totais')
    
    # 3. Dividir em treino e teste
    print("\n✂️  Dividindo em conjuntos de treino e teste (60/40)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.44, random_state=42
    )
    
    print(f"   Treino: {X_train.shape[0]} amostras")
    print(f"   Teste:  {X_test.shape[0]} amostras")
    
    # 4. Treinar modelos
    trainer = ModelTrainer(random_state=42)
    trainer.create_models()
    results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # 5. Exibir tabela comparativa
    print("\n📊 TABELA COMPARATIVA DE MODELOS:")
    print(trainer.get_results_dataframe())
    
    # 6. Salvar modelo
    trainer.save_model('models/best_model.pkl')
    
    # 7. Salvar também o preprocessador
    preprocessor = MissingValueHandler()
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("✓ Preprocessador salvo em: models/preprocessor.pkl")
    
    return trainer

if __name__ == "__main__":
    trainer = main()