# -*- coding: utf-8 -*-
"""
Script de Ingestão de Dados
Carrega o dataset da pasta data/raw
"""

import pandas as pd
import os

def load_data(file_path='data/raw/pib-ocorrencias.csv'):
    """
    Carrega o dataset de criminalidade e PIB
    
    Parameters:
    -----------
    file_path : str
        Caminho para o arquivo CSV
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame com os dados carregados
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    df = pd.read_csv(file_path)
    
    print(f"✓ Dados carregados com sucesso!")
    print(f"  - Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"  - Período: {df['ano'].min()} a {df['ano'].max()}")
    print(f"  - Número de municípios: {df['municipio_agrupado'].nunique()}")
    
    return df

if __name__ == "__main__":
    # Teste do script
    df = load_data()
    print("\nPrimeiras linhas:")
    print(df.head())
    print("\nColunas disponíveis:")
    print(df.columns.tolist())