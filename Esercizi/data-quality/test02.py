import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Creazione di un dataset clienti di esempio
def create_sample_customer_dataset(n_samples=1000):
    """
    Crea un dataset di esempio con dati clienti.
    """
    np.random.seed(42)
    
    # Generazione dati normali
    data = {
        'customer_id': range(1, n_samples + 1),
        'età': np.random.normal(45, 15, n_samples).clip(18, 90),
        'spesa_mensile': np.random.normal(500, 200, n_samples).clip(0, None),
        'frequenza_acquisti': np.random.normal(5, 2, n_samples).clip(0, None),
        'tempo_ultimo_acquisto': np.random.normal(30, 20, n_samples).clip(0, None),
        'categoria_preferita': np.random.choice(['Elettronica', 'Abbigliamento', 'Alimentari', 'Casa'], n_samples),
        'metodo_pagamento': np.random.choice(['Carta', 'PayPal', 'Bonifico', 'Contanti'], n_samples)
    }
    
    # Creazione DataFrame
    df = pd.DataFrame(data)
    
    # Aggiunta di alcune anomalie
    n_anomalies = int(n_samples * 0.05)  # 5% di anomalie
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    # Inserimento anomalie
    df.loc[anomaly_indices, 'spesa_mensile'] *= np.random.uniform(5, 10, n_anomalies)
    df.loc[anomaly_indices, 'frequenza_acquisti'] *= np.random.uniform(3, 5, n_anomalies)
    
    return df

# 2. Preparazione dei dati
def prepare_data(df):
    """
    Prepara i dati per il modello, gestendo sia feature numeriche che categoriche.
    """
    # Definizione delle colonne per tipo
    numeric_features = ['età', 'spesa_mensile', 'frequenza_acquisti', 'tempo_ultimo_acquisto']
    categorical_features = ['categoria_preferita', 'metodo_pagamento']
    
    # Creazione del preprocessore
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    
    return preprocessor

# 3. Addestramento e predizione
def train_and_predict(df, contamination=0.1):
    """
    Addestra il modello e effettua le predizioni.
    """
    # Rimozione della colonna customer_id per l'addestramento
    features_df = df.drop('customer_id', axis=1)
    
    # Preparazione del pipeline
    preprocessor = prepare_data(features_df)
    
    # Creazione del pipeline completo
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', IsolationForest(contamination=contamination, random_state=42))
    ])
    
    # Addestramento e predizione
    predictions = pipeline.fit_predict(features_df)
    
    # Aggiunta risultati al DataFrame originale
    results_df = df.copy()
    results_df['is_anomaly'] = predictions == -1
    
    return results_df, pipeline

# 4. Analisi dei risultati
def analyze_results(df):
    """
    Analizza e visualizza i risultati del rilevamento anomalie.
    """
    # Statistiche di base
    print("\nRiepilogo Anomalie:")
    print(f"Totale clienti analizzati: {len(df)}")
    print(f"Numero anomalie rilevate: {df['is_anomaly'].sum()}")
    print(f"Percentuale anomalie: {(df['is_anomaly'].sum() / len(df)):.2%}")
    
    # Visualizzazione della distribuzione delle anomalie
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Spesa mensile vs Frequenza acquisti
    plt.subplot(2, 2, 1)
    plt.scatter(df[~df['is_anomaly']]['spesa_mensile'], 
               df[~df['is_anomaly']]['frequenza_acquisti'], 
               c='blue', label='Normali', alpha=0.5)
    plt.scatter(df[df['is_anomaly']]['spesa_mensile'], 
               df[df['is_anomaly']]['frequenza_acquisti'], 
               c='red', label='Anomalie', alpha=0.7)
    plt.xlabel('Spesa Mensile')
    plt.ylabel('Frequenza Acquisti')
    plt.legend()
    plt.title('Anomalie per Spesa e Frequenza')
    
    # Plot 2: Distribuzione anomalie per categoria preferita
    plt.subplot(2, 2, 2)
    anomaly_by_category = df[df['is_anomaly']]['categoria_preferita'].value_counts()
    plt.pie(anomaly_by_category, labels=anomaly_by_category.index, autopct='%1.1f%%')
    plt.title('Distribuzione Anomalie per Categoria')
    
    # Plot 3: Box plot spesa mensile
    plt.subplot(2, 2, 3)
    sns.boxplot(x='is_anomaly', y='spesa_mensile', data=df)
    plt.title('Distribuzione Spesa Mensile per Anomalie')
    
    # Plot 4: Box plot età
    plt.subplot(2, 2, 4)
    sns.boxplot(x='is_anomaly', y='età', data=df)
    plt.title('Distribuzione Età per Anomalie')
    
    plt.tight_layout()
    plt.show()
    
    return df[df['is_anomaly']].sort_values('spesa_mensile', ascending=False)

# 5. Funzione principale
def detect_customer_anomalies(df=None, contamination=0.05):
    """
    Esegue l'intero processo di rilevamento anomalie sui dati clienti.
    """
    if df is None:
        df = create_sample_customer_dataset()
    
    # Rilevamento anomalie
    results_df, model = train_and_predict(df, contamination)
    
    # Analisi risultati
    anomalies_df = analyze_results(results_df)
    
    return results_df, anomalies_df, model

# Esempio di utilizzo
if __name__ == "__main__":
    # Creazione o caricamento dati
    df = create_sample_customer_dataset()
    
    # Esecuzione analisi
    results_df, anomalies_df, model = detect_customer_anomalies(df)
    
    # Stampa primi 10 clienti anomali
    print("\nTop 10 Clienti Anomali:")
    print(anomalies_df[['customer_id', 'spesa_mensile', 'frequenza_acquisti', 'età']].head(10))