import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Creazione di un dataset di esempio
def create_sample_dataset(n_samples=1000):
    """
    Crea un dataset di esempio con alcune anomalie.
    """
    # Dati normali
    np.random.seed(42)
    normal_data = np.random.normal(loc=0, scale=1, size=(n_samples, 2))
    
    # Aggiunta di alcune anomalie
    anomalies = np.random.uniform(low=-4, high=4, size=(int(n_samples * 0.1), 2))
    
    # Combinazione dei dati
    X = np.vstack([normal_data, anomalies])
    
    # Creazione DataFrame
    df = pd.DataFrame(X, columns=['feature1', 'feature2'])
    
    return df

# 2. Preparazione dei dati
def prepare_data(df):
    """
    Prepara i dati per il modello standardizzando le feature.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return X_scaled, scaler

# 3. Addestramento del modello
def train_isolation_forest(X, contamination=0.1, random_state=42):
    """
    Addestra un modello Isolation Forest.
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    model.fit(X)
    return model

# 4. Valutazione e visualizzazione
def evaluate_and_visualize(df, predictions, title="Rilevamento Anomalie"):
    """
    Visualizza i risultati del rilevamento anomalie.
    """
    plt.figure(figsize=(10, 6))
    # Plot punti normali
    plt.scatter(df[predictions == 1]['feature1'], 
               df[predictions == 1]['feature2'], 
               c='blue', label='Normali')
    # Plot anomalie
    plt.scatter(df[predictions == -1]['feature1'], 
               df[predictions == -1]['feature2'], 
               c='red', label='Anomalie')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# 5. Funzione principale
def detect_anomalies(df, contamination=0.1):
    """
    Esegue l'intero processo di rilevamento anomalie.
    """
    # Preparazione dei dati
    X_scaled, scaler = prepare_data(df)
    
    # Divisione in train e test
    X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
    
    # Addestramento modello
    model = train_isolation_forest(X_train, contamination=contamination)
    
    # Predizioni
    predictions = model.predict(X_scaled)
    
    # Calcolo percentuale anomalie
    anomaly_ratio = (predictions == -1).sum() / len(predictions)
    
    return predictions, model, anomaly_ratio

# Esempio di utilizzo
if __name__ == "__main__":

    print("== START ==")
    
    # Creazione dataset
    df = create_sample_dataset()
    
    # Rilevamento anomalie
    predictions, model, anomaly_ratio = detect_anomalies(df)
    
    # Stampa risultati
    print(f"Percentuale di anomalie rilevate: {anomaly_ratio:.2%}")
    
    # Visualizzazione risultati
    evaluate_and_visualize(df, predictions)

    print("== END ==")
    print("")
