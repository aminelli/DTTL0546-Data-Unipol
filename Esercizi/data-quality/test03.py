import pandas as pd
import numpy as np

data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Nome': ['Mario Rossi', 'Luigi Verdi', 'Mario Rossi', 'Giovanni Bianchi', 'Maria Neri', 'Maria Neri', 'Paolo Gialli', 'Laura Blu', 'Laura Blu', 'Marco Viola'],
    'Email': ['mario.rossi@email.com', 'luigi.verdi@email.com', 'mario.rossi@email.com', 'giovanni.bianchi@email.com', 'maria.neri@email.com', 'maria.neri@email.com', 'paolo.gialli@email.com', 'laura.blu@email.com', 'laura.blu@email.com', 'marco.viola@email.com'],
    'Telefono': ['1234567890', '0987654321', '1234567890', '1122334455', '3344556677', '3344556677', '5566778899', '7788990011', '7788990011', '9900112233'],
    'Indirizzo': ['Via Roma 1', 'Via Milano 2', 'Via Roma 1', 'Via Torino 3', 'Via Napoli 4', 'Via Napoli 4', 'Via Firenze 5', 'Via Venezia 6', 'Via Venezia 6', 'Via Bologna 7'],
    'Città': ['Roma', 'Milano', 'Roma', 'Torino', 'Napoli', 'Napoli', 'Firenze', 'Venezia', 'Venezia', 'Bologna']
}

df = pd.DataFrame(data)
print("Dataset originale:")
print(df)

# Identificazione dei duplicati basati su tutte le colonne
duplicati = df[df.duplicated()]
print("\nDuplicati basati su tutte le colonne:")
print(duplicati)

# Identificazione dei duplicati basati su colonne specifiche (es. Nome e Email)
duplicati_nome_email = df[df.duplicated(['Nome', 'Email'])]
print("\nDuplicati basati su Nome e Email:")
print(duplicati_nome_email)

# Controllo dei valori mancanti
valori_mancanti = df.isnull().sum()
print("\nValori mancanti per colonna:")
print(valori_mancanti)

# Controllo di inconsistenze nei dati (es. numeri di telefono con meno di 10 cifre)
df['Telefono'] = df['Telefono'].astype(str)
anomalie_telefono = df[df['Telefono'].str.len() != 10]
print("\nAnomalie nei numeri di telefono:")
print(anomalie_telefono)

# Rimozione dei duplicati basati su tutte le colonne
df_pulito = df.drop_duplicates()

# Rimozione dei duplicati basati su colonne specifiche (es. Nome e Email)
df_pulito_nome_email = df.drop_duplicates(subset=['Nome', 'Email'])

# Correzione delle anomalie nei numeri di telefono (es. rimozione delle righe con numeri di telefono non validi)
df_pulito = df_pulito[df_pulito['Telefono'].str.len() == 10]

print("\nDataset pulito:")
print(df_pulito)

df_pulito.to_csv('anagrafica_clienti_pulita.csv', index=False)

print("\nAnalisi finale del dataset pulito:")
print(df_pulito.info())
print(df_pulito.describe())
