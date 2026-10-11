import pandas as pd

def cargar_datos(ruta_archivo):
    df = pd.read_csv(ruta_archivo)
    return df[['x']].values, df['y'].values