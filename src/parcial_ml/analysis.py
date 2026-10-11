import numpy as np
from parcial_ml.models import ajustar_modelo, predecir
from parcial_ml.risk import riesgo_empirico

def evaluar_grados(X_train, y_train, X_val, y_val, grados):
    resultados = []
    modelos = {}
    for d in grados:
        modelo = ajustar_modelo(X_train, y_train, d)
        y_train_pred = predecir(modelo, X_train)
        y_val_pred = predecir(modelo, X_val)
        
        r_train = riesgo_empirico(y_train, y_train_pred)
        r_val = riesgo_empirico(y_val, y_val_pred)
        
        resultados.append({'grado': d, 'riesgo_train': r_train, 'riesgo_val': r_val})
        modelos[d] = modelo
    return resultados, modelos

def calcular_varianza_malla(diccionario_modelos, grados, m=300):
    # Malla de m=300 puntos igualmente espaciados en [-2.5, 2.5]
    X_malla = np.linspace(-2.5, 2.5, m).reshape(-1, 1)
    V_d_resultados = {}
    predicciones_malla = {}
    
    for d in grados:
        # predicciones de los modelos ajustados en A, B, C para el grado d
        preds = np.array([predecir(modelo, X_malla) for modelo in diccionario_modelos[d]])
        
        # h_barra_d
        h_barra = np.mean(preds, axis=0)
        
        # Cálculo de V_d
        V_d = np.mean(np.mean((preds - h_barra)**2, axis=0))
        
        V_d_resultados[d] = V_d
        predicciones_malla[d] = preds
        
    return V_d_resultados, X_malla, predicciones_malla

def riesgo_promedio_validacion(diccionario_modelos, X_val, y_val, grados):
    R_val_promedio = {}
    for d in grados:
        riesgos = []
        for modelo in diccionario_modelos[d]:
            y_pred = predecir(modelo, X_val)
            riesgos.append(riesgo_empirico(y_val, y_pred))
        R_val_promedio[d] = np.mean(riesgos)
    return R_val_promedio