import numpy as np

def riesgo_empirico(y_verdadero, y_predicho):
    return np.mean((y_verdadero - y_predicho)**2)