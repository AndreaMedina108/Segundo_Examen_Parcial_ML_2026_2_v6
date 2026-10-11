from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def ajustar_modelo(X, y, grado):
    modelo = make_pipeline(PolynomialFeatures(grado), LinearRegression())
    modelo.fit(X, y)
    return modelo

def predecir(modelo, X):
    return modelo.predict(X)