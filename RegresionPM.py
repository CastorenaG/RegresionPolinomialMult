import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos 
data = pd.read_csv('c:\\Fish.csv')

# Extraer las columnas Length1 y Length2
X1 = data['Length1']
X2 = data['Length2']

# Crear las columnas con las potencias y productos necesarios
X = pd.DataFrame({'Length1': X1, 'Length2': X2})
X['Length1^2'] = X['Length1'] ** 2
X['Length2^2'] = X['Length2'] ** 2
X['Length1*Length2'] = X['Length1'] * X['Length2']

# Normalizar los datos 
X = (X - X.mean()) / X.std()

# Crear la matriz de diseño y el vector de etiquetas
X = X.to_numpy()
y = data['Weight'].to_numpy()

# Agregar una columna de unos para el término de sesgo (bias)
X = np.column_stack((np.ones(X.shape[0]), X))

# Definir hiperparámetros
learning_rate = 0.001  # Reducido para evitar problemas de convergencia
num_iterations = 11000
alpha = 0.1  # Parámetro de regularización
decay_rate = 0.9  # Tasa de decaimiento de la tasa de aprendizaje

# Inicializar los coeficientes con valores pequeños
coefficients = np.random.rand(X.shape[1]) * 0.01

# Lista para almacenar los coeficientes en cada iteración
coefficients_history = []

for iteration in range(num_iterations):
    # Calcular las predicciones
    predictions = np.dot(X, coefficients)

    # Calcular el error
    error = y - predictions

    # Calcular el gradiente con regularización L2
    gradient = -2 * np.dot(X.T, error) + 2 * alpha * coefficients

    # Actualizar los coeficientes con la tasa de aprendizaje adaptativa
    learning_rate *= decay_rate
    coefficients -= learning_rate * gradient

    # Almacenar los coeficientes en esta iteración
    coefficients_history.append(coefficients.copy())

# Calcular el MSE
mse = ((y - predictions) ** 2).mean()

print(f'Coeficientes finales: {coefficients}')
print(f'MSE: {mse}')

#Grafica
plt.figure(figsize=(10, 6))
plt.scatter(X1, y, label='Datos Length1', color='blue')
plt.scatter(X2, y, label='Datos Length2', color='green')
plt.plot(X1, predictions, color='red', label='Línea de regresión')

plt.xlabel('Length')
plt.ylabel('Weight')
plt.legend()
plt.show()

