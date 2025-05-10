# -*- coding: utf-8 -*-
"""
Задание 2:
1 измените программу так, чтобы нейрон работал с тремя признаками
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data.csv')  
print(df.head())

# Подготовка данных
y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)

# Теперь используем ТРИ признака (столбцы 0, 1, 2)
X = df.iloc[:, [0, 1, 2]].values  


plt.figure()
plt.scatter(X[y==1, 0], X[y==1, 1], color='red', marker='o', label='Class 1')
plt.scatter(X[y==-1, 0], X[y==-1, 1], color='blue', marker='x', label='Class -1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Данные (первые два признака)')
plt.show()

# Функция нейрона для ТРЕХ признаков
def neuron(w, x):
    # w[0] - смещение (bias), w[1], w[2], w[3] - веса для признаков
    decision = w[0] + w[1]*x[0] + w[2]*x[1] + w[3]*x[2]  # Добавлен третий признак
    return 1 if decision >= 0 else -1

# Инициализация весов (теперь их 4: w0 + 3 признака)
w = np.random.random(4)
print("Начальные веса:", w)

# Обучение с тремя признаками
eta = 0.01
w_iter = []
errors = []

for epoch in range(10):  
    epoch_errors = 0
    for xi, target in zip(X, y):
        predict = neuron(w, xi)
        error = target - predict
        
        w[1:] += eta * error * xi  # w[1], w[2], w[3]
        w[0] += eta * error  # w[0] (bias)
        epoch_errors += int(error != 0)
    
    w_iter.append(w.copy())
    errors.append(epoch_errors)
    print(f"Эпоха {epoch}, ошибок: {epoch_errors}")


total_errors = sum(neuron(w, xi) != target for xi, target in zip(X, y))
print(f"\nИтоговые веса: {w}")
print(f"Всего ошибок после обучения: {total_errors}/{len(y)}")


plt.figure()
plt.scatter(X[y==1, 0], X[y==1, 1], color='red', marker='o')
plt.scatter(X[y==-1, 0], X[y==-1, 1], color='blue', marker='x')

xl = np.linspace(min(X[:,0]), max(X[:,0]), 50)
for i, weights in enumerate(w_iter[:10]):  
    
    z_fixed = np.mean(X[:, 2])
    yl = -(weights[0] + weights[1]*xl + weights[3]*z_fixed) / weights[2]
    plt.plot(xl, yl, alpha=0.3, label=f'Iter {i}' if i < 3 else "")

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Эволюция разделяющей границы (2D проекция)')
plt.legend()
plt.show()