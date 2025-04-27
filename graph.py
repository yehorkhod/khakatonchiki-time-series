import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Загрузка данных
# Предполагается, что файл data.parquet находится в рабочей директории
# и столбец с датами называется 'Date'.
df = pd.read_parquet("data.parquet").set_index('Date')

# Выбор рядов Series2, Series3, Series4 и обработка пропусков
selected = df[['Series2', 'Series3', 'Series4']].copy()
selected.fillna(selected.mean(), inplace=True)

# 1. Построение отдельных графиков для каждой серии
for col in selected.columns:
    plt.figure(figsize=(10, 4))
    plt.plot(selected.index, selected[col], label=col)
    plt.title(f'Временной ряд {col}')
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 2. Применение PCA к выбранным рядам
#    - стандартизация данных
#    - выделение 1 главной компоненты
scaler = StandardScaler()
scaled = scaler.fit_transform(selected)

pca = PCA(n_components=1)
pc1 = pca.fit_transform(scaled)

# Формируем серию с индексом дат и значениями компоненты
pc_series = pd.Series(pc1.flatten(), index=selected.index, name='PC1_234')

# 3. Построение графика главной компоненты
plt.figure(figsize=(10, 4))
plt.plot(pc_series.index, pc_series, label='PC1_234', color='black')
explained = pca.explained_variance_ratio_[0]
plt.title(f'Первая главная компонента (объясненная дисперсия: {explained:.2%})')
plt.xlabel('Дата')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
