# ==============================================================================
# Лабораторная работа №12: Кластеризация методом K-средних
# Задание 1: Группировка клиентов по схожести в отзывах
# Датасет: Womens Clothing E-Commerce Reviews (весь датасет)
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ======================== 1. Загрузка и подготовка данных ======================
print("=" * 60)
print("ЗАДАНИЕ 1: Кластеризация клиентов (K-средних)")
print("=" * 60)

df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv', index_col=0)
print(f"\nРазмер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(f"Столбцы: {list(df.columns)}")

# Удаляем строки без отзывов
df = df.dropna(subset=['Review Text'])
print(f"Строк с отзывами: {df.shape[0]}")

# ======================== 2. Векторизация текста (TF-IDF) =====================
print("\n--- Шаг 2: Векторизация текста методом TF-IDF ---")
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = tfidf.fit_transform(df['Review Text'])
print(f"Размер матрицы TF-IDF: {X_tfidf.shape}")

# ======================== 3. Метод локтя для выбора K =========================
print("\n--- Шаг 3: Определение оптимального числа кластеров (метод локтя) ---")
inertias = []
K_range = range(2, 8)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_tfidf)
    inertias.append(kmeans.inertia_)
    print(f"  k={k}: инерция = {kmeans.inertia_:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(list(K_range), inertias, 'bo-', linewidth=2)
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Инерция (сумма квадратов расстояний)')
plt.title('Метод локтя для определения оптимального k')
plt.grid(True, alpha=0.3)
plt.savefig('week12_elbow_method.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сохранён: week12_elbow_method.png")

# ======================== 4. Кластеризация K-средних ==========================
optimal_k = 4
print(f"\n--- Шаг 4: Кластеризация K-средних (k={optimal_k}) ---")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_tfidf)
df['Cluster'] = clusters

# Silhouette Score
sil_score = silhouette_score(X_tfidf, clusters, sample_size=5000, random_state=42)
print(f"Silhouette Score: {sil_score:.4f}")

# ======================== 5. Анализ кластеров =================================
print("\n--- Шаг 5: Анализ кластеров ---")
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    print(f"\nКластер {i}: {len(cluster_data)} клиентов")
    print(f"  Средний рейтинг: {cluster_data['Rating'].mean():.2f}")
    print(f"  % рекомендаций: {cluster_data['Recommended IND'].mean() * 100:.1f}%")
    print(f"  Средний возраст: {cluster_data['Age'].mean():.1f}")
    
    # Топ-5 слов кластера
    cluster_tfidf = tfidf.transform(cluster_data['Review Text'])
    mean_tfidf = cluster_tfidf.mean(axis=0).A1
    top_indices = mean_tfidf.argsort()[-5:][::-1]
    feature_names = tfidf.get_feature_names_out()
    top_words = [feature_names[idx] for idx in top_indices]
    print(f"  Ключевые слова: {', '.join(top_words)}")

# ======================== 6. Визуализация кластеров (PCA) =====================
print("\n--- Шаг 6: Визуализация кластеров ---")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_tfidf.toarray())

plt.figure(figsize=(10, 7))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
for i in range(optimal_k):
    mask = clusters == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=colors[i], label=f'Кластер {i}', alpha=0.5, s=10)

plt.xlabel(f'Главная компонента 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'Главная компонента 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title(f'Кластеризация клиентов K-средних (k={optimal_k})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('week12_clusters_pca.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сохранён: week12_clusters_pca.png")

# ======================== 7. Распределение по отделам =========================
print("\n--- Шаг 7: Распределение кластеров по отделам ---")
ct = pd.crosstab(df['Cluster'], df['Department Name'])
print(ct)

ct_norm = ct.div(ct.sum(axis=1), axis=0)
ct_norm.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set2')
plt.title('Распределение отделов по кластерам')
plt.xlabel('Кластер')
plt.ylabel('Доля')
plt.legend(title='Отдел', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('week12_clusters_departments.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сохранён: week12_clusters_departments.png")

print("\n" + "=" * 60)
print("ЗАДАНИЕ 1 ВЫПОЛНЕНО УСПЕШНО!")
print("=" * 60)

