# ==============================================================================
# Лабораторная работа №11: Метод K-ближайших соседей (k-NN)
# Задание 4: Прогнозирование оценки клиента на основе схожих отзывов
# Датасет: Womens Clothing E-Commerce Reviews (весь датасет)
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ======================== 1. Загрузка и подготовка данных ======================
print("=" * 60)
print("ЗАДАНИЕ 4: K-ближайших соседей (k-NN)")
print("Прогнозирование оценки клиента на основе схожих отзывов")
print("=" * 60)

df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv', index_col=0)
df = df.dropna(subset=['Review Text'])
print(f"\nРазмер датасета: {df.shape[0]} строк")

# ======================== 2. Подготовка признаков =============================
print("\n--- Шаг 2: Подготовка признаков ---")

# Числовые признаки
num_features = ['Age', 'Recommended IND', 'Positive Feedback Count']
X_num = df[num_features].fillna(0).values

# Текстовые признаки через TF-IDF
tfidf = TfidfVectorizer(max_features=200, stop_words='english')
X_text = tfidf.fit_transform(df['Review Text']).toarray()

# Объединяем числовые и текстовые признаки
X = np.hstack([X_num, X_text])
y = df['Rating'].values
print(f"Итоговый размер признаков: {X.shape}")
print(f"Распределение оценок:")
for rating in sorted(df['Rating'].unique()):
    count = (y == rating).sum()
    print(f"  Оценка {rating}: {count} ({count/len(y)*100:.1f}%)")

# Нормализация (стандартизация)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
print(f"\nОбучающая выборка: {X_train.shape[0]}")
print(f"Тестовая выборка: {X_test.shape[0]}")

# ======================== 3. Принцип работы KNN ===============================
print("\n--- Шаг 3: Принцип работы KNN ---")
print("""
Алгоритм K-ближайших соседей (K-Nearest Neighbors):
  1) Вычисляется расстояние между тестовым и всеми обучающими образцами
  2) Выбирается k-ближайших образцов (соседей), где k задаётся заранее
  3) Итоговым прогнозом будет мода среди выбранных k-ближайших образцов
     (класс, который встречается чаще всего)
  4) Предыдущие шаги повторяются для всех тестовых образцов
""")

# ======================== 4. Выбор оптимального K =============================
print("--- Шаг 4: Выбор оптимального k ---")
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    accuracies.append(acc)
    if k <= 10 or k % 5 == 0:
        print(f"  k={k}: accuracy = {acc:.4f}")

# Визуализация выбора k
plt.figure(figsize=(10, 5))
plt.plot(list(k_values), accuracies, 'bo-', linewidth=2)
plt.xlabel('Значение k (количество соседей)')
plt.ylabel('Точность (Accuracy)')
plt.title('Выбор оптимального k для KNN')
plt.grid(True, alpha=0.3)
best_k = list(k_values)[np.argmax(accuracies)]
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Лучший k={best_k}')
plt.legend()
plt.savefig('week11_knn_optimal_k.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nОптимальное k: {best_k} (accuracy = {max(accuracies):.4f})")
print("График сохранён: week11_knn_optimal_k.png")

# ======================== 5. Обучение KNN с оптимальным k =====================
print(f"\n--- Шаг 5: Обучение KNN (k={best_k}) ---")

# Евклидово расстояние – это наиболее простая и общепринятая метрика
knn_best = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
knn_best.fit(X_train, y_train)

y_pred = knn_best.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Точность на тестовой выборке: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ======================== 6. Сравнение метрик расстояния ======================
print("\n--- Шаг 6: Сравнение метрик расстояния ---")

metrics = {
    'euclidean': 'Евклидово',
    'manhattan': 'Манхэттенское',
    'cosine': 'Косинусное'
}

metric_accs = {}
for metric_name, metric_label in metrics.items():
    knn_m = KNeighborsClassifier(n_neighbors=best_k, metric=metric_name)
    knn_m.fit(X_train, y_train)
    acc_m = accuracy_score(y_test, knn_m.predict(X_test))
    metric_accs[metric_label] = acc_m
    print(f"  {metric_label}: accuracy = {acc_m:.4f}")

plt.figure(figsize=(8, 5))
bars = plt.bar(metric_accs.keys(), metric_accs.values(), 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.ylabel('Accuracy')
plt.title('Сравнение метрик расстояния для KNN')
for bar, acc in zip(bars, metric_accs.values()):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
             f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.savefig('week11_knn_metrics.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сохранён: week11_knn_metrics.png")

# ======================== 7. Confusion Matrix =================================
print("\n--- Шаг 7: Матрица ошибок ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation='nearest', cmap='YlOrRd')
classes = sorted(df['Rating'].unique())
ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
       xticklabels=classes, yticklabels=classes,
       ylabel='Истинная оценка',
       xlabel='Предсказанная оценка',
       title=f'Матрица ошибок KNN (k={best_k})')
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center', fontsize=10,
                color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.colorbar(im)
plt.tight_layout()
plt.savefig('week11_knn_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сохранён: week11_knn_confusion_matrix.png")

# ======================== 8. Примеры предсказаний =============================
print("\n--- Шаг 8: Примеры предсказаний ---")
for i in range(10):
    predicted = y_pred[i]
    actual = y_test[i]
    status = "✓" if predicted == actual else "✗"
    print(f"  Клиент {i+1}: Предсказано: {predicted}, Истинно: {actual} {status}")

print("\n" + "=" * 60)
print("ЗАДАНИЕ 4 ВЫПОЛНЕНО УСПЕШНО!")
print("=" * 60)

