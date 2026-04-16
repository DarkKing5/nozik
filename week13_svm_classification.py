# ==============================================================================
# Лабораторная работа №13: Классификация данных с использованием SVM
# Задание 2: Классификация отзывов как положительных или отрицательных
# Датасет: Womens Clothing E-Commerce Reviews (до 100 строк!)
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

# ======================== 1. Загрузка и подготовка данных ======================
print("=" * 60)
print("ЗАДАНИЕ 2: Классификация отзывов (SVM)")
print("ВАЖНО: Используется до 100 строк данных!")
print("=" * 60)

df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv', index_col=0)
df = df.dropna(subset=['Review Text'])

# Берем ТОЛЬКО 100 строк (по требованию задания)
df = df.head(100)
print(f"\nИспользуется строк: {len(df)}")

# Создаём бинарную метку: положительный (Rating >= 4) / отрицательный (Rating < 4)
df['Sentiment'] = (df['Rating'] >= 4).astype(int)
print(f"Положительных отзывов: {df['Sentiment'].sum()}")
print(f"Отрицательных отзывов: {(1 - df['Sentiment']).sum()}")

# ======================== 2. Векторизация текста (TF-IDF) =====================
print("\n--- Шаг 2: Векторизация текста методом TF-IDF ---")
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
X = tfidf.fit_transform(df['Review Text'])
y = df['Sentiment'].values
print(f"Размер матрицы TF-IDF: {X.shape}")

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"Обучающая выборка: {X_train.shape[0]} строк")
print(f"Тестовая выборка: {X_test.shape[0]} строк")

# ======================== 3. Обучение SVM с разными ядрами ====================
# (По методичке: линейное, RBF, полиномиальное, сигмоидное)
print("\n--- Шаг 3: Обучение SVM с различными ядрами ---")

C = 1.0  # параметр регуляризации SVM

# Создаём экземпляр SVM и обучаем модель с использованием линейного ядра
linear_svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)

# Создаём экземпляр SVM и обучаем модель с использованием RBF-ядра
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)

# Создаём экземпляр SVM и обучаем модель с использованием полиномиального ядра
poly_svc = svm.SVC(kernel='poly', degree=2, C=C).fit(X_train, y_train)

# Создаём экземпляр SVM и обучаем модель с использованием сигмоидного ядра
sig_svc = svm.SVC(kernel='sigmoid', C=C).fit(X_train, y_train)

# оцениваем качество моделей
print('Accuracy of linear kernel:', accuracy_score(y_test, linear_svc.predict(X_test)))
print('Accuracy of polynomial kernel:', accuracy_score(y_test, poly_svc.predict(X_test)))
print('Accuracy of RBF kernel:', accuracy_score(y_test, rbf_svc.predict(X_test)))
print('Accuracy of sigmoid kernel:', accuracy_score(y_test, sig_svc.predict(X_test)))

# ======================== 4. Детальный отчёт лучшей модели ====================
print("\n--- Шаг 4: Детальный отчёт (линейное ядро) ---")
y_pred = linear_svc.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Отрицательный', 'Положительный']))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ======================== 5. Визуализация результатов =========================
print("\n--- Шаг 5: Визуализация ---")

# 5a. Сравнение точности ядер
kernels = ['Linear', 'RBF', 'Polynomial', 'Sigmoid']
accuracies = [
    accuracy_score(y_test, linear_svc.predict(X_test)),
    accuracy_score(y_test, rbf_svc.predict(X_test)),
    accuracy_score(y_test, poly_svc.predict(X_test)),
    accuracy_score(y_test, sig_svc.predict(X_test))
]

plt.figure(figsize=(10, 5))
bars = plt.bar(kernels, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
plt.ylabel('Accuracy')
plt.title('Сравнение точности SVM с различными ядрами')
plt.ylim(0, 1.1)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.savefig('week13_svm_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сохранён: week13_svm_accuracy.png")

# 5b. Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.set(xticks=[0, 1], yticks=[0, 1],
       xticklabels=['Отриц.', 'Полож.'],
       yticklabels=['Отриц.', 'Полож.'],
       ylabel='Истинное значение',
       xlabel='Предсказанное значение',
       title='Матрица ошибок (SVM, линейное ядро)')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center', fontsize=16, fontweight='bold',
                color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.colorbar(im)
plt.tight_layout()
plt.savefig('week13_svm_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сохранён: week13_svm_confusion_matrix.png")

# ======================== 6. Визуализация границ решений ======================
# (По методичке: используем make_classification для 2D визуализации)
print("\n--- Шаг 6: Визуализация границ решений (2D пример из методички) ---")

# Генерируем данные для обучения
X2d, y2d = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                n_informative=2, random_state=1, 
                                n_clusters_per_class=1)

C = 1.0
linear_svc_2d = svm.SVC(kernel='linear', C=C).fit(X2d, y2d)
rbf_svc_2d = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X2d, y2d)
poly_svc_2d = svm.SVC(kernel='poly', degree=2, C=C).fit(X2d, y2d)
sig_svc_2d = svm.SVC(kernel='sigmoid', C=C).fit(X2d, y2d)

# Создаём сетку для графиков
h = 0.02  # шаг в сетке
x_min, x_max = X2d[:, 0].min() - 0.5, X2d[:, 0].max() + 0.5
y_min, y_max = X2d[:, 1].min() - 0.5, X2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Заголовки графиков
titles = ['SVM with linear kernel',
          'SVM with RBF kernel',
          'SVM with polynomial (degree 2) kernel',
          'SVM with sigmoid kernel']

# Создаём график
plt.figure(figsize=(12, 8))
for i, clf in enumerate((linear_svc_2d, rbf_svc_2d, poly_svc_2d, sig_svc_2d)):
    # Рисуем границы принятия решений на графике
    plt.subplot(2, 2, i+1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Set1, alpha=0.8)
    # Рисуем точки данных
    plt.scatter(X2d[:, 0], X2d[:, 1], c=y2d, cmap=plt.cm.Set1, edgecolor='k')
    plt.title(titles[i])

plt.tight_layout()
plt.savefig('week13_svm_decision_boundaries.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сохранён: week13_svm_decision_boundaries.png")

print("\n" + "=" * 60)
print("ЗАДАНИЕ 2 ВЫПОЛНЕНО УСПЕШНО!")
print("=" * 60)

