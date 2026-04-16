# ==============================================================================
# Лабораторные работы №9-10: Нейронная сеть
# Задание 3: Прогнозирование вероятности положительного/отрицательного отзыва
# Датасет: Womens Clothing E-Commerce Reviews (весь датасет)
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ======================== Сигмоидная функция активации ========================
# (По методичке: f(x) = 1 / (1 + e^(-x)))
def sigmoid(x):
    # Сигмоидная функция активации: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def deriv_sigmoid(x):
    # Производная сигмоиды: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    # y_true и y_pred – массивы numpy одинаковой длины.
    return ((y_true - y_pred) ** 2).mean()

# ======================== Класс нейронной сети ================================
class OurNeuralNetwork:
    '''
    Нейронная сеть с:
    - N входами (определяется из данных)
    - скрытым слоем с 2 нейронами (h1, h2)
    - выходным слоем с 1 нейроном (o1)
    *** DISCLAIMER ***:
    Следующий код простой и обучающий, но НЕ оптимальный.
    Код реальных нейронных сетей совсем на него не похож. НЕ копируйте его!
    Изучайте и запускайте его, чтобы понять, как работает эта нейронная сеть.
    '''
    def __init__(self, n_features):
        # Веса - инициализируем случайным образом
        np.random.seed(42)
        self.w_hidden1 = np.random.normal(size=n_features) * 0.1
        self.w_hidden2 = np.random.normal(size=n_features) * 0.1
        self.w_output = np.random.normal(size=2) * 0.1
        # Пороги
        self.b1 = np.random.normal() * 0.1
        self.b2 = np.random.normal() * 0.1
        self.b3 = np.random.normal() * 0.1

    def feedforward(self, x):
        # Прямой проход по сети
        h1 = sigmoid(np.dot(x, self.w_hidden1) + self.b1)
        h2 = sigmoid(np.dot(x, self.w_hidden2) + self.b2)
        o1 = sigmoid(self.w_output[0] * h1 + self.w_output[1] * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues, epochs=500, learn_rate=0.1):
        '''
        - data – массив numpy (n x features)
        - all_y_trues – массив numpy с n элементами.
        '''
        losses = []

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Прямой проход (эти значения нам понадобятся позже)
                sum_h1 = np.dot(x, self.w_hidden1) + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = np.dot(x, self.w_hidden2) + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w_output[0] * h1 + self.w_output[1] * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- Считаем частные производные.
                # --- Имена: d_L_d_w1 = "частная производная L по w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Нейрон o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)
                d_ypred_d_h1 = self.w_output[0] * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w_output[1] * deriv_sigmoid(sum_o1)

                # Нейрон h1
                d_h1_d_w = x * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Нейрон h2
                d_h2_d_w = x * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # --- Обновляем веса и пороги
                # Нейрон h1
                self.w_hidden1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Нейрон h2
                self.w_hidden2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Нейрон o1
                self.w_output[0] -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w_output[1] -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # --- Считаем полные потери в конце каждой эпохи
            if epoch % 50 == 0:
                y_preds = np.array([self.feedforward(x) for x in data])
                loss = mse_loss(all_y_trues, y_preds)
                losses.append(loss)
                print("Epoch %d loss: %.4f" % (epoch, loss))

        return losses

# ======================== 1. Загрузка данных ==================================
print("=" * 60)
print("ЗАДАНИЕ 3: Нейронная сеть (прогнозирование отзывов)")
print("=" * 60)

df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv', index_col=0)
df = df.dropna(subset=['Review Text'])
print(f"\nРазмер датасета: {df.shape[0]} строк")

# Бинарная метка: положительный (Rating >= 4) = 1, отрицательный = 0
df['Sentiment'] = (df['Rating'] >= 4).astype(int)
print(f"Положительных: {df['Sentiment'].sum()}, Отрицательных: {(1-df['Sentiment']).sum()}")

# ======================== 2. Подготовка признаков =============================
print("\n--- Шаг 2: Подготовка числовых признаков ---")
# Используем числовые признаки для нейросети (как в методичке)
# ВАЖНО: НЕ включаем 'Rating' — он является основой для метки Sentiment!
# Это избегает утечки данных (data leakage)
features = ['Recommended IND', 'Positive Feedback Count', 'Age']
X_full = df[features].fillna(0).values

# Нормализация (сдвигаем данные чтобы было проще использовать)
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)
y_full = df['Sentiment'].values

# Берём подвыборку для обучения нейросети (полный датасет слишком медленный для нашей сети)
sample_size = 2000
np.random.seed(42)
indices = np.random.choice(len(X_full_scaled), sample_size, replace=False)
X_sample = X_full_scaled[indices]
y_sample = y_full[indices]

X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.3, random_state=42
)
print(f"Обучающая выборка: {X_train.shape[0]}")
print(f"Тестовая выборка: {X_test.shape[0]}")

# ======================== 3. Обучение нейронной сети ===========================
print("\n--- Шаг 3: Обучение нейронной сети ---")
print("Архитектура: 3 входа (Age, Recommended, Feedback) → 2 скрытых нейрона → 1 выход")
print("Функция активации: сигмоида")
print("Функция потерь: MSE (средняя квадратичная ошибка)")
print("Оптимизатор: SGD (стохастический градиентный спуск)")
print()

network = OurNeuralNetwork(n_features=X_train.shape[1])
losses = network.train(X_train, y_train, epochs=300, learn_rate=0.05)

# ======================== 4. Оценка модели ====================================
print("\n--- Шаг 4: Оценка модели ---")
y_preds_prob = np.array([network.feedforward(x) for x in X_test])
y_preds = (y_preds_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_preds)
print(f"\nТочность на тестовой выборке: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_preds, target_names=['Отрицательный', 'Положительный']))

# ======================== 5. Примеры предсказаний =============================
print("\n--- Шаг 5: Примеры предсказаний ---")
for i in range(5):
    prob = network.feedforward(X_test[i])
    actual = "Положительный" if y_test[i] == 1 else "Отрицательный"
    predicted = "Положительный" if prob >= 0.5 else "Отрицательный"
    print(f"  Клиент {i+1}: Вероятность = {prob:.3f}, "
          f"Предсказано: {predicted}, Истинно: {actual}")

# ======================== 6. Визуализация потерь ==============================
print("\n--- Шаг 6: Визуализация ---")

plt.figure(figsize=(10, 5))
plt.plot(range(0, 300, 50), losses, 'b-o', linewidth=2)
plt.xlabel('Эпоха')
plt.ylabel('MSE Loss (потери)')
plt.title('Обучение нейронной сети: уменьшение потерь')
plt.grid(True, alpha=0.3)
plt.savefig('week9_neural_network_loss.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сохранён: week9_neural_network_loss.png")

# Распределение вероятностей
plt.figure(figsize=(10, 5))
plt.hist(y_preds_prob[y_test == 1], bins=20, alpha=0.7, label='Положительные', color='#4ECDC4')
plt.hist(y_preds_prob[y_test == 0], bins=20, alpha=0.7, label='Отрицательные', color='#FF6B6B')
plt.xlabel('Предсказанная вероятность')
plt.ylabel('Количество')
plt.title('Распределение предсказанных вероятностей')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('week9_probability_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сохранён: week9_probability_distribution.png")

# Confusion Matrix
cm = confusion_matrix(y_test, y_preds)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.set(xticks=[0, 1], yticks=[0, 1],
       xticklabels=['Отриц.', 'Полож.'],
       yticklabels=['Отриц.', 'Полож.'],
       ylabel='Истинное значение',
       xlabel='Предсказанное значение',
       title='Матрица ошибок (Нейронная сеть)')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center', fontsize=16, fontweight='bold',
                color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.colorbar(im)
plt.tight_layout()
plt.savefig('week9_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сохранён: week9_confusion_matrix.png")

print("\n" + "=" * 60)
print("ЗАДАНИЕ 3 ВЫПОЛНЕНО УСПЕШНО!")
print("=" * 60)

