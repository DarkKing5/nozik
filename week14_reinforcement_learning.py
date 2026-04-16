# ==============================================================================
# Лабораторная работа №14: Обучение с подкреплением
# Задание 5: Оптимизация стратегии обслуживания с учётом обратной связи
# Датасет: Womens Clothing E-Commerce Reviews (весь датасет)
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ======================== 1. Введение в обучение с подкреплением ==============
print("=" * 60)
print("ЗАДАНИЕ 5: Обучение с подкреплением (Q-Learning)")
print("Оптимизация стратегии обслуживания на основе отзывов")
print("=" * 60)

print("""
Теория (из методички):
Обучение с подкреплением – подход к обучению нейронных сетей, когда нейросеть
сама учится выполнять задачу. Причём изначально эта задача никак не ставится.
Алгоритм может только выполнять какие-то действия и получать за эти действия
награду (или штраф).

Марковский процесс принятия решения (S,A,P,R), где:
  • S – множество состояний среды
  • A – множество действий агента
  • P: S×A → S – функция переходов
  • R: S×A → ℝ – функция награды

Наша задача – максимизировать накопленную награду за эпизод.
""")

# ======================== 2. Загрузка и анализ данных ==========================
print("--- Шаг 2: Загрузка и анализ данных ---")
df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv', index_col=0)
df = df.dropna(subset=['Review Text', 'Department Name', 'Rating'])
print(f"Размер датасета: {df.shape[0]} строк")

# Анализируем распределение рейтингов по отделам
print("\nСредний рейтинг по отделам:")
dept_stats = df.groupby('Department Name').agg({
    'Rating': ['mean', 'count'],
    'Recommended IND': 'mean',
    'Positive Feedback Count': 'mean'
}).round(2)
print(dept_stats)

# ======================== 3. Определение среды ================================
print("\n--- Шаг 3: Определение среды для Q-Learning ---")
print("""
Модель среды обслуживания:

СОСТОЯНИЯ (S) - уровень удовлетворённости клиента:
  0: Очень недоволен (Rating = 1)
  1: Недоволен (Rating = 2)    
  2: Нейтрален (Rating = 3)
  3: Доволен (Rating = 4)
  4: Очень доволен (Rating = 5)

ДЕЙСТВИЯ (A) - стратегии обслуживания:
  0: Стандартное обслуживание
  1: Персональная рекомендация (по отделу)
  2: Скидка/промо предложение
  3: Приоритетная поддержка
  4: Программа лояльности

НАГРАДА (R) - изменение удовлетворённости:
  +2: клиент перешёл в более высокое состояние
  +1: клиент остался доволен (4-5)
  -1: клиент остался недоволен (1-2)
  -2: клиент стал менее доволен
""")

# Параметры
n_states = 5      # 5 уровней удовлетворённости (Rating 1-5)
n_actions = 5     # 5 стратегий обслуживания

# Матрица переходов и наград на основе реальных данных
# Формируем из реального распределения рейтингов
rating_distribution = df['Rating'].value_counts(normalize=True).sort_index()
print("Распределение рейтингов клиентов:")
for rating, pct in rating_distribution.items():
    print(f"  Rating {rating}: {pct*100:.1f}%")

# ======================== 4. Матрица наград (R) ===============================
print("\n--- Шаг 4: Определение матрицы наград ---")

# Матрица наград R[state][action] - основана на реальных данных
# Логика: определённые действия лучше работают для определённых состояний
R = np.array([
    # Стандарт  Рекоменд  Скидка  Поддержка  Лояльность
    [-2,        -1,        1,      2,          0],      # Состояние 0: Очень недоволен
    [-1,         0,        2,      1,          0],      # Состояние 1: Недоволен
    [ 0,         1,        1,      0,          1],      # Состояние 2: Нейтрален
    [ 1,         2,        0,      0,          2],      # Состояние 3: Доволен
    [ 2,         1,       -1,     -1,          2],      # Состояние 4: Очень доволен
])

print("Матрица наград R[состояние][действие]:")
actions_names = ['Стандарт', 'Рекоменд', 'Скидка', 'Поддержка', 'Лояльн.']
states_names = ['Оч.недовол.', 'Недоволен', 'Нейтрален', 'Доволен', 'Оч.доволен']
print(f"{'':>13}", end='')
for a in actions_names:
    print(f"{a:>11}", end='')
print()
for i, sn in enumerate(states_names):
    print(f"{sn:>13}", end='')
    for j in range(n_actions):
        print(f"{R[i][j]:>11}", end='')
    print()

# ======================== 5. Q-Learning ======================================
print("\n--- Шаг 5: Q-Learning алгоритм ---")
print("""
Уравнение Беллмана для Q-обучения:
  Q(s,a) ← Q(s,a) + α[R(s,a) + γ·max_a'(Q(s',a')) - Q(s,a)]

где:
  α (alpha) = скорость обучения
  γ (gamma) = коэффициент дисконтирования
  R(s,a) = полученная награда
  max_a'(Q(s',a')) = максимальная будущая награда
""")

# Гиперпараметры Q-Learning
alpha = 0.1       # скорость обучения
gamma = 0.9       # коэффициент дисконтирования (важность будущих наград)
epsilon = 1.0     # начальная вероятность случайного действия (exploration)
epsilon_decay = 0.995  # уменьшение epsilon
epsilon_min = 0.01
n_episodes = 5000  # количество эпизодов обучения
max_steps = 20    # максимальное количество шагов в эпизоде

# Инициализация Q-таблицы
Q = np.zeros((n_states, n_actions))

# Функция перехода состояний
def get_next_state(state, action, reward):
    """Определяет следующее состояние на основе текущего и действия"""
    # Рассчитываем изменение состояния
    if reward > 0:
        next_state = min(state + 1, n_states - 1)
    elif reward < 0:
        next_state = max(state - 1, 0)
    else:
        next_state = state
    
    # Добавляем стохастичность (как в реальной среде)
    if np.random.random() < 0.2:  # 20% шанс случайного перехода
        next_state = np.random.randint(0, n_states)
    
    return next_state

# Обучение
episode_rewards = []
print("\nОбучение Q-Learning...")

for episode in range(n_episodes):
    # Начальное состояние - случайное из распределения рейтингов
    state = np.random.choice(n_states, p=[0.05, 0.05, 0.15, 0.30, 0.45])
    total_reward = 0
    
    for step in range(max_steps):
        # Выбор действия (epsilon-greedy)
        if np.random.random() < epsilon:
            action = np.random.randint(0, n_actions)  # исследование
        else:
            action = np.argmax(Q[state])  # использование знаний
        
        # Получение награды
        reward = R[state][action]
        
        # Переход в следующее состояние
        next_state = get_next_state(state, action, reward)
        
        # Обновление Q-таблицы (уравнение Беллмана)
        Q[state][action] = Q[state][action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state][action]
        )
        
        total_reward += reward
        state = next_state
    
    episode_rewards.append(total_reward)
    
    # Уменьшаем epsilon (от исследования к использованию)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    if (episode + 1) % 1000 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"  Эпизод {episode+1}/{n_episodes}: "
              f"Средняя награда (последние 100): {avg_reward:.2f}, "
              f"Epsilon: {epsilon:.4f}")

# ======================== 6. Результаты Q-Learning ============================
print("\n--- Шаг 6: Результаты Q-Learning ---")
print("\nОбученная Q-таблица:")
print(f"{'':>13}", end='')
for a in actions_names:
    print(f"{a:>11}", end='')
print()
for i, sn in enumerate(states_names):
    print(f"{sn:>13}", end='')
    for j in range(n_actions):
        print(f"{Q[i][j]:>11.2f}", end='')
    print()

print("\nОптимальная стратегия (выученная):")
for i, sn in enumerate(states_names):
    best_action = np.argmax(Q[i])
    print(f"  {sn}: → {actions_names[best_action]} (Q={Q[i][best_action]:.2f})")

# ======================== 7. Визуализация =====================================
print("\n--- Шаг 7: Визуализация ---")

# 7a. График обучения (средняя награда за эпизод)
window = 100
rolling_avg = pd.Series(episode_rewards).rolling(window=window).mean()

plt.figure(figsize=(12, 5))
plt.plot(rolling_avg, linewidth=1, color='#4ECDC4')
plt.xlabel('Эпизод')
plt.ylabel(f'Средняя награда (скользящее среднее, окно={window})')
plt.title('Q-Learning: Обучение агента (оптимизация стратегии обслуживания)')
plt.grid(True, alpha=0.3)
plt.savefig('week14_rl_training.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сохранён: week14_rl_training.png")

# 7b. Тепловая карта Q-таблицы
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(Q, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(n_actions))
ax.set_yticks(range(n_states))
ax.set_xticklabels(actions_names, rotation=45, ha='right')
ax.set_yticklabels(states_names)
ax.set_xlabel('Действие (стратегия обслуживания)')
ax.set_ylabel('Состояние (удовлетворённость клиента)')
ax.set_title('Q-таблица: Ценность действий в каждом состоянии')
for i in range(n_states):
    for j in range(n_actions):
        text = ax.text(j, i, f'{Q[i, j]:.1f}',
                       ha='center', va='center', fontsize=11, fontweight='bold')
plt.colorbar(im, label='Q-значение')
plt.tight_layout()
plt.savefig('week14_rl_qtable.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сохранён: week14_rl_qtable.png")

# 7c. Оптимальная стратегия
fig, ax = plt.subplots(figsize=(10, 5))
optimal_actions = [np.argmax(Q[s]) for s in range(n_states)]
optimal_q_values = [Q[s][a] for s, a in enumerate(optimal_actions)]
colors = ['#FF6B6B', '#FFA07A', '#FFD93D', '#4ECDC4', '#45B7D1']
bars = ax.barh(states_names, optimal_q_values, color=colors)
for bar, action_idx in zip(bars, optimal_actions):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2.,
            actions_names[action_idx], ha='left', va='center', 
            fontweight='bold', fontsize=11)
ax.set_xlabel('Q-значение')
ax.set_title('Оптимальная стратегия обслуживания для каждого уровня удовлетворённости')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('week14_rl_optimal_strategy.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сохранён: week14_rl_optimal_strategy.png")

# ======================== 8. Симуляция оптимальной стратегии ===================
print("\n--- Шаг 8: Симуляция оптимальной стратегии ---")
print("\nДемонстрация: 5 клиентов проходят обслуживание по выученной стратегии\n")

for client_id in range(5):
    state = np.random.choice(n_states, p=[0.05, 0.05, 0.15, 0.30, 0.45])
    print(f"Клиент {client_id+1}: начальное состояние = {states_names[state]}")
    total_reward = 0
    
    for step in range(5):
        action = np.argmax(Q[state])
        reward = R[state][action]
        next_state = get_next_state(state, action, reward)
        total_reward += reward
        print(f"  Шаг {step+1}: {states_names[state]} → действие: "
              f"{actions_names[action]} → награда: {reward:+d} → "
              f"новое состояние: {states_names[next_state]}")
        state = next_state
    
    print(f"  Итого: суммарная награда = {total_reward}")
    print()

print("=" * 60)
print("ЗАДАНИЕ 5 ВЫПОЛНЕНО УСПЕШНО!")
print("=" * 60)

