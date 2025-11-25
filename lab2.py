import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. Згенерувати випадковий набір даних (1000 значень)
# ==========================================
np.random.seed(42)  # Для відтворюваності результатів

# Генеруємо X (ознаки): 1000 точок від 0 до 10
X = np.sort(10 * np.random.rand(1000, 1), axis=0)

# Генеруємо y (цільова змінна): функція sin(x) + випадковий шум
y = np.sin(X).ravel() + np.random.normal(0, 0.2, 1000)

print(f"Згенеровано даних: {X.shape[0]} точок.")

# ==========================================
# 2. Нормалізувати значення
# ==========================================
# Для KNN нормалізація критична, оскільки метод базується на відстанях
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. Розділити на навчальну і тестову вибірки
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Навчальна вибірка: {X_train.shape[0]}, Тестова вибірка: {X_test.shape[0]}")

# ==========================================
# 4 & 5. Навчити KNN з різними K та вибрати найкраще
# ==========================================
k_values = range(1, 30)  # Перевіримо K від 1 до 30
mse_errors = []
r2_scores = []

for k in k_values:
    # Створення та навчання моделі
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Передбачення на тестовій вибірці
    y_pred = knn.predict(X_test)

    # Оцінка якості (MSE - чим менше, тим краще)
    mse = mean_squared_error(y_test, y_pred)
    mse_errors.append(mse)
    r2_scores.append(r2_score(y_test, y_pred))

# Знаходимо K з мінімальною помилкою
best_k_index = np.argmin(mse_errors)
best_k = k_values[best_k_index]
min_mse = mse_errors[best_k_index]

print(f"\nНайкраще значення K: {best_k}")
print(f"Мінімальна помилка MSE: {min_mse:.4f}")
print(f"R2 Score для найкращого K: {r2_scores[best_k_index]:.4f}")

# ==========================================
# 6. Візуалізація отриманих рішень
# ==========================================
plt.figure(figsize=(14, 6))

# Графік 1: Залежність помилки від K
plt.subplot(1, 2, 1)
plt.plot(k_values, mse_errors, marker='o', linestyle='--', color='blue')
plt.scatter(best_k, min_mse, color='red', s=100, label=f'Best K={best_k}')
plt.title('Залежність помилки MSE від кількості сусідів (K)')
plt.xlabel('Значення K')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)

# Графік 2: Результат регресії для найкращого K
plt.subplot(1, 2, 2)

# Навчимо фінальну модель на найкращому K
final_knn = KNeighborsRegressor(n_neighbors=best_k)
final_knn.fit(X_train, y_train)

# Створимо лінію для графіку (весь діапазон даних)
X_range = np.linspace(X_scaled.min(), X_scaled.max(), 500).reshape(-1, 1)
y_range_pred = final_knn.predict(X_range)

# Відобразимо тестові дані та лінію регресії
plt.scatter(X_test, y_test, color='gray', s=10, alpha=0.5, label='Тестові дані')
plt.plot(X_range, y_range_pred, color='green', linewidth=2, label=f'KNN Регресія (K={best_k})')
plt.title(f'Результат регресії (K={best_k})')
plt.xlabel('Нормалізоване значення X')
plt.ylabel('Значення Y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
