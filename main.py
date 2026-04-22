import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Union

sns.set(style="whitegrid")


def create_vector() -> np.ndarray:
    """
    Создать массив от 0 до 9.

    Returns:
        np.ndarray: Массив чисел от 0 до 9 включительно.
    """
    return np.arange(10)


def create_matrix() -> np.ndarray:
    """
    Создать матрицу 5x5 со случайными числами [0,1].

    Returns:
        np.ndarray: Матрица 5x5 со случайными значениями от 0 до 1.
    """
    return np.random.rand(5, 5)


def reshape_vector(vec: np.ndarray) -> np.ndarray:
    """
    Преобразовать вектор формы (10,) в матрицу формы (2, 5).

    Args:
        vec (np.ndarray): Входной массив формы (10,).

    Returns:
        np.ndarray: Преобразованный массив формы (2, 5).
    """
    return vec.reshape(2, 5)


def transpose_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Транспонирование матрицы.

    Args:
        mat (np.ndarray): Входная матрица.

    Returns:
        np.ndarray: Транспонированная матрица.
    """
    return mat.T


def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Сложение векторов одинаковой длины (векторизация без циклов).

    Args:
        a (np.ndarray): Первый вектор.
        b (np.ndarray): Второй вектор.

    Returns:
        np.ndarray: Результат поэлементного сложения.
    """
    return a + b


def scalar_multiply(vec: np.ndarray, scalar: Union[int, float]) -> np.ndarray:
    """
    Умножение вектора на число.

    Args:
        vec (np.ndarray): Входной вектор.
        scalar (Union[int, float]): Число для умножения.

    Returns:
        np.ndarray: Результат умножения вектора на скаляр.
    """
    return vec * scalar


def elementwise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Поэлементное умножение двух массивов.

    Args:
        a (np.ndarray): Первый вектор/матрица.
        b (np.ndarray): Второй вектор/матрица.

    Returns:
        np.ndarray: Результат поэлементного умножения.
    """
    return a * b


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Скалярное произведение двух векторов.

    Args:
        a (np.ndarray): Первый вектор.
        b (np.ndarray): Второй вектор.

    Returns:
        float: Скалярное произведение векторов.
    """
    return np.dot(a, b)


def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Умножение матриц.

    Args:
        a (np.ndarray): Первая матрица.
        b (np.ndarray): Вторая матрица.

    Returns:
        np.ndarray: Результат умножения матриц.
    """
    return np.matmul(a, b)


def matrix_determinant(a: np.ndarray) -> float:
    """
    Вычисление определителя квадратной матрицы.

    Args:
        a (np.ndarray): Квадратная матрица.

    Returns:
        float: Определитель матрицы.
    """
    return np.linalg.det(a)


def matrix_inverse(a: np.ndarray) -> np.ndarray:
    """
    Вычисление обратной матрицы.

    Args:
        a (np.ndarray): Квадратная невырожденная матрица.

    Returns:
        np.ndarray: Обратная матрица.
    """
    return np.linalg.inv(a)


def solve_linear_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решение системы линейных уравнений Ax = b.

    Args:
        a (np.ndarray): Матрица коэффициентов A.
        b (np.ndarray): Вектор свободных членов b.

    Returns:
        np.ndarray: Решение системы x.
    """
    return np.linalg.solve(a, b)


def load_dataset(path: str = "data/students_scores.csv") -> np.ndarray:
    """
    Загрузить CSV файл и вернуть данные в виде NumPy массива.

    Args:
        path (str): Путь к CSV файлу.

    Returns:
        np.ndarray: Загруженные данные в виде массива.
    """
    df = pd.read_csv(path)
    return df.to_numpy()


def statistical_analysis(data: np.ndarray) -> Dict[str, float]:
    """
    Статистический анализ одномерного массива данных.

    Вычисляет: средний балл, медиану, стандартное отклонение,
    минимум, максимум, 25-й и 75-й перцентили.

    Args:
        data (np.ndarray): Одномерный массив данных.

    Returns:
        Dict[str, float]: Словарь со статистическими показателями.
    """
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "percentile_25": np.percentile(data, 25),
        "percentile_75": np.percentile(data, 75),
    }


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Min-Max нормализация данных к диапазону [0, 1].

    Формула: (x - min) / (max - min)

    Args:
        data (np.ndarray): Входной массив данных.

    Returns:
        np.ndarray: Нормализованный массив данных.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)


def plot_histogram(data: np.ndarray) -> None:
    """
    Построить гистограмму распределения данных и сохранить в файл.

    Args:
        data (np.ndarray): Данные для гистограммы.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=10, color='skyblue', edgecolor='black')
    plt.title("Distribution of Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/histogram.png")
    plt.close()


def plot_heatmap(matrix: np.ndarray) -> None:
    """
    Построить тепловую карту корреляции и сохранить в файл.

    Args:
        matrix (np.ndarray): Матрица корреляции.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/heatmap.png")
    plt.close()


def plot_line(x: np.ndarray, y: np.ndarray) -> None:
    """
    Построить линейный график зависимости и сохранить в файл.

    Args:
        x (np.ndarray): Значения по оси X (например, номера студентов).
        y (np.ndarray): Значения по оси Y (например, оценки).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title("Student Scores Trend")
    plt.xlabel("Student ID")
    plt.ylabel("Math Score")
    plt.xticks(x)
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/line_plot.png")
    plt.close()


if __name__ == "__main__":
    print("Running local checks...")
    v = create_vector()
    print(f"Vector created: {v}")
    
    data = load_dataset()
    stats = statistical_analysis(data[:, 0])
    print(f"Stats: {stats}")
    
    norm = normalize_data(data[:, 0])
    print(f"Normalized first 5: {norm[:5]}")
    
    plot_histogram(data[:, 0])
    print("Plots saved to 'plots/' directory.")
