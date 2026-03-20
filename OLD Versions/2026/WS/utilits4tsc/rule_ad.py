import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

class QuantileAD(BaseEstimator, RegressorMixin):
    """
    Обнаружение аномалий на основе квантилей.
    Если значение выходит за пределы [low, high] квантилей, оно считается аномалией.
    """
    def __init__(self, low=None, high=None):
        self.low = low
        self.high = high

    def fit(self, X, y):
        # Проверка входных данных. X может быть пустым (dummy), y должен быть 1D массивом.
        # allow_nd=True позволяет передавать X как 2D массив, даже если мы используем только y.
        X, y = check_X_y(X, y, y_numeric=True, ensure_2d=True)
        
        y_arr = np.asarray(y)
        
        # Вычисляем границы
        self.abs_low_ = np.quantile(y_arr, self.low) if self.low is not None else -np.inf
        self.abs_high_ = np.quantile(y_arr, self.high) if self.high is not None else np.inf
        
        # Сохраняем количество признаков для проверки в predict
        self.n_features_in_ = X.shape[1]
        
        return self

    def predict(self, X):
        # Проверяем, был ли вызван fit
        check_is_fitted(self)
        # Проверяем входные данные
        X = check_array(X, ensure_2d=True)
        
        # Проверка на соответствие размерности (если обучались на X с n фичами, ожидаем n)
        if X.shape[1] != self.n_features_in_:
             raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        # Используем первый столбец как значение ряда (lag1)
        y_arr = X[:, 0]
        return np.where((y_arr > self.abs_high_) | (y_arr < self.abs_low_), -1.0, 1.0)


class InterQuartileRangeAD(BaseEstimator, RegressorMixin):
    """
    Обнаружение аномалий на основе межквартильного размаха (IQR).
    """
    def __init__(self, c=3.0):
        self.c = c

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        y_arr = np.asarray(y)
        
        q1, q3 = np.percentile(y_arr, [25, 75])
        iqr = q3 - q1
        
        # Обработка параметра c (может быть числом или кортежем)
        if isinstance(self.c, (tuple, list)):
            c_low, c_high = self.c
        else:
            c_low, c_high = self.c, self.c
            
        self.abs_low_ = q1 - iqr * c_low
        self.abs_high_ = q3 + iqr * c_high
        self.n_features_in_ = X.shape[1]
        
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        if X.shape[1] != self.n_features_in_:
             raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        y_arr = X[:, 0]  # lag1
        return np.where((y_arr > self.abs_high_) | (y_arr < self.abs_low_), -1.0, 1.0)


class PersistAD(BaseEstimator, RegressorMixin):
    """
    Обнаружение аномалий на основе сохранения разностей (Persistence).
    Аномалия детектируется, если разница между соседними точками аномальна.
    """
    def __init__(self, c=3.0, side='both'):
        self.c = c
        self.side = side

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        y_arr = np.asarray(y)
        
        # Разности целевого ряда
        diff = np.diff(y_arr)
        
        # Защита от случая, когда ряд слишком короткий
        if len(diff) == 0:
            raise ValueError("y must have at least 2 elements to compute differences.")
            
        q1, q3 = np.percentile(diff, [25, 75])
        iqr = q3 - q1
        
        self.abs_low_ = q1 - iqr * self.c
        self.abs_high_ = q3 + iqr * self.c
        self.n_features_in_ = X.shape[1]
        
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        # X ожидается содержащим как минимум 2 лага (lag1, lag2)
        if X.shape[1] < 2:
            raise ValueError("X must contain at least 2 columns (lag1 and lag2) for PersistAD.")
            
        diff = X[:, 0] - X[:, 1]  # lag1 - lag2
        
        if self.side == 'positive':
            mask = diff > self.abs_high_
        elif self.side == 'negative':
            mask = diff < self.abs_low_
        else: # 'both'
            mask = (diff > self.abs_high_) | (diff < self.abs_low_)
            
        return np.where(mask, -1.0, 1.0)


class SeasonalAD(BaseEstimator, RegressorMixin):
    """
    Обнаружение аномалий на основе сезонной декомпозиции.
    """
    def __init__(self, freq=52, c=3.0, side='both'):
        self.freq = freq
        self.c = c
        self.side = side

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        
        # Импорт внутри метода, чтобы библиотека не требовалась, если класс не используется
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
        except ImportError:
            raise ImportError("statsmodels is required for SeasonalAD. Please install it.")

        y_arr = pd.Series(np.asarray(y))
        
        # extrapolate_trend='freq' требует достаточно длинного ряда
        result = seasonal_decompose(y_arr, model='additive', period=self.freq, extrapolate_trend='freq')
        
        # Сохраняем сезонную компоненту. Важно: длина равна len(y)
        # Для predict мы будем использовать её циклически
        self.seasonal_ = result.seasonal.values
        
        residual = result.resid.dropna().abs()
        q1, q3 = np.percentile(residual, [25, 75])
        self.abs_high_ = q3 + (q3 - q1) * self.c
        
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        y_approx = X[:, 0]  # lag1
        
        # Циклическое повторение сезонной компоненты
        # Используем модуль для зацикливания паттерна
        # Внимание: это подразумевает, что predict начинается с той же фазы, что и fit,
        # либо фаза сбивается. В реальных задачах нужно передавать индекс времени.
        # Здесь оставлена логика оригинала с корректной индексацией.
        seasonal_idx = np.arange(len(y_approx)) % len(self.seasonal_)
        seasonal = self.seasonal_[seasonal_idx]
        
        residual = y_approx - seasonal
        
        if self.side == 'positive':
            mask = residual > self.abs_high_
        elif self.side == 'negative':
            mask = residual < -self.abs_high_
        else:
            mask = np.abs(residual) > self.abs_high_
            
        return np.where(mask, -1.0, 1.0)

class StagnationAD(BaseEstimator, RegressorMixin):
    """
    Аномалия, если значение остается неизменным (или почти неизменным) 
    в пределах окна (lags).
    """
    def __init__(self, tolerance=0.0):
        self.tolerance = tolerance

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        # Проверяем диапазон значений в каждой строке (по всем лагам)
        # Если max - min <= tolerance, значит ряд стоит на месте
        row_ranges = np.ptp(X, axis=1) # ptp = peak to peak (max - min)
        
        # Если range равен 0 (или меньше допуска), это стагнация -> аномалия
        anomalies = (row_ranges <= self.tolerance)
        
        return np.where(anomalies, -1.0, 1.0)

class DiffThresholdAD(BaseEstimator, RegressorMixin):
    """
    Аномалия, если разница между соседними точками (lag1 - lag2) 
    выходит за пределы [min_diff, max_diff].
    """
    def __init__(self, min_diff=None, max_diff=None):
        self.min_diff = min_diff
        self.max_diff = max_diff

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        if X.shape[1] < 2:
            raise ValueError("DiffThresholdAD требует как минимум 2 лага (window_length >= 2)")
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        # lag1 - lag2
        diff = X[:, 0] - X[:, 1]
        
        anomalies = np.zeros(len(diff), dtype=bool)
        if self.min_diff is not None:
            anomalies |= (diff < self.min_diff)
        if self.max_diff is not None:
            anomalies |= (diff > self.max_diff)
            
        return np.where(anomalies, -1.0, 1.0)
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
class ThresholdAD(BaseEstimator, RegressorMixin):
    """
    Аномалия, если значение выходит за пределы [min_val, max_val].
    Если граница не указана (None), она игнорируется.
    """
    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X, y):
        # Нам не нужно обучаться, но мы должны проверить вход и сохранить атрибуты
        X, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        
        if self.min_val is None and self.max_val is None:
            raise ValueError("Должен быть задан хотя бы один из порогов: min_val или max_val")
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        # Берем последнее значение (lag1)
        values = X[:, 0]
        
        # Проверяем условия
        anomalies = np.zeros(len(values), dtype=bool)
        if self.min_val is not None:
            anomalies |= (values < self.min_val)
        if self.max_val is not None:
            anomalies |= (values > self.max_val)
            
        return np.where(anomalies, -1.0, 1.0)
class CustomRuleAD(BaseEstimator, RegressorMixin):
    """
    Универсальный детектор, принимающий функцию-предикат.
    Функция должна принимать массив X (n_samples, n_lags) 
    и возвращать булев массив (True если аномалия).
    """
    def __init__(self, rule_func):
        self.rule_func = rule_func

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        if not callable(self.rule_func):
            raise ValueError("rule_func должен быть вызываемой функцией")
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        # Применяем пользовательскую функцию
        anomalies = self.rule_func(X)
        
        return np.where(anomalies, -1.0, 1.0)