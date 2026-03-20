import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from pandas.tseries.frequencies import to_offset
from sklearn.metrics import mean_pinball_loss

import re
from typing import TYPE_CHECKING, Dict, List, Optional, Union

try:
    import matplotlib as mpl
    import matplotlib.colors as cm
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        "matplotlib is not installed. Please install it and try again.\n"
        "You can find detailed instructions at https://matplotlib.org/stable/users/installing/index.html"
    )
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import plotly

from typing import Optional, List, Dict, Union, Tuple
from utilsforecast.plotting import plot_series as uf_plot_series   
from utilsforecast.evaluation import evaluate

import re





def n_step_ahead_forecasting(model, df, h, n_windows, refit=False):
    """
    Генерирует n-step-ahead прогнозы на исторических данных (Backtesting).
    Окна идут последовательно друг за другом БЕЗ пересечений.
    
    Параметры:
    -----------
    model : object
        Инициализированная модель (StatsForecast, MLForecast, NeuralForecast).
    df : pd.DataFrame
        Данные в формате Nixtla (unique_id, ds, y).
    h : int
        Горизонт прогнозирования (длина одного прогноза).
    n_windows : int
        Количество прогнозов (окон), которые нужно сделать в прошлом.
    refit : bool
        Если False (по умолчанию) - модель обучается один раз в начале.
        Если True - модель переобучается перед каждым окном (медленно).
        
    Возвращает:
    -----------
    pd.DataFrame : Исторические прогнозы с колонкой 'step'.
    """
    
    # Ключевое условие для НЕпересекающихся последовательных окон:
    # Шаг сдвига (step_size) должен быть равен горизонту (h).
    step_size = h
    
    # Общий размер тестового периода, который будет "нарезан" на окна
    test_size = h * n_windows
    
    print(f"Генерация прогнозов: h={h}, окон={n_windows}, test_size={test_size}")

    # Запуск генерации (используем cross_validation как инструмент бэктестинга)
    forecasts_df = model.cross_validation(
        df=df,
        h=h,
        test_size=test_size,
        step_size=step_size,
        refit=refit
    )
    
    # Добавляем номер шага прогноза (1, 2, ... h)
    # ВАЖНО: группируем по unique_id И cutoff, чтобы шаги считались правильно для каждого ряда
    forecasts_df['step'] = forecasts_df.groupby(['unique_id', 'cutoff']).cumcount() + 1
    
    return forecasts_df
    
    
def fix_forecast_format(df, strip_suffixes=None):
    """
    Преобразует DataFrame: сбрасывает индекс, проверяет колонки,
    и удаляет указанные суффиксы из названий колонок моделей.

    Параметры:
    -----------
    df : pd.DataFrame
        Исходный датафрейм с прогнозами.
    strip_suffixes : list or str, optional
        Суффикс или список суффиксов, которые нужно удалить из имен колонок.
        Пример: strip_suffixes=['-median', '-mean'] или просто '-median'.
    """
    # Создаем копию, чтобы не менять исходный датафрейм
    df_fixed = df.copy()
    
    # === 1. Работа с индексами ===
    # Если unique_id является индексом, сбрасываем его
    if 'unique_id' not in df_fixed.columns and 'unique_id' in df_fixed.index.names:
        df_fixed = df_fixed.reset_index()
    
    # Сбрасываем индекс полностью, чтобы избежать дубликатов типа 'level_0'
    df_fixed = df_fixed.reset_index(drop=True)
    
    # === 2. Проверка обязательных колонок ===
    if 'ds' not in df_fixed.columns:
        raise ValueError("Колонка 'ds' отсутствует в данных после преобразования.")
    
    # Приводим ds к datetime
    df_fixed['ds'] = pd.to_datetime(df_fixed['ds'])
    
    # === 3. Удаление суффиксов (Новая логика) ===
    if strip_suffixes:
        # Если передана строка, превращаем в список
        if isinstance(strip_suffixes, str):
            strip_suffixes = [strip_suffixes]
            
        # Функция для очистки имени одной колонки
        def clean_col_name(col_name):
            # Не трогаем служебные колонки
            if col_name in ['unique_id', 'ds', 'y', 'cutoff']:
                return col_name
            
            # Удаляем суффиксы, если они есть в конце строки
            for suffix in strip_suffixes:
                if col_name.endswith(suffix):
                    # Обрезаем суффикс
                    return col_name[:-len(suffix)]
            return col_name
        
        # Применяем переименование
        df_fixed.columns = [clean_col_name(col) for col in df_fixed.columns]
        
        # Предупреждение, если появились дубликаты (например, LSTM-median и LSTM-mean стали просто LSTM)
        if df_fixed.columns.duplicated().any():
            print("Внимание! После удаления суффиксов появились дубликаты колонок.")
            # Можно оставить первые вхождения или вызвать ошибку, здесь оставляем первые
            df_fixed = df_fixed.loc[:, ~df_fixed.columns.duplicated()]

    # === 4. Упорядочивание колонок ===
    # Сначала служебные, потом модели
    cols_order = ['unique_id', 'ds', 'y', 'cutoff']
    # Добавляем остальные колонки, которых нет в списке выше
    final_cols = [c for c in cols_order if c in df_fixed.columns]
    other_cols = [c for c in df_fixed.columns if c not in final_cols]
    
    return df_fixed[final_cols + other_cols]
 

    

def evaluate_cv(
    crossvalidation_df,
    metrics,
    model_names,
    target_col='y',
    level=None,
    ts_aggregate=True,
    cutoff_aggregate=True
):
    """
    Оценка результатов кросс-валидации с гибкой агрегацией.
    
    Параметры:
    ----------
    crossvalidation_df : pd.DataFrame
        Данные в лонг-формате с колонкой 'cutoff'
    metrics : list
        Список функций метрик (например, [smape, mae, rmse])
    model_names : list
        Список названий моделей
    target_col : str, default='y'
        Название колонки с целевой переменной
    level : float or None, default=None
        Уровень для квантильных прогнозов
    ts_aggregate : bool, default=True
        Если True — агрегировать по временным рядам (уникальным unique_id)
    cutoff_aggregate : bool, default=True
        Если True — агрегировать по окнам кросс-валидации (cutoff)
    
    Возвращает:
    -----------
    pd.io.formats.style.Styler
        Стилизованный датафрейм с метриками
    """
    evaluations = []
    cutoffs = crossvalidation_df['cutoff'].unique()
    
    for c in cutoffs:
        df_cv = crossvalidation_df.query('cutoff == @c')
        evaluation = evaluate(
            df=df_cv,
            metrics=metrics,
            models=model_names,
            level=level,
            target_col=target_col
        )
        evaluation['cutoff'] = c  # сохраняем информацию об окне
        evaluations.append(evaluation)
    
    evaluations = pd.concat(evaluations, ignore_index=True)
    
    # Определяем уровни группировки
    group_cols = ['metric']
    if not ts_aggregate and 'unique_id' in evaluations.columns:
        group_cols.append('unique_id')
    if not cutoff_aggregate:
        group_cols.append('cutoff')
    
    # Агрегация
    if len(group_cols) > 0:
        evaluations = evaluations.groupby(group_cols).mean(numeric_only=True)
    else:
        evaluations = evaluations.mean(numeric_only=True).to_frame().T
    
    # Стилизация с учётом структуры индекса
    if isinstance(evaluations.index, pd.MultiIndex):
        # Для мультииндекса применяем градиент по строкам
        styled = evaluations.style.background_gradient(
            cmap='RdYlGn_r', 
            axis=1,
            vmin=evaluations.min().min(),
            vmax=evaluations.max().max()
        )
    else:
        styled = evaluations.style.background_gradient(cmap='RdYlGn_r', axis=1)
    
    return styled.format("{:.2f}")  # Форматирование до 2 знаков (согласно вашим предпочтениям)
    
def plot_cv_windows(
    df,
    cutoffs=None,
    h=7,
    input_size=None,
    step_size=1,
    test_size=None,
    freq='D',
    refit=True,
    gap=0,
    unique_id=None,  # ← НОВЫЙ ПАРАМЕТР
    title="Cross-Validation Windows",
    figsize=(12, 6),  # увеличил высоту по умолчанию
    **kwargs
):
    """
    Визуализация окон кросс-валидации с поддержкой расширяющегося и фиксированного окна.
    Обучающие окна отображаются только там, где происходит обучение (refit).
    
    Параметры:
    ----------
    df : pd.DataFrame
        Данные с колонками 'ds', 'y' и опционально 'unique_id'
    unique_id : str, optional
        Если задан и колонка 'unique_id' существует — отображается сам временной ряд справа
    ... остальные параметры без изменений ...
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
    
    # Фильтрация по unique_id если задан
    if unique_id is not None and 'unique_id' in df.columns:
        df_plot = df[df['unique_id'] == unique_id].copy()
        title = f"{title} ({unique_id})"
    else:
        df_plot = df.copy()
        unique_id = None  # чтобы не пытаться рисовать ряд
    
    df_plot = df_plot.sort_values('ds').reset_index(drop=True)
    ds_min = df_plot['ds'].min()
    ds_max = df_plot['ds'].max()
    
    # ... весь существующий код до создания фигуры без изменений ...
    
    # Нормализуем freq к строке
    if hasattr(freq, 'freqstr'):
        freq_str = freq.freqstr
    elif hasattr(freq, '_prefix'):
        prefix = freq._prefix
        freq_str = f'{freq.n}{prefix}' if hasattr(freq, 'n') and freq.n != 1 else prefix
    else:
        freq_str = str(freq)
    
    if freq_str.replace('-', '').replace('+', '').isalnum():
        freq_multiplier = 1
        freq_suffix = freq_str
    else:
        import re
        match = re.match(r'([+-]?\d+)([A-Za-z]+)', freq_str)
        if match:
            freq_multiplier = int(match.group(1))
            freq_suffix = match.group(2)
        else:
            freq_multiplier = 1
            freq_suffix = freq_str

    # Автоматический расчёт cutoffs если не заданы
    if cutoffs is None:
        if test_size is None:
            raise ValueError("Either 'cutoffs' or 'test_size' must be provided")
        
        last_cutoff = df_plot['ds'].iloc[-test_size]
        first_cutoff = df_plot['ds'].iloc[-(test_size + h - 1)]
        
        cutoffs = []
        current = first_cutoff
        offset_step = pd.tseries.frequencies.to_offset(f'{step_size * freq_multiplier}{freq_suffix}')
        while current <= last_cutoff:
            cutoffs.append(current)
            current += offset_step
    else:
        cutoffs = sorted(pd.to_datetime(cutoffs))
    
    if isinstance(h, (int, float)):
        h_offset = pd.tseries.frequencies.to_offset(f'{h * freq_multiplier}{freq_suffix}')
    else:
        h_offset = pd.to_timedelta(h)
    
    # Создаём фигуру
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Цвета
    train_color = '#1f77b4'
    test_color = '#d62728'
    cutoff_color = 'black'
    no_train_color = '#cccccc'
    series_color = '#2ca02c'  # зелёный для временного ряда
    
    # Определяем, где происходит обучение
    fit_windows = []
    if refit is True:
        fit_windows = list(range(len(cutoffs)))
    elif refit is False:
        fit_windows = [0]
    elif isinstance(refit, int) and refit > 0:
        fit_windows = list(range(0, len(cutoffs), refit))
    else:
        fit_windows = [0]
    
    # Рисуем окна
    for i, cutoff in enumerate(cutoffs):
        y_level = i + 1
        
        if input_size is None:
            train_start_full = ds_min
        else:
            offset_input = pd.tseries.frequencies.to_offset(f'{input_size * freq_multiplier}{freq_suffix}')
            train_start_full = cutoff - offset_input
            train_start_full = train_start_full if train_start_full >= ds_min else ds_min
        
        test_end = cutoff + h_offset
        test_end = min(test_end, ds_max)
        
        # Тестовое окно
        ax.plot([cutoff, test_end], [y_level, y_level],
                color=test_color, linestyle='--', linewidth=2.5, solid_capstyle='butt')
        
        # Обучающее окно
        if i in fit_windows:
            ax.plot([train_start_full, cutoff], [y_level, y_level],
                    color=train_color, linewidth=2.5, solid_capstyle='butt')
        else:
            prev_cutoff = cutoffs[i-1] if i > 0 else ds_min
            ax.plot([prev_cutoff, cutoff], [y_level, y_level],
                    color=no_train_color, linewidth=2.5, solid_capstyle='butt')
        
        ax.scatter([cutoff], [y_level], color=cutoff_color, s=40, zorder=5, marker='|')
        ax.scatter([test_end], [y_level], color=cutoff_color, s=40, zorder=5, marker='|')
    
    # Настройка основных осей
    ax.set_yticks(range(1, len(cutoffs) + 1))
    ax.set_yticklabels([f"Window {i+1}" for i in range(len(cutoffs))], fontsize=9)
    ax.set_xlabel('Date', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.grid(True, axis='x', linestyle='--', alpha=0.6, linewidth=0.8)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.tick_params(axis='x', rotation=45)
    
    # Форматирование дат
    if (ds_max - ds_min).days > 365:
        date_format = '%Y-%m'
    elif (ds_max - ds_min).days > 30:
        date_format = '%Y-%m-%d'
    else:
        date_format = '%m-%d'
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    fig.autofmt_xdate()
    
    # ДОБАВЛЕНО: Отображение временного ряда на правой оси
    if unique_id is not None:
        # Создаём вторичную ось
        ax_series = ax.twinx()
        
        # Рисуем сам временной ряд
        ax_series.plot(df_plot['ds'], df_plot['y'], 
                      color=series_color, linewidth=1.5, alpha=0.7, label='Time Series')
        
        # Настройка правой оси
        ax_series.set_ylabel('Value', color=series_color, fontsize=10, fontweight='bold')
        ax_series.tick_params(axis='y', colors=series_color)
        ax_series.spines['right'].set_color(series_color)
        
        # Добавляем легенду для ряда
        legend_elements = [
            Patch(facecolor=train_color, edgecolor='none', label='Training window (with refit)'),
            Patch(facecolor=no_train_color, edgecolor='none', label='No refit (using previous model)'),
            Patch(facecolor=test_color, edgecolor='none', label='Forecast horizon (h)'),
            plt.Line2D([0], [0], color=cutoff_color, marker='|', linestyle='None',
                       markersize=8, label='Cutoff / Forecast end'),
            plt.Line2D([0], [0], color=series_color, linewidth=1.5, 
                       label='Time Series (y)')
        ]
    else:
        legend_elements = [
            Patch(facecolor=train_color, edgecolor='none', label='Training window (with refit)'),
            Patch(facecolor=no_train_color, edgecolor='none', label='No refit (using previous model)'),
            Patch(facecolor=test_color, edgecolor='none', label='Forecast horizon (h)'),
            plt.Line2D([0], [0], color=cutoff_color, marker='|', linestyle='None',
                       markersize=8, label='Cutoff / Forecast end')
        ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    
    # Аннотация
    window_type = f"Expanding ({freq_str})" if input_size is None else f"Rolling (size={input_size}{freq_str})"
    refit_info = f", refit={'all' if refit is True else ('first' if refit is False else f'every {refit}')}"
    ax.text(0.02, 0.98, window_type + refit_info, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax.axvline(x=ds_max, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='_nolegend_')
    
    plt.tight_layout()
    return fig, ax

    
def vanilla_ensemble(
    forecasts_df: pd.DataFrame,
    aliases: list,
    levels: list = None,
    weights=None,
    agg_func: str = 'median',  # ← новое! 'median' или 'mean'
    ensemble_name: str = 'Ensemble'
) -> pd.DataFrame:
    """
    Гибкий ансамбль с поддержкой:
      - weights = None → agg_func ('median' или 'mean')
      - weights = {'ETS': 0.3, 'MSTL': 0.7} → глобальные веса
      - weights = {'uid1': {'ETS': 0.2, ...}} → локальные веса
    """
    
    result = forecasts_df[['unique_id', 'ds']].copy()
    
    # === Случай 1: без весов → agg_func ===
    if weights is None:
        point_cols = [name for name in aliases if name in forecasts_df.columns]
        if point_cols:
            agg = np.median if agg_func == 'median' else np.mean
            result[ensemble_name] = agg(forecasts_df[point_cols].values, axis=1)
        if levels:
            for level in levels:
                lo_cols = [f"{name}-lo-{level}" for name in aliases if f"{name}-lo-{level}" in forecasts_df.columns]
                hi_cols = [f"{name}-hi-{level}" for name in aliases if f"{name}-hi-{level}" in forecasts_df.columns]
                if lo_cols:
                    agg = np.median if agg_func == 'median' else np.mean
                    result[f'{ensemble_name}-lo-{level}'] = agg(forecasts_df[lo_cols].values, axis=1)
                if hi_cols:
                    agg = np.median if agg_func == 'median' else np.mean
                    result[f'{ensemble_name}-hi-{level}'] = agg(forecasts_df[hi_cols].values, axis=1)
        return result
    
    # === Случай 2: глобальные веса (dict по именам моделей) ===
    if isinstance(weights, dict) and all(isinstance(k, str) for k in weights.keys()):
        for name in aliases:
            if name not in weights:
                raise KeyError(f"Вес не задан для модели '{name}'")
        w_array = np.array([weights[name] for name in aliases])
        w_array = w_array / w_array.sum()
        
        point_cols = [name for name in aliases if name in forecasts_df.columns]
        if point_cols:
            values = forecasts_df[point_cols].values
            result[ensemble_name] = (values * w_array).sum(axis=1)
        if levels:
            for level in levels:
                lo_cols = [f"{name}-lo-{level}" for name in aliases if f"{name}-lo-{level}" in forecasts_df.columns]
                hi_cols = [f"{name}-hi-{level}" for name in aliases if f"{name}-hi-{level}" in forecasts_df.columns]
                if lo_cols:
                    values = forecasts_df[lo_cols].values
                    result[f'{ensemble_name}-lo-{level}'] = (values * w_array).sum(axis=1)
                if hi_cols:
                    values = forecasts_df[hi_cols].values
                    result[f'{ensemble_name}-hi-{level}'] = (values * w_array).sum(axis=1)
        return result
    
    # === Случай 3: локальные веса (dict of dicts) ===
    if isinstance(weights, dict):
        all_rows = []
        for uid in forecasts_df['unique_id'].unique():
            sub = forecasts_df[forecasts_df['unique_id'] == uid].copy()
            if uid not in weights:
                raise KeyError(f"Веса не заданы для unique_id='{uid}'")
            w_dict = weights[uid]
            for name in aliases:
                if name not in w_dict:
                    raise KeyError(f"Для {uid} не задан вес модели '{name}'")
            w_array = np.array([w_dict[name] for name in aliases])
            w_array = w_array / w_array.sum()
            
            point_cols = [name for name in aliases if name in sub.columns]
            if point_cols:
                values = sub[point_cols].values
                sub[ensemble_name] = (values * w_array).sum(axis=1)
            if levels:
                for level in levels:
                    lo_cols = [f"{name}-lo-{level}" for name in aliases if f"{name}-lo-{level}" in sub.columns]
                    hi_cols = [f"{name}-hi-{level}" for name in aliases if f"{name}-hi-{level}" in sub.columns]
                    if lo_cols:
                        values = sub[lo_cols].values
                        sub[f'{ensemble_name}-lo-{level}'] = (values * w_array).sum(axis=1)
                    if hi_cols:
                        values = sub[hi_cols].values
                        sub[f'{ensemble_name}-hi-{level}'] = (values * w_array).sum(axis=1)
            all_rows.append(sub)
        
        final = pd.concat(all_rows, ignore_index=True)
        cols_order = ['unique_id', 'ds', ensemble_name]
        if levels:
            cols_order += [f'{ensemble_name}-lo-{l}' for l in levels] + [f'{ensemble_name}-hi-{l}' for l in levels]
        return final[cols_order]
    
    raise TypeError("weights должен быть None, dict с именами моделей или dict of dicts")

    
def ensemble_weights_from_metrics(
    metrics_df: pd.DataFrame,
    aliases: list,
    metric_name: str = 'scaled_crps',
    per_series: bool = False,
    epsilon: float = 1e-8
):
    """
    Вычисляет веса на основе метрик из utilsforecast.evaluate.
    
    Параметры:
        metrics_df: результат evaluate(...) — широкий формат
        aliases: список имён моделей
        metric_name: имя метрики ('scaled_crps', 'mase', ...)
        per_series: если True → возвращает {'uid': {'model': weight}}, иначе {'model': weight}
    
    Возвращает:
        dict или dict of dicts — готов к передаче в flexible_ensemble(weights=...)
    """
    # Фильтруем нужную метрику
    metric_rows = metrics_df[metrics_df['metric'] == metric_name]
    if metric_rows.empty:
        raise ValueError(f"Метрика '{metric_name}' не найдена")
    
    if per_series:
        # Веса по каждому unique_id
        weights_dict = {}
        for uid in metric_rows['unique_id'].unique():
            row = metric_rows[metric_rows['unique_id'] == uid]
            scores = {name: row[name].iloc[0] for name in aliases if name in row.columns}
            if not scores:
                continue
            inv_scores = {name: 1.0 / (score + epsilon) for name, score in scores.items()}
            total = sum(inv_scores.values())
            weights_dict[uid] = {name: w / total for name, w in inv_scores.items()}
        return weights_dict
    else:
        # Глобальные веса (усреднённые по всем рядам)
        mean_scores = metric_rows.drop(columns=['metric', 'unique_id']).mean(axis=0)
        scores = {name: mean_scores[name] for name in aliases if name in mean_scores.index}
        inv_scores = {name: 1.0 / (score + epsilon) for name, score in scores.items()}
        total = sum(inv_scores.values())
        return {name: w / total for name, w in inv_scores.items()}


def quantile_ensemble_forecast(
    eval_df: pd.DataFrame,
    model_names: list,
    levels: list,
    target_col: str = 'y',
    epsilon: float = 1e-8,
    alies = 'QEnsemble'
) -> pd.DataFrame:
    """
    Строит ансамблевый прогноз в формате StatsForecast:
      - 'Ensemble'        → медиана (q=0.5)
      - 'Ensemble-lo-X'   → нижняя граница уровня X
      - 'Ensemble-hi-X'   → верхняя граница уровня X
    """
    result = eval_df[['unique_id', 'ds']].copy()
    
    # --- Точечный прогноз (медиана, q=0.5) ---
    point_cols = [name for name in model_names if name in eval_df.columns]
    if point_cols:
        # Веса по Pinball Loss на медиане
        losses = [
            mean_pinball_loss(eval_df[target_col], eval_df[col], alpha=0.5)
            for col in point_cols
        ]
        weights = np.array([1.0 / (l + epsilon) for l in losses])
        weights /= weights.sum()
        result['QEnsemble'] = sum(w * eval_df[col] for w, col in zip(weights, point_cols))
    else:
        raise ValueError("Не найдены точечные прогнозы моделей.")
    
    # --- Квантильные интервалы ---
    for level in levels:
        alpha_low = (100 - level) / 200.0   # например, level=90 → alpha=0.05
        alpha_high = 1.0 - alpha_low        # → 0.95
        
        # Нижняя граница
        lo_cols = [f"{name}-lo-{level}" for name in model_names 
                   if f"{name}-lo-{level}" in eval_df.columns]
        if lo_cols:
            losses_lo = [
                mean_pinball_loss(eval_df[target_col], eval_df[col], alpha=alpha_low)
                for col in lo_cols
            ]
            weights_lo = np.array([1.0 / (l + epsilon) for l in losses_lo])
            weights_lo /= weights_lo.sum()
            result[f'QEnsemble-lo-{level}'] = sum(w * eval_df[col] for w, col in zip(weights_lo, lo_cols))
        
        # Верхняя граница
        hi_cols = [f"{name}-hi-{level}" for name in model_names 
                   if f"{name}-hi-{level}" in eval_df.columns]
        if hi_cols:
            losses_hi = [
                mean_pinball_loss(eval_df[target_col], eval_df[col], alpha=alpha_high)
                for col in hi_cols
            ]
            weights_hi = np.array([1.0 / (l + epsilon) for l in losses_hi])
            weights_hi /= weights_hi.sum()
            result[f'QEnsemble-hi-{level}'] = sum(w * eval_df[col] for w, col in zip(weights_hi, hi_cols))
    
    return result


def plot_cv_windows_subplots(
    df_original,
    cv_df,
    cutoffs,
    series_id='Consumption',
    model=None,
    level=None,
    plot_anomalies=True,
    refit=True,
    input_size=None,
    figsize_per_row=2.5
):
    """
    Визуализация каждого окна CV на отдельном subplot.
    """
    n_windows = len(cutoffs)
    figsize = (12, n_windows * figsize_per_row)
    fig, axes = plt.subplots(n_windows, 1, figsize=figsize, sharex=True)
    
    if n_windows == 1:
        axes = [axes]
    
    orig = df_original[df_original['unique_id'] == series_id].sort_values('ds')
    
    all_model_cols = [col for col in cv_df.columns 
                     if col not in ['unique_id', 'ds', 'cutoff', 'y']]
    if model is None:
        model_col = all_model_cols[0]
    else:
        model_col = model
    
    if refit is True:
        fit_windows = list(range(n_windows))
    elif refit is False:
        fit_windows = [0]
    elif isinstance(refit, int) and refit > 0:
        fit_windows = list(range(0, n_windows, refit))
    else:
        fit_windows = [0]
    
    for k, (ax, cutoff) in enumerate(zip(axes, cutoffs)):
        ax.plot(orig['ds'], orig['y'], 'k-', linewidth=1, alpha=0.7, label='Target')
        
        window_data = cv_df[
            (cv_df['cutoff'] == cutoff) & 
            (cv_df['unique_id'] == series_id)
        ].sort_values('ds')
        
        if not window_data.empty:
            ax.plot(window_data['ds'], window_data[model_col], 
                   'r-', linewidth=2.5, label=f'Forecast ({model_col})')
            
            if level is not None:
                for lev in level:
                    lo_col = f'{model_col}-lo-{lev}'
                    hi_col = f'{model_col}-hi-{lev}'
                    if lo_col in window_data.columns and hi_col in window_data.columns:
                        ax.fill_between(
                            window_data['ds'],
                            window_data[lo_col],
                            window_data[hi_col],
                            alpha=0.2,
                            color='red',
                            label=f'Level {lev}%'
                        )
            
            if plot_anomalies and level is not None:
                for lev in level:
                    lo_col = f'{model_col}-lo-{lev}'
                    hi_col = f'{model_col}-hi-{lev}'
                    if lo_col in window_data.columns and hi_col in window_data.columns:
                        anomalies = (window_data['y'] < window_data[lo_col]) | \
                                   (window_data['y'] > window_data[hi_col])
                        if anomalies.any():
                            anom_data = window_data[anomalies]
                            ax.scatter(anom_data['ds'], anom_data['y'], 
                                     color='red', s=30, alpha=0.8, 
                                     label=f'Anomalies {lev}%')
            
            cutoff_dt = pd.to_datetime(cutoff)
            ax.axvline(x=cutoff_dt, color='red', linestyle='--', alpha=0.8, label='Test start')
            
            test_end = window_data['ds'].max()
            ax.axvline(x=test_end, color='red', linestyle='--', alpha=0.8, label='Test end')
            
            if input_size is not None:
                orig_series = df_original[df_original['unique_id'] == series_id].sort_values('ds')
                if len(orig_series) > 1:
                    freq = pd.infer_freq(orig_series['ds'])
                    if freq:
                        train_start = cutoff_dt - pd.tseries.frequencies.to_offset(f'{input_size}{freq}')
                    else:
                        min_step = orig_series['ds'].diff().min()
                        train_start = cutoff_dt - input_size * min_step
                else:
                    train_start = cutoff_dt - pd.Timedelta(days=input_size)
                
                train_end_color = 'blue' if k in fit_windows else 'gray'
                train_end_style = '-' if k in fit_windows else ':'
                ax.axvline(x=train_start, color=train_end_color, linestyle=train_end_style, 
                          alpha=0.8, label='Train start')
        
        ax.set_ylabel(f'Window {k+1}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Date')
    title = f'CV Windows - {series_id} ({model_col})'
    if refit is not True:
        title += f', refit={refit}'
    if input_size:
        title += f', train_len={input_size}'
    fig.suptitle(title, y=0.98)
    plt.tight_layout()
    return fig, axes


import pandas as pd
import numpy as np
from scipy.stats import t, norm
from typing import Optional, List

def dm_test(
    eval_sf: pd.DataFrame,
    models: Optional[List[str]] = None,
    loss: str = 'mae',
    h: int = 1,
    alpha: float = 0.05,
    alternative: str = 'two-sided',
    by_series: bool = False,
    metric_col: str = 'diff_mae',
    correction: str = 'hln',
    ci_level: float = 0.95,
    block_bootstrap: bool = True,
    n_boot: int = 1000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Diebold-Mariano test для сравнения прогнозирующих моделей на финальном holdout.
    
    ════════════════════════════════════════════════════════════════════════════════
    КОНТЕКСТ ПРИМЕНЕНИЯ
    ════════════════════════════════════════════════════════════════════════════════
    
    Тест Дьёльда-Мариано сравнивает две модели по функции потерь на НЕПЕРЕКРЫ-
    ВАЮЩЕМСЯ отрезке данных. ВАЖНО:
    
    • НЕ применяйте к результатам кросс-валидации с перекрывающимися окнами
      (нарушается независимость наблюдений → завышенная значимость).
    
    • Применяйте ТОЛЬКО к финальному прогнозу (результат forecast() или predict()),
      где каждая точка прогнозируется независимо относительно доступной истории.
    
    • Для всех моделей statsforecast (ETS, ARIMA, Naive, MSTL, TBATS) при стан-
      дартном вызове forecast(h=H) → параметр h_test = 1 всегда.
      Обоснование: прогнозы рекурсивные одношаговые, расстояние между стартами = 1.
    
    ════════════════════════════════════════════════════════════════════════════════
    ПАРАМЕТРЫ
    ════════════════════════════════════════════════════════════════════════════════
    
    eval_sf : pd.DataFrame
        Датафрейм с прогнозами и фактическими значениями.
        Обязательные колонки: 'unique_id', 'ds', 'y'
        Модельные колонки: имена моделей (например, 'AutoETS', 'MSTL')
        Автоматически игнорируются: '-lo-', '-hi-', 'cutoff', 'metric'
    
    models : list[str], optional
        Список моделей для сравнения. Если None — все числовые колонки кроме
        служебных. Минимум 2 модели.
    
    loss : str, default='mae'
        Функция потерь для сравнения:
        • 'mae'  — Mean Absolute Error (рекомендуется для интерпретируемости)
        • 'mse'  — Mean Squared Error (чувствителен к выбросам)
        • 'mape' — Mean Absolute Percentage Error (требует y ≠ 0)
    
    h : int, default=1
        Расстояние между стартами прогнозов в тесте.
        Для рекурсивных прогнозов statsforecast ВСЕГДА = 1.
        Примеры:
          ┌─────────────────────────────────────────────────────────────┐
          │ Сценарий                     │ Значение h │ Почему          │
          ├─────────────────────────────────────────────────────────────┤
          │ Последовательные прогнозы    │     1      │ Старты каждую   │
          │ (как в forecast(h=24))      │            │ точку           │
          │ Многошаговые без перекрытия  │    =горизонт│ Старты раз в    │
          │ (прогноз на 7 дней раз в нед)│            │ 7 точек         │
          └─────────────────────────────────────────────────────────────┘
    
    alpha : float, default=0.05
        Уровень значимости для статистических тестов.
    
    alternative : str, default='two-sided'
        Направление альтернативной гипотезы:
        • 'two-sided' — H₁: модели различаются (любое направление)
        • 'less'      — H₁: model_1 лучше model_2 (потери меньше)
        • 'greater'   — H₁: model_1 хуже model_2 (потери больше)
    
    by_series : bool, default=False
        • False — агрегированный тест по всем рядам (рекомендуется, мощнее)
        • True  — отдельный тест для каждого ряда (возвращает колонку 'unique_id')
    
    metric_col : str, default='diff_mae'
        Метрика для отображения в колонке 'metric_value':
        • 'diff_mae'  — разность MAE (model_1 - model_2) в абсолютных единицах
        • 'mae_1'     — MAE первой модели в паре
        • 'mae_2'     — MAE второй модели в паре
        • 'diff_rmse' — разность RMSE
        • 'rmse_1'    — RMSE первой модели
        • 'rmse_2'    — RMSE второй модели
    
    correction : str, default='hln'
        Метод коррекции статистики для малых выборок:
        • 'none' — оригинальный тест (асимптотический, риск завышенной значимости)
        • 'hln'  — поправка Harvey-Leybourne-Newbold (рекомендуется)
        • 'auto' — автоматически: HLN если n_obs < 100, иначе 'none'
        
        Формула поправки HLN:
            DM_HLN = DM × √[(T + 1 - 2h + h(h-1)/T) / (T - h)]
        где T — количество наблюдений, h — расстояние между стартами.
        
        Влияние поправки на критическое значение:
          T=10  → +5.4%  |  T=30  → +1.7%  |  T=100 → +0.5%  |  T=200 → +0.25%
    
    ci_level : float, default=0.95
        Уровень доверительного интервала для разности потерь (0.90, 0.95, 0.99).
    
    block_bootstrap : bool, default=True
        ┌──────────────────────────────────────────────────────────────────────┐
        │ БЛОЧНЫЙ БУТСТРАП (Block Bootstrap)                                   │
        ├──────────────────────────────────────────────────────────────────────┤
        │ Проблема: в временных рядах наблюдения зависимы (автокорреляция).    │
        │ Обычный бутстрап (случайная выборка точек) разрушает эту зависимость │
        │ → заниженная дисперсия → завышенная значимость.                      │
        │                                                                      │
        │ Решение: сэмплировать БЛОКИ последовательных наблюдений, а не точки. │
        │ Это сохраняет автокорреляционную структуру внутри блока.             │
        │                                                                      │
        │ Реализация: Moving Block Bootstrap (MBB)                             │
        │   1. Делим ряд на неперекрывающиеся блоки размера B                  │
        │   2. Случайно выбираем блоки с возвратом                             │
        │   3. Склеиваем выбранные блоки в новый ряд                           │
        │                                                                      │
        │ Выбор размера блока B:                                               │
        │   •  маленький (B=1) → обычный бутстрап → игнорирует зависимость     │
        │   • Слишком большой (B=T)   → одна выборка → нет вариации            │
        │   • Оптимально: B ≈ 5–30 точек (эвристика: min(30, max(5, T//10)))   │
        │                                                                      │
        │ Когда отключать (block_bootstrap=False):                             │
        │   • Ряды с очень слабой автокорреляцией (|ρ₁| < 0.1)                 │
        │   • Для ускорения расчётов при очень больших выборках (T > 10⁴)      │
        │   • При использовании асимптотического CI (менее надёжно)            │
        └──────────────────────────────────────────────────────────────────────┘
    
    n_boot : int, default=1000
        Количество итераций бутстрапа для расчёта доверительного интервала.
        Рекомендации: 500 (быстро), 1000 (стандарт), 2000 (точно).
    
    random_state : int, default=42
        Для воспроизводимости бутстрапа.
    
    ════════════════════════════════════════════════════════════════════════════════
    ВОЗВРАЩАЕТ
    ════════════════════════════════════════════════════════════════════════════════
    
    pd.DataFrame с колонками:
    
    Базовые:
    • model_1, model_2       — сравниваемые модели
    • unique_id              — только если by_series=True
    
    Статистика теста:
    • dm_stat                — статистика DM (согласно выбранной коррекции)
    • p_value                — p-value теста
    • significant            — True если p_value < alpha
    • correction_applied     — применённая коррекция: 'none' или 'hln'
    • n_obs                  — количество наблюдений в тесте
    
    Практическая значимость:
    • metric_value           — значение метрики согласно metric_col
    • effect_size            — Cohen's d_z = mean(разность) / std(разность)
                             Интерпретация: |d|<0.2 слабый, 0.2-0.5 малый,
                             0.5-0.8 средний, ≥0.8 сильный
    • ci_low, ci_high        — доверительный интервал для разности потерь
    • is_0_in_CI             — True если 0 ∈ [ci_low, ci_high]
                             False → практическая значимость (различия достаточно
                             велики для принятия решения)
    • better_model           — модель с меньшими потерями
    
    ════════════════════════════════════════════════════════════════════════════════
    ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ
    ════════════════════════════════════════════════════════════════════════════════
    
    Надёжное различие требует ВСЕХ трёх условий:
      1. significant = True      (статистическая значимость, p < alpha)
      2. is_0_in_CI = False      (практическая значимость, 0 ∉ CI)
      3. |effect_size| ≥ 0.3     (умеренный эффект)
    
    Пример вывода:
      ┌──────────┬──────────┬─────────┬─────────┬────────────┬──────────────┐
      │ model_1  │ model_2  │ p_value │is_0_in_CI│effect_size │ better_model │
      ├──────────┼──────────┼─────────┼─────────┼────────────┼──────────────┤
      │ AutoETS  │ Naive    │  0.021  │  False  │    0.42    │   AutoETS    │ ← НАДЁЖНО
      │ MSTL     │ TBATS    │  0.003  │  True   │    0.18    │   MSTL       │ ← Слабый эффект
      │ ARIMA    │ ETS      │  0.142  │  True   │    0.09    │   ARIMA      │ ← Незначимо
      └──────────┴──────────┴─────────┴─────────┴────────────┴──────────────┘
    
    ════════════════════════════════════════════════════════════════════════════════
    ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
    ════════════════════════════════════════════════════════════════════════════════
    
    # 1. Быстрый запуск (рекомендуемые настройки по умолчанию)
    results = dm_test(eval_sf, loss='mae')
    
    # 2. One-sided тест: проверяем гипотезу "Модель А лучше Б"
    results = dm_test(
        eval_sf,
        models=['AutoETS', 'SeasonalNaive'],
        alternative='less',   # H₁: AutoETS лучше SeasonalNaive
        correction='hln'
    )
    
    # 3. Отключить блочный бутстрап для ускорения (только при слабой автокорреляции)
    results = dm_test(eval_sf, block_bootstrap=False, n_boot=500)
    
    # 4. Сравнение по каждому ряду отдельно
    results = dm_test(eval_sf, by_series=True)
    
    # 5. Фильтрация надёжных различий
    reliable = results[
        (results['significant']) & 
        (~results['is_0_in_CI']) & 
        (abs(results['effect_size']) >= 0.3)
    ]
    """
    # === 1. Определение моделей ===
    reserved = {'unique_id', 'ds', 'y', 'cutoff', 'metric', 'date', 'timestamp'}
    candidate_cols = [
        c for c in eval_sf.columns 
        if c not in reserved 
        and '-lo-' not in c 
        and '-hi-' not in c
        and pd.api.types.is_numeric_dtype(eval_sf[c])
    ]
    
    if models is None:
        models = candidate_cols
    if len(models) < 2:
        raise ValueError(f"Требуется минимум 2 модели, получено: {len(models)}")
    
    valid_metrics = ['diff_mae', 'mae_1', 'mae_2', 'diff_rmse', 'rmse_1', 'rmse_2']
    if metric_col not in valid_metrics:
        raise ValueError(f"metric_col должен быть одним из {valid_metrics}")
    
    valid_corrections = ['none', 'hln', 'auto']
    if correction not in valid_corrections:
        raise ValueError(f"correction должен быть одним из {valid_corrections}, получено: {correction}")
    
    # === 2. Вспомогательные функции ===
    def get_loss(actual, pred, loss_type):
        if loss_type == 'mse':
            return (actual - pred) ** 2
        elif loss_type == 'mae':
            return np.abs(actual - pred)
        elif loss_type == 'mape':
            mask = actual != 0
            loss = np.full_like(actual, np.nan, dtype=float)
            loss[mask] = np.abs((actual[mask] - pred[mask]) / actual[mask])
            return loss
        else:
            raise ValueError("loss: 'mae', 'mse' или 'mape'")
    
    def calc_metric(actual, pred, metric_type):
        if metric_type.startswith('mae'):
            return np.mean(np.abs(actual - pred))
        elif metric_type.startswith('rmse'):
            return np.sqrt(np.mean((actual - pred) ** 2))
        else:
            raise ValueError
    
    def calc_dm(d, h, alt, correction_method, T):
        """Возвращает (dm_stat, p_value, correction_used)"""
        d_bar = np.mean(d)
        gamma = [np.var(d, ddof=1)]
        for lag in range(1, h):
            gamma.append(np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar)))
        
        var_d_bar = (gamma[0] + 2 * sum(gamma[1:h])) / T
        if var_d_bar <= 0:
            return np.nan, np.nan, 'invalid'
        
        dm_raw = d_bar / np.sqrt(var_d_bar)
        
        # Применение коррекции
        if correction_method == 'hln' or (correction_method == 'auto' and T < 100):
            hln_factor = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / (T - h))
            dm_adj = dm_raw * hln_factor
            correction_used = 'hln'
        else:
            dm_adj = dm_raw
            correction_used = 'none'
        
        # Расчёт p-value
        if alt == 'two-sided':
            p_val = 2 * t.cdf(-abs(dm_adj), df=T-1)
        elif alt == 'less':
            p_val = t.cdf(dm_adj, df=T-1)
        elif alt == 'greater':
            p_val = 1 - t.cdf(dm_adj, df=T-1)
        else:
            raise ValueError
        
        return dm_adj, p_val, correction_used
    
    def calc_ci(d, h, ci_level, block_bootstrap, n_boot, rng, T):
        """Возвращает (ci_low, ci_high, effect_size)"""
        # Эффект-размер
        effect_size = np.mean(d) / np.std(d, ddof=1) if np.std(d, ddof=1) > 0 else 0.0
        
        # Доверительный интервал
        if block_bootstrap:
            # Moving Block Bootstrap
            block_size = max(5, min(30, T // 10))  # эвристика размера блока
            n_blocks = T // block_size + (1 if T % block_size else 0)
            block_starts = list(range(0, T, block_size))
            boot_means = []
            
            for _ in range(n_boot):
                sampled_blocks = rng.choice(block_starts, size=n_blocks, replace=True)
                idx = np.concatenate([
                    np.arange(start, min(start + block_size, T)) 
                    for start in sampled_blocks
                ])
                boot_means.append(np.mean(d[idx]))
            
            ci_low, ci_high = np.percentile(boot_means, [100*(1-ci_level)/2, 100*(1-(1-ci_level)/2)])
        else:
            # Асимптотический CI (игнорирует автокорреляцию — менее надёжен)
            d_bar = np.mean(d)
            gamma = [np.var(d, ddof=1)]
            for lag in range(1, h):
                gamma.append(np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar)))
            var_d_bar = (gamma[0] + 2 * sum(gamma[1:h])) / T
            se = np.sqrt(var_d_bar)
            z = norm.ppf(1 - (1 - ci_level) / 2)
            ci_low, ci_high = d_bar - z * se, d_bar + z * se
        
        return ci_low, ci_high, effect_size
    
    # === 3. Расчёт тестов ===
    results = []
    rng = np.random.default_rng(random_state)
    
    if by_series:
        for uid, group in eval_sf.groupby('unique_id'):
            actual = group['y'].values
            for i in range(len(models)):
                for j in range(i+1, len(models)):
                    d = get_loss(actual, group[models[i]].values, loss) - \
                        get_loss(actual, group[models[j]].values, loss)
                    d = d[~np.isnan(d)]
                    T = len(d)
                    if T < max(10, h+1):
                        continue
                    
                    dm, p, corr_used = calc_dm(d, h, alternative, correction, T)
                    if np.isnan(dm):
                        continue
                    
                    ci_l, ci_h, es = calc_ci(d, h, ci_level, block_bootstrap, n_boot, rng, T)
                    
                    # Метрика для отображения
                    if metric_col == 'diff_mae':
                        metric_val = np.mean(np.abs(actual - group[models[i]].values) - 
                                           np.abs(actual - group[models[j]].values))
                    elif metric_col == 'mae_1':
                        metric_val = calc_metric(actual, group[models[i]].values, 'mae')
                    elif metric_col == 'mae_2':
                        metric_val = calc_metric(actual, group[models[j]].values, 'mae')
                    elif metric_col == 'diff_rmse':
                        metric_val = np.sqrt(np.mean((actual - group[models[i]].values)**2)) - \
                                    np.sqrt(np.mean((actual - group[models[j]].values)**2))
                    elif metric_col == 'rmse_1':
                        metric_val = calc_metric(actual, group[models[i]].values, 'rmse')
                    elif metric_col == 'rmse_2':
                        metric_val = calc_metric(actual, group[models[j]].values, 'rmse')
                    
                    better = models[i] if np.mean(d) < 0 else models[j]
                    
                    results.append({
                        'unique_id': uid,
                        'model_1': models[i],
                        'model_2': models[j],
                        'dm_stat': dm,
                        'p_value': p,
                        'significant': p < alpha,
                        'better_model': better,
                        'metric_value': metric_val,
                        'effect_size': es,
                        'ci_low': ci_l,
                        'ci_high': ci_h,
                        'is_0_in_CI': (ci_l <= 0 <= ci_h),
                        'correction_applied': corr_used,
                        'n_obs': T
                    })
    else:
        actual = eval_sf['y'].values
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                d = get_loss(actual, eval_sf[models[i]].values, loss) - \
                    get_loss(actual, eval_sf[models[j]].values, loss)
                d = d[~np.isnan(d)]
                T = len(d)
                if T < max(10, h+1):
                    continue
                
                dm, p, corr_used = calc_dm(d, h, alternative, correction, T)
                if np.isnan(dm):
                    continue
                
                ci_l, ci_h, es = calc_ci(d, h, ci_level, block_bootstrap, n_boot, rng, T)
                
                if metric_col == 'diff_mae':
                    metric_val = np.mean(np.abs(actual - eval_sf[models[i]].values) - 
                                       np.abs(actual - eval_sf[models[j]].values))
                elif metric_col == 'mae_1':
                    metric_val = calc_metric(actual, eval_sf[models[i]].values, 'mae')
                elif metric_col == 'mae_2':
                    metric_val = calc_metric(actual, eval_sf[models[j]].values, 'mae')
                elif metric_col == 'diff_rmse':
                    metric_val = np.sqrt(np.mean((actual - eval_sf[models[i]].values)**2)) - \
                                np.sqrt(np.mean((actual - eval_sf[models[j]].values)**2))
                elif metric_col == 'rmse_1':
                    metric_val = calc_metric(actual, eval_sf[models[i]].values, 'rmse')
                elif metric_col == 'rmse_2':
                    metric_val = calc_metric(actual, eval_sf[models[j]].values, 'rmse')
                
                better = models[i] if np.mean(d) < 0 else models[j]
                
                results.append({
                    'model_1': models[i],
                    'model_2': models[j],
                    'dm_stat': dm,
                    'p_value': p,
                    'significant': p < alpha,
                    'better_model': better,
                    'metric_value': metric_val,
                    'effect_size': es,
                    'ci_low': ci_l,
                    'ci_high': ci_h,
                    'is_0_in_CI': (ci_l <= 0 <= ci_h),
                    'correction_applied': corr_used,
                    'n_obs': T
                })
    
    if not results:
        raise ValueError("Недостаточно данных для теста (минимум 10 наблюдений)")
    
    df_res = pd.DataFrame(results)
    
    # Сортировка
    sort_cols = ['significant', 'is_0_in_CI', 'p_value']
    sort_asc = [False, True, True]
    if by_series:
        sort_cols = ['unique_id'] + sort_cols
        sort_asc = [True] + sort_asc
    
    df_res = df_res.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)
    
    return df_res
