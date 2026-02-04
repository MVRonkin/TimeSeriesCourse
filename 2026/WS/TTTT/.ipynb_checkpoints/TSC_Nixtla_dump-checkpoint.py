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

def extract_model_names(df, base_cols= ['unique_id', 'ds', 'y', 'cutoff']):
    """
    Извлекает уникальные названия моделей из колонок DataFrame.
    Корректно удаляет суффиксы квантилей: -lo-95, -hi-0.5, _lo_90, _hi_0.25 и т.п.
    
    Параметры:
    -----------
    df : pd.DataFrame
        Входной датафрейм
    base_cols : list или None
        Базовые колонки, которые не являются прогнозами моделей.
        По умолчанию: ['unique_id', 'ds', 'y', 'cutoff']
    
    Возвращает:
    ------------
    list
        Отсортированный список уникальных названий моделей
    """
    if base_cols is None:
        base_cols = ['unique_id', 'ds', 'y', 'cutoff']
    
    base_set = set(base_cols)
    cols = [c for c in df.columns if c not in base_set]
    
    # Исправленное регулярное выражение: поддержка целых и дробных чисел (95, 0.5, 0.25)
    models = {
        re.sub(r'[-_](lo|hi)[-_]\d+(\.\d+)?$', '', c)
        for c in cols
    }    
    # Удаляем пустые строки и базовые колонки (на случай артефактов)
    models = {m for m in models if m and m not in base_set}    
    return sorted(models)
    

def plt_style_GOST(fig_size = (12, 2.0)):
    plt.rcParams.update({
        # ШРИФТ
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "font.size": 11,                      # ГОСТ: 10–12 pt
    
        # ОСИ И ПОДПИСИ
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
    
        # РАЗМЕР ФИГУРЫ (A4, отчёты)
        # "figure.figsize": (6.5, 4.0),         # ~16.5 × 10 см
        "figure.figsize": fig_size,         
        "figure.dpi": 150,
        "savefig.dpi": 300,
    
        # СОХРАНЕНИЕ
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    
        # ЛИНИИ И ОСИ
        "axes.linewidth": 1.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
    
        "lines.linewidth": 1.5,
        "patch.linewidth": 1.0,
    
        # СЕТКА
        "axes.grid": True,                   # ГОСТ: обычно без сетки
        "axes.axisbelow": True,
    
        # TEX
        "text.usetex": False,
    })


def plot_series_v2(
    df: Optional[pd.DataFrame] = None,
    forecasts_df: Optional[pd.DataFrame] = None,
    palette: Optional[str] = 'tab10',
    ids: Optional[List[str]] = None,
    plot_random: bool = True,
    max_ids: int = 8,
    models: Optional[List[str]] = None,
    level: Optional[List[float]] = None,
    max_insample_length: Optional[int] = None,
    plot_anomalies: bool = False,
    engine: str = "matplotlib",
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    seed: int = 0,
    resampler_kwargs: Optional[Dict] = None,
    ax: Optional[Union[plt.Axes, np.ndarray, "plotly.graph_objects.Figure"]] = None,
    figsize_per_plot: Tuple[float, float] = (12, 2),
    n_cols: int = 1,  # ← НОВЫЙ ПАРАМЕТР
):
    """
    Обёртка над utilsforecast.plotting.plot_series с предварительным отбором временных рядов,
    поддержкой настраиваемого размера каждого графика и мультиколоночной сетки.
    
    Параметры:
        ...
        figsize_per_plot: (width, height) — размер одного подграфика в дюймах.
        n_cols: int = 1 — количество столбцов в сетке подграфиков, если -1 то все ВР 
        ...
    """

#     fig = plot_series(prep)
# fig.set_size_inches(18, 4, forward=True)
# fig
    if engine != "matplotlib":
        # Для plotly сетка и figsize не управляются здесь — передаём как есть
        return uf_plot_series(
            df=df, forecasts_df=forecasts_df, models=models, level=level,
            max_insample_length=max_insample_length, plot_anomalies=plot_anomalies,
            engine=engine, palette=palette, id_col=id_col, time_col=time_col,
            target_col=target_col, resampler_kwargs=resampler_kwargs, ax=ax
        )

    if df is None and forecasts_df is None:
        raise ValueError("At least one of `df` or `forecasts_df` must be provided.")

    # === Отбор ID ===
    all_ids = set()
    if df is not None:
        all_ids.update(df[id_col].unique())
    if forecasts_df is not None:
        all_ids.update(forecasts_df[id_col].unique())
    all_ids = sorted(all_ids)

    if not all_ids:
        raise ValueError("No series found in provided data.")

    if ids is None:
        if plot_random:
            np.random.seed(seed)
            selected_ids = list(np.random.choice(all_ids, size=min(max_ids, len(all_ids)), replace=False))
        else:
            selected_ids = all_ids[:max_ids]
    else:
        selected_ids = [uid for uid in ids if uid in all_ids][:max_ids]

    if not selected_ids:
        raise ValueError("No valid IDs to plot.")

    # Фильтрация
    df_filtered = df[df[id_col].isin(selected_ids)] if df is not None else None
    forecasts_filtered = forecasts_df[forecasts_df[id_col].isin(selected_ids)] if forecasts_df is not None else None

    n_plots = len(selected_ids)

    # Если пользователь передал ax — используем его (игнорируем сетку и figsize_per_plot)
    if ax is not None:
        return uf_plot_series(
            df=df_filtered, forecasts_df=forecasts_filtered, models=models, level=level,
            max_insample_length=max_insample_length, plot_anomalies=plot_anomalies,
            engine=engine, palette=palette, id_col=id_col, time_col=time_col,
            target_col=target_col, resampler_kwargs=resampler_kwargs, ax=ax
        )

    # === Расчёт сетки ===
    if n_cols == -1:
        n_cols, n_rows = n_plots, 1
    else:
        n_cols = min(n_cols, n_plots)  # избегаем избыточных столбцов
        n_rows = (n_plots + n_cols - 1) // n_cols  # ceil division

    # === Создаём фигуру с нужным размером ===
    width, height = figsize_per_plot
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(width * n_cols, height * n_rows),
        sharex=True,
        squeeze=False  # всегда возвращает 2D массив
    )
    axes = axes.flatten()

    # Скрываем неиспользуемые оси при неполной сетке
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    # Вызываем оригинальную функцию с подготовленными осями
    return uf_plot_series(
        df=df_filtered,
        forecasts_df=forecasts_filtered,
        models=models,
        level=level,
        max_insample_length=max_insample_length,
        plot_anomalies=plot_anomalies,
        engine=engine,
        palette=palette,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        resampler_kwargs=resampler_kwargs,
        ax=axes[:n_plots],  # передаём только нужные оси
    )


 
def evaluate_and_plot(df_train, df_test, forecasts_or_eval, metrics, levels=None, model_names=None, plot=True, modeh=True):
    """
    Универсальная оценка и визуализация прогнозов.
    
    Автоматически определяет:
      - Если переданы чистые прогнозы (без колонки 'y') → делает merge с df_test
      - Если передан смерженный датафрейм (с колонкой 'y') → использует как есть
    
    Параметры
    ----------
    df_train : pd.DataFrame
        Тренировочные данные
    df_test : pd.DataFrame
        Тестовые данные (должны содержать 'y')
    forecasts_or_eval : pd.DataFrame
        Либо чистые прогнозы (без 'y'), либо уже смерженный датафрейм (с 'y')
    metrics : list
        Список метрик из utilsforecast
    levels : list, optional
        Уровни интервалов для оценки (требуют наличия колонок -lo-/-hi-)
    model_names : list, optional
        Явный список моделей. Если None — извлекаются автоматически.
    plot : bool, default=True
        Рисовать ли графики
    
    Возвращает
    ----------
    eval_df : pd.DataFrame
        Смерженный датафрейм с прогнозами и фактом
    metrics_df : pd.DataFrame
        Таблица метрик
    """
    # === Определяем тип входа ===
    has_y = 'y' in forecasts_or_eval.columns
    has_uid_ds = {'unique_id', 'ds'}.issubset(forecasts_or_eval.columns)
    
    if has_y and has_uid_ds:
        # Уже смерженный датафрейм (содержит 'y')
        eval_df = forecasts_or_eval.copy()
    elif has_uid_ds:
        # Чистые прогнозы (нет 'y') — мержим с тестом
        eval_df = df_test[['unique_id', 'ds', 'y']].merge(
            forecasts_or_eval,
            on=['unique_id', 'ds'],
            how='inner'
        )
    else:
        raise ValueError("Input must contain ['unique_id', 'ds'] columns. "
                        "If it also contains 'y' — treated as eval_df, "
                        "otherwise as forecasts.")
    
    # === Извлекаем имена моделей ===
    if model_names is None:
        # Исключаем ВСЕ служебные колонки, включая дубликаты после мержа
        base_cols = ['unique_id', 'ds', 'y', 'cutoff', 'index']
        model_names = extract_model_names(eval_df, base_cols=base_cols)
    
    if not model_names:
        raise ValueError("No forecast models detected. Check input DataFrame columns.")
    
    # === Проверяем наличие интервалов для запрошенных уровней ===
    eval_level = levels
    if levels is not None:
        missing_intervals = []
        for m in model_names:
            for lvl in levels:
                for suffix in ['lo', 'hi']:
                    col = f'{m}-{suffix}-{lvl}'
                    if col not in eval_df.columns:
                        missing_intervals.append(col)
        
        if missing_intervals:
            print(f" Warning: Missing interval columns for level={levels}. "
                  f"Evaluating point metrics only. Missing: {missing_intervals[:3]}...")
            eval_level = None  # отключаем интервалы для оценки метрик
    
    # === Оценка метрик ===
    metrics_df = evaluate(
        df=eval_df,
        metrics=metrics,
        models=model_names,  # ← критически важно указать явно!
        train_df=df_train,
        level=eval_level,
    )

    if modeh:
        metrics_df=metrics_df.pivot(
            index='metric',
            columns='unique_id',
            values=model_names
        )
    else:
        metrics_df=metrics_df.pivot_table(
        index=['unique_id', 'metric'],
        values=model_names
        )
    display(metrics_df.style.format('{:.2f}'))
    
    # === Визуализация ===
    if plot:
        display(plot_series_v2(
            df_train,
            forecasts_df=eval_df,
            level=levels,       
            models=model_names,
            palette='Set1',
        ))
    

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

