
import pandas as pd

from statsforecast import StatsForecast

from utilsforecast.evaluation import evaluate

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from utilsforecast.evaluation import evaluate
from typing import List, Callable, Optional

import matplotlib.pyplot as plt

   
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

def plot_cv_metric(result, metric='mae', series_list=None, figsize=(12, 8)):
    """
    Визуализация CV метрики:
    - слева: динамика метрики по CV-окнам
    - справа: распределение метрики (boxplot)
    
    Параметры:
    ----------
    result : pd.DataFrame
        Результат evaluate_cv() с MultiIndex: metric, unique_id, cutoff
        и колонками — модели
    metric : str
        Метрика для визуализации ('mae', 'rmse', 'smape')
    series_list : list, optional
        Список временных рядов (unique_id). Если None — берутся все уникальные.
    figsize : tuple
        Размер фигуры
    """
    df_clean = result.copy()
    # df_clean = df_clean.reset_index()  # превращаем MultiIndex в колонки
    
    # Список уникальных рядов
    if series_list is None:
        series_list = sorted(df_clean['unique_id'].unique())
    
    # Список моделей (колонки после metric, unique_id, cutoff)
    model_cols = df_clean.columns.difference(['metric', 'unique_id', 'cutoff'])
    
    n_series = len(series_list)
    fig, axes = plt.subplots(n_series, 2, figsize=figsize)
    
    if n_series == 1:
        axes = [axes]
    
    for i, series in enumerate(series_list):
        data = df_clean[(df_clean['unique_id'] == series) & (df_clean['metric'] == metric)]
        data = data.sort_values('cutoff')
        windows = range(1, len(data) + 1)
        
        # Левый график: динамика по CV-окнам
        for model in model_cols:
            axes[i][0].plot(windows, data[model], label=model, marker='o')
        axes[i][0].set_title(f'{series} - {metric.upper()} over windows')
        axes[i][0].set_ylabel(metric.upper())
        axes[i][0].legend()
        axes[i][0].grid(True)
        
        # Правый график: boxplot распределения
        box_data = [data[m].values for m in model_cols]
        axes[i][1].boxplot(box_data, tick_labels=model_cols)
        axes[i][1].set_title(f'{series} - {metric.upper()} distribution')
        axes[i][1].set_ylabel(metric.upper())
        axes[i][1].grid(True)
    
    # Подписи осей X
    axes[-1][0].set_xlabel('CV Window Number')
    axes[-1][1].set_xlabel('Model')
    
    plt.tight_layout()
    plt.show()

def plot_cv_ranks(eval_df, metrics=None, figsize=(10, 4), error_type='iqr'):
    """
    Визуализация средних рангов моделей по метрикам с показателем разброса.
    
    Параметры:
    -----------
    eval_df : pd.DataFrame
        Результат evaluate с колонками ['unique_id', 'metric', 'cutoff'] и колонками моделей
    metrics : list, optional
        Список метрик для отображения. Если None - используются все уникальные метрики
    figsize : tuple, default=(10, 4)
        Размер фигуры (ширина, высота)
    error_type : str, default='iqr'
        Тип меры разброса:
        - 'iqr': интерквартильный размах (75% - 25% квантиль)
        - 'mad': медианное абсолютное отклонение
        - 'std': стандартное отклонение
    
    Возвращает:
    -----------
    None
        Выводит график и печатает таблицу статистики
    
    Особенности:
    ------------
    - Ранжирует модели внутри каждой метрики по каждому окну кросс-валидации
    - Строит горизонтальные бары для каждой модели с разбивкой по метрикам
    - Показывает усы с ошибками для каждой метрики отдельно
    - Ограничивает ошибки, чтобы не уходили за границы возможных рангов
    """
    if metrics is None:
        metrics = eval_df['metric'].unique()

    model_cols = eval_df.columns.difference(['unique_id', 'cutoff', 'metric'])
    n_models = len(model_cols)

    # Ранжируем модели
    ranked_rows = []
    for metric in metrics:
        df_metric = eval_df.query('metric == @metric').copy()
        df_metric[model_cols] = df_metric[model_cols].rank(axis=1, method='average')
        df_metric['metric'] = metric
        ranked_rows.append(df_metric)

    df_ranks = pd.concat(ranked_rows, ignore_index=True)

    # Преобразуем в long формат
    df_long = df_ranks.melt(
        id_vars=['unique_id', 'cutoff', 'metric'],
        value_vars=model_cols,
        var_name='model',
        value_name='rank'
    )

    # Агрегация
    if error_type == 'iqr':
        agg = df_long.groupby(['metric', 'model'])['rank'].agg(['median', lambda x: x.quantile(0.75) - x.quantile(0.25)]).reset_index()
        agg.columns = ['metric', 'model', 'median', 'iqr']
        error_col = 'iqr'
        central_col = 'median'
    elif error_type == 'mad':
        from scipy.stats import median_abs_deviation
        agg = df_long.groupby(['metric', 'model'])['rank'].agg(['median', median_abs_deviation]).reset_index()
        agg.columns = ['metric', 'model', 'median', 'mad']
        error_col = 'mad'
        central_col = 'median'
    else:  # std
        agg = df_long.groupby(['metric', 'model'])['rank'].agg(['mean', 'std']).reset_index()
        agg.columns = ['metric', 'model', 'mean', 'std']
        error_col = 'std'
        central_col = 'mean'

    # Подготовка для графика
    models = sorted(model_cols)

    fig, ax = plt.subplots(figsize=figsize)

    # Цвета для метрик
    palette = sns.color_palette("Set2", len(metrics))
    metric_to_color = dict(zip(metrics, palette))

    n_metrics = len(metrics)
    y_positions = np.arange(len(models))
    bar_width = 0.8 / n_metrics

    for i, metric in enumerate(metrics):
        color = metric_to_color[metric]
        metric_data = agg[agg['metric'] == metric].set_index('model')
        y_offsets = y_positions + (i - (n_metrics - 1)/2) * bar_width

        means = [metric_data.loc[m, central_col] if m in metric_data.index else np.nan for m in models]
        stds = [metric_data.loc[m, error_col] if m in metric_data.index else np.nan for m in models]

        # Ограничиваем усики: не должны уходить за [1, n_models]
        capped_errors = []
        for mean_val, std_val in zip(means, stds):
            if np.isnan(mean_val) or np.isnan(std_val):
                capped_errors.append(np.nan)
            else:
                # Усики не должны уходить за границы [1, n_models]
                lower_bound = max(1, mean_val - std_val)
                upper_bound = min(n_models, mean_val + std_val)
                capped_error = min(std_val, mean_val - lower_bound, upper_bound - mean_val)
                capped_errors.append(capped_error)

        # Бары
        bars = ax.barh(y_offsets, means, height=bar_width, color=color, label=metric, alpha=0.8)

        # Усики
        for y, mean_val, err in zip(y_offsets, means, capped_errors):
            if not np.isnan(mean_val) and not np.isnan(err):
                ax.errorbar(
                    x=mean_val,
                    y=y,
                    xerr=err,
                    fmt='none',
                    ecolor=color,
                    capsize=4,
                    capthick=1.5,
                    elinewidth=1.2
                )

    # Настройка осей
    ax.set_yticks(y_positions)
    ax.set_yticklabels(models)
    ax.set_xlabel('Average/Median Rank (lower is better)')
    ax.set_xlim(left=0)  # чтобы не уходило в отрицательные значения
    ax.set_title(f"Rank statistics with {error_type.upper()}:")
    ax.legend(title='Metric')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # # Вывод статистики
    # print(f"\nRank statistics (Central ± {error_type.upper()}):")
    # for model in models:
    #     print(f"\n{model}:")
    #     for metric in metrics:
    #         row = agg[(agg['model'] == model) & (agg['metric'] == metric)]
    #         if not row.empty:
    #             center = row[central_col].iloc[0]
    #             err = row[error_col].iloc[0]
    #             print(f"  {metric}: {center:.2f} ± {err:.2f}")      
    
def plot_cv_series(
    df_original,
    cv_df,
    cutoffs,
    series_id='Consumption',
    model=None,  # ← теперь может быть None или 'all'
    level=None,
    plot_anomalies=True,
    refit=True,
    input_size=None,
    max_insample_length=None,
    figsize_per_plot=(12, 2),
    **kwargs
):
    """
    Визуализация каждого окна кросс-валидации на отдельном subplot.
    
    Параметры:
    ----------
    model : str or list or None
        Имя модели, список моделей или 'all'. Если None - используется первая доступная.
        См. extract_model_names для автоматического извлечения.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import re

    def extract_model_names(df, base_cols=['unique_id', 'ds', 'y', 'cutoff']):
        """Извлекает уникальные названия моделей из колонок DataFrame."""
        if base_cols is None:
            base_cols = ['unique_id', 'ds', 'y', 'cutoff']
        base_set = set(base_cols)
        cols = [c for c in df.columns if c not in base_set]
        models = {
            re.sub(r'[-_](lo|hi)[-_]\d+(\.\d+)?$', '', c)
            for c in cols
        }
        models = {m for m in models if m and m not in base_set}
        return sorted(models)

    n_windows = len(cutoffs)
    figsize = (figsize_per_plot[0], n_windows * figsize_per_plot[1])
    fig, axes = plt.subplots(n_windows, 1, figsize=figsize, sharex=True)
    
    if n_windows == 1:
        axes = [axes]
    
    orig = df_original[df_original['unique_id'] == series_id].sort_values('ds')
    
    # Извлекаем все модели из cv_df
    all_models = extract_model_names(cv_df)
    
    # Определяем, какие модели использовать
    if model is None:
        models_to_plot = [all_models[0]] if all_models else []
    elif model == 'all':
        models_to_plot = all_models
    elif isinstance(model, str):
        models_to_plot = [model]
    elif isinstance(model, list):
        models_to_plot = model
    else:
        models_to_plot = []
    
    # Оставляем только существующие модели
    models_to_plot = [m for m in models_to_plot if m in all_models]
    if not models_to_plot:
        models_to_plot = [all_models[0]] if all_models else []

    if refit is True:
        fit_windows = list(range(n_windows))
    elif refit is False:
        fit_windows = [0]
    elif isinstance(refit, int) and refit > 0:
        fit_windows = list(range(0, n_windows, refit))
    else:
        fit_windows = [0]
    
    for k, (ax, cutoff) in enumerate(zip(axes, cutoffs)):
        # Инсемпл: последние max_insample_length точек до cutoff
        cutoff_mask = orig['ds'] <= cutoff
        orig_filtered = orig[cutoff_mask]
        if max_insample_length is not None:
            recent_orig = orig_filtered.tail(max_insample_length)
        else:
            recent_orig = orig_filtered
        ax.plot(recent_orig['ds'], recent_orig['y'], 'k-', linewidth=1, alpha=0.7, label='Target')
        
        window_data = cv_df[
            (cv_df['cutoff'] == cutoff) & 
            (cv_df['unique_id'] == series_id)
        ].sort_values('ds')
        
        if not window_data.empty:
            # Таргет на тестовом участке (тонкая линия)
            if 'y' in window_data.columns:
                ax.plot(window_data['ds'], window_data['y'], 'k-', linewidth=1, alpha=0.7)
            
            # Прогнозы для выбранных моделей
            for model_name in models_to_plot:
                if model_name in window_data.columns:
                    ax.plot(window_data['ds'], window_data[model_name], 
                           linewidth=2.5, label=f'Forecast ({model_name})')
                    
                    # Интервалы и аномалии только для текущей модели
                    if level is not None:
                        for lev in level:
                            lo_col = f'{model_name}-lo-{lev}'
                            hi_col = f'{model_name}-hi-{lev}'
                            if lo_col in window_data.columns and hi_col in window_data.columns:
                                ax.fill_between(
                                    window_data['ds'],
                                    window_data[lo_col],
                                    window_data[hi_col],
                                    alpha=0.2,
                                    label=f'Level {lev}% ({model_name})'
                                )
                    
                    if plot_anomalies and level is not None:
                        for lev in level:
                            lo_col = f'{model_name}-lo-{lev}'
                            hi_col = f'{model_name}-hi-{lev}'
                            if lo_col in window_data.columns and hi_col in window_data.columns:
                                anomalies = (window_data['y'] < window_data[lo_col]) | \
                                           (window_data['y'] > window_data[hi_col])
                                if anomalies.any():
                                    anom_data = window_data[anomalies]
                                    ax.scatter(anom_data['ds'], anom_data['y'], 
                                             s=30, alpha=0.8, 
                                             label=f'Anomalies {lev}% ({model_name})')
            
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
    model_str = ', '.join(models_to_plot)
    title = f'CV Windows - {series_id} ({model_str})'
    if refit is not True:
        title += f', refit={refit}'
    if input_size:
        title += f', train_len={input_size}'
    fig.suptitle(title, y=0.98)
    plt.tight_layout()
    return fig, axes



# def evaluate_cv(
#     crossvalidation_df,
#     metrics,
#     model_names,
#     train_df = None,
#     target_col='y',
#     level=None,
#     ts_aggregate=True,
#     cutoff_aggregate=True,
#     ret_styled = True
# ):
#     """
#     Оценка результатов кросс-валидации с гибкой агрегацией.
    
#     Параметры:
#     ----------
#     crossvalidation_df : pd.DataFrame
#         Данные в лонг-формате с колонкой 'cutoff'     
#     metrics : list
#         Список функций метрик (например, [smape, mae, rmse])
#     model_names : list
#         Список названий моделей
#     train_df : pd.DataFrame
#         Данные для scaled метрик  
#     target_col : str, default='y'
#         Название колонки с целевой переменной
#     level : float or None, default=None
#         Уровень для квантильных прогнозов
#     ts_aggregate : bool, default=True
#         Если True — агрегировать по временным рядам (уникальным unique_id)
#     cutoff_aggregate : bool, default=True
#         Если True — агрегировать по окнам кросс-валидации (cutoff)
#     ret_styled: bool, default=True
#         Если True — стильная таблица на выходе
#     Возвращает:
#     -----------
#     pd.io.formats.style.Styler
#         Стилизованный датафрейм с метриками
#     """
#     evaluations = []
#     cutoffs = crossvalidation_df['cutoff'].unique()
    
#     for c in cutoffs:
#         df_cv = crossvalidation_df.query('cutoff == @c')
#         evaluation = evaluate(
#             df=df_cv,
#             metrics=metrics,
#             models=model_names,
#             level=level,
#             train_df = train_df,
#             target_col=target_col
#         )
#         evaluation['cutoff'] = c  # сохраняем информацию об окне
#         evaluations.append(evaluation)
    
#     evaluations = pd.concat(evaluations, ignore_index=True)
    
#     # Определяем уровни группировки
#     group_cols = ['metric']
#     if not ts_aggregate and 'unique_id' in evaluations.columns:
#         group_cols.append('unique_id')
#     if not cutoff_aggregate:
#         group_cols.append('cutoff')
    
#     # Агрегация
#     if len(group_cols) > 0:
#         evaluations = evaluations.groupby(group_cols).mean(numeric_only=True)
#     else:
#         evaluations = evaluations.mean(numeric_only=True).to_frame().T
    
#     # Стилизация с учётом структуры индекса
#     if isinstance(evaluations.index, pd.MultiIndex):
#         # Для мультииндекса применяем градиент по строкам
#         styled = evaluations.style.background_gradient(
#             cmap='RdYlGn_r', 
#             axis=1,
#             vmin=evaluations.min().min(),
#             vmax=evaluations.max().max()
#         )
#     else:
#         styled = evaluations.style.background_gradient(cmap='RdYlGn_r', axis=1)
    
#     if ret_styled:
#         return styled.format("{:.2f}")  # Форматирование до 2 знаков (согласно вашим предпочтениям)
#     else:
#         return styled.data