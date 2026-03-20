import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from pandas.tseries.frequencies import to_offset

import re
from typing import TYPE_CHECKING, Dict, List, Optional, Union, Tuple, List

try:
    import matplotlib as mpl
    import matplotlib.colors as cm
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        "matplotlib is not installed. Please install it and try again.\n"
        "You can find detailed instructions at https://matplotlib.org/stable/users/installing/index.html"
    )

from utilsforecast.plotting import plot_series as uf_plot_series   
from utilsforecast.evaluation import evaluate

import re


def plot_series_v2(
    df: Optional[pd.DataFrame] = None,
    forecasts_df: Optional[pd.DataFrame] = None,
    anomalies_df: Optional[pd.DataFrame] = None,
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
    n_cols: int = 1,
    marker_size: int = 50,
    anomaly_width: Optional[Union[str, float, pd.Timedelta]] = None,
    plot_cutoff: Optional[Union[bool, str]] = False,
    # === НОВЫЙ ПАРАМЕТР ===
    second_axis: Optional[str] = None,  # Название колонки для правой оси
    second_axis_label: Optional[str] = None,  # Подпись правой оси (опционально)
    second_axis_color: str = 'tab:red',  # Цвет линии второй оси
):
    if engine != "matplotlib":
        return uf_plot_series(
            df=df, forecasts_df=forecasts_df, models=models, level=level,
            max_insample_length=max_insample_length, plot_anomalies=plot_anomalies,
            engine=engine, palette=palette, id_col=id_col, time_col=time_col,
            target_col=target_col, resampler_kwargs=resampler_kwargs, ax=ax
        )

    if df is None and forecasts_df is None:
        raise ValueError("At least one of `df` or `forecasts_df` must be provided.")

    # === Проверка second_axis ===
    if second_axis is not None:
        if forecasts_df is None or second_axis not in forecasts_df.columns:
            raise ValueError(f"Column '{second_axis}' not found in forecasts_df")
        if engine != "matplotlib":
            raise ValueError("second_axis currently supported only for matplotlib engine")

    # ... (остальной код отбора ID без изменений) ...
    
    all_ids = set()
    if df is not None: all_ids.update(df[id_col].unique())
    if forecasts_df is not None: all_ids.update(forecasts_df[id_col].unique())
    if anomalies_df is not None: all_ids.update(anomalies_df[id_col].unique())
    
    all_ids = sorted(all_ids)

    if not all_ids: raise ValueError("No series found in provided data.")

    if ids is None:
        if plot_random:
            np.random.seed(seed)
            selected_ids = list(np.random.choice(all_ids, size=min(max_ids, len(all_ids)), replace=False))
        else:
            selected_ids = all_ids[:max_ids]
    else:
        selected_ids = [uid for uid in ids if uid in all_ids][:max_ids]

    if not selected_ids: raise ValueError("No valid IDs to plot.")

    # Фильтрация основных данных
    df_filtered = df[df[id_col].isin(selected_ids)] if df is not None else None
    forecasts_filtered = forecasts_df[forecasts_df[id_col].isin(selected_ids)] if forecasts_df is not None else None

    n_plots = len(selected_ids)

    # Если передан внешний ax
    if ax is not None:
        # Если передан внешний ax и запрошена вторая ось — обрабатываем отдельно
        if second_axis is not None:
            raise ValueError("second_axis not supported with external ax parameter")
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
        n_cols = min(n_cols, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

    # === Создаём фигуру ===
    width, height = figsize_per_plot
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(width * n_cols, height * n_rows),
        sharex=True, squeeze=False
    )
    axes = axes.flatten()

    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    # === Определяем порядок ID ===
    plotting_order = []
    if df_filtered is not None:
        plotting_order = df_filtered[id_col].unique().tolist()
    elif forecasts_filtered is not None:
        plotting_order = forecasts_filtered[id_col].unique().tolist()
    
    plotting_order = [uid for uid in plotting_order if uid in selected_ids]
    
    active_cutoff_col = None

    # ==========================================
    # === ЛОГИКА ОТРИСОВКИ CUTOFF ===
    # ==========================================
    if plot_cutoff:
        cutoff_col = 'cutoff' if plot_cutoff is True else str(plot_cutoff)
        
        source_df = None
        if forecasts_df is not None and cutoff_col in forecasts_df.columns:
            source_df = forecasts_df
        elif df is not None and cutoff_col in df.columns:
            source_df = df
        
        if source_df is not None:
            active_cutoff_col = cutoff_col
            
            unique_cutoffs = source_df[cutoff_col].dropna().unique()
            try:
                unique_cutoffs = pd.to_datetime(unique_cutoffs)
            except:
                pass
            unique_cutoffs = np.sort(unique_cutoffs)

            max_time = None
            if forecasts_filtered is not None:
                max_time = forecasts_filtered[time_col].max()
            if df_filtered is not None:
                df_max = df_filtered[time_col].max()
                if max_time is None or df_max > max_time:
                    max_time = df_max
            
            if max_time is None and len(unique_cutoffs) > 0:
                max_time = unique_cutoffs[-1]

            for i in range(n_plots):
                ax_i = axes[i]
                
                for j in range(len(unique_cutoffs) - 1):
                    start = unique_cutoffs[j]
                    end = unique_cutoffs[j+1]
                    ax_i.axvspan(start, end, color='grey', alpha=0.05, label='_nolegend_', zorder=0)
                
                if len(unique_cutoffs) > 0 and max_time is not None:
                    last_cut = unique_cutoffs[-1]
                    if max_time > last_cut:
                        ax_i.axvspan(last_cut, max_time, color='grey', alpha=0.05, label='_nolegend_', zorder=0)

                for cut in unique_cutoffs:
                    ax_i.axvline(cut, color='grey', linestyle='--', linewidth=1.5, alpha=0.8, label='_nolegend_', zorder=1)

    # === Рисуем аномалии ===
    if anomalies_df is not None:
        anom_filtered = anomalies_df[anomalies_df[id_col].isin(plotting_order)].copy()
        
        for i, uid in enumerate(plotting_order):
            points = anom_filtered[anom_filtered[id_col] == uid]
            
            if not points.empty:
                if anomaly_width is not None:
                    is_datetime = pd.api.types.is_datetime64_any_dtype(points[time_col])
                    half_w = None
                    
                    if isinstance(anomaly_width, str):
                        half_w = pd.to_timedelta(anomaly_width) / 2
                    elif isinstance(anomaly_width, pd.Timedelta):
                        half_w = anomaly_width / 2
                    elif isinstance(anomaly_width, (int, float)):
                        if is_datetime:
                            half_w = pd.to_timedelta(days=anomaly_width) / 2
                        else:
                            half_w = anomaly_width / 2
                    
                    if half_w is not None:
                        for t in points[time_col]:
                            start = t - half_w
                            end = t + half_w
                            axes[i].axvspan(start, end, color='red', alpha=0.2, zorder=0)

                axes[i].scatter(
                    points[time_col], points[target_col],
                    color='red', s=marker_size, marker='o',
                    edgecolors='black', linewidths=1, label='Anomaly', zorder=100
                )

    # ==========================================
    # === УДАЛЕНИЕ СЛУЖЕБНЫХ КОЛОНОК ===
    # ==========================================
    forecasts_clean = forecasts_filtered
    if forecasts_clean is not None and active_cutoff_col is not None:
        if active_cutoff_col in forecasts_clean.columns:
            forecasts_clean = forecasts_clean.drop(columns=[active_cutoff_col])
            
    df_clean = df_filtered
    if df_clean is not None and active_cutoff_col is not None:
        if active_cutoff_col in df_clean.columns:
            df_clean = df_clean.drop(columns=[active_cutoff_col])

    # ==========================================
    # === ОТРИСОВКА ВТОРОЙ ОСИ (ДО uf_plot_series) ===
    # ==========================================
    ax2_list = []  # Сохраняем ссылки на вторые оси для легенды
    
    if second_axis is not None and forecasts_clean is not None:
        for i, uid in enumerate(plotting_order):
            ax_main = axes[i]
            ax2 = ax_main.twinx()  # Создаём вторую ось
            
            # Данные для текущего ID
            data_2nd = forecasts_clean[forecasts_clean[id_col] == uid]
            
            if not data_2nd.empty:
                ax2.plot(
                    data_2nd[time_col], 
                    data_2nd[second_axis],
                    color=second_axis_color,
                    linewidth=2,
                    linestyle='--',
                    label=second_axis,
                    alpha=0.8
                )
                
                # Подпись оси
                label = second_axis_label or second_axis
                ax2.set_ylabel(label, color=second_axis_color)
                ax2.tick_params(axis='y', labelcolor=second_axis_color)
                
                # Убираем сетку второй оси, чтобы не мешала
                ax2.grid(False)
            
            ax2_list.append(ax2)
            
            # Для последних графиков в ряду скрываем правую ось, 
            # чтобы не было наложения (опционально)
            if (i + 1) % n_cols != 0 and (i + 1) < n_plots:
                ax2.set_yticklabels([])

    # ==========================================
    # === ОСНОВНАЯ ОТРИСОВКА ЧЕРЕЗ uf_plot_series ===
    # ==========================================
    # Убираем second_axis из forecasts_clean, чтобы uf_plot_series не ругался
    forecasts_for_main = forecasts_clean
    if second_axis is not None and forecasts_for_main is not None:
        if second_axis in forecasts_for_main.columns:
            forecasts_for_main = forecasts_for_main.drop(columns=[second_axis])

    result = uf_plot_series(
        df=df_clean,
        forecasts_df=forecasts_for_main,
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
        ax=axes[:n_plots],
    )
    
    # ==========================================
    # === ПОСТ-ОБРАБОТКА: ЛЕГЕНДА ===
    # ==========================================
    # Если нужно объединить легенды с второй осью
    if second_axis is not None and ax2_list:
        for i, ax_main in enumerate(axes[:n_plots]):
            if i < len(ax2_list):
                ax2 = ax2_list[i]
                # Получаем линии и подписи с обоих осей
                lines1, labels1 = ax_main.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                # Объединяем
                ax_main.legend(lines1 + lines2, labels1 + labels2, 
                             loc='upper left', bbox_to_anchor=(1.02, 1))

    return result

# ==========================================
def evaluate_and_plot(df_train, 
                      df_test, 
                      forecasts_or_eval, 
                      metrics, 
                      levels=None, 
                      model_names=None, 
                      plot=True, 
                      modeh=True):
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