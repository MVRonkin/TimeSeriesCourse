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