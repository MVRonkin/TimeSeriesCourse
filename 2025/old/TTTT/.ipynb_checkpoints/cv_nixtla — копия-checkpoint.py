
import pandas as pd

from statsforecast import StatsForecast

from utilsforecast.evaluation import evaluate

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from utilsforecast.evaluation import evaluate
from typing import List, Callable, Optional

def make_cv_splits(
    df, 
    h, 
    step_size, 
    train_window=None, 
    strategy='expanding',
    n_windows_per_train=1,
    test_step=None  # ← НОВЫЙ ПАРАМЕТР
):
    """
    Генерирует CV-окна.
    
    Новые параметры:
    - n_windows_per_train: сколько тестовых окон на каждую позицию обучения.
    - test_step: шаг между тестовыми окнами (в шагах временной шкалы).
                 Если None → равен h (тесты идут подряд без пропусков).
    """
    if test_step is None:
        test_step = h
    
    df = df.sort_values(['unique_id', 'ds'])
    all_splits = []
    
    for uid, group in df.groupby('unique_id'):
        dates = group['ds'].tolist()
        n = len(dates)
        
        if strategy == 'backtest':
            if train_window is None:
                raise ValueError("Для backtest нужен train_window")
            train_start_idx = 0
            train_end_idx = train_window - 1
            
            # Первая позиция теста
            base_test_start = train_end_idx + 1
            pos = base_test_start
            
            while pos + h - 1 < n:
                for w in range(n_windows_per_train):
                    test_start = pos + w * test_step
                    test_end = test_start + h - 1
                    if test_end >= n:
                        break
                    all_splits.append({
                        'unique_id': uid,
                        'train_start': dates[train_start_idx],
                        'train_end': dates[train_end_idx],
                        'test_start': dates[test_start],
                        'test_end': dates[test_end]
                    })
                # Сдвигаемся к следующей позиции обучения
                pos += step_size
                
        else:
            # expanding / sliding
            # Последняя возможная позиция cutoff'а
            last_possible = n - 1 - (n_windows_per_train - 1) * test_step - h
            if last_possible < 0:
                continue
                
            cutoff_positions = []
            pos = last_possible
            min_train_len = train_window if strategy == 'sliding' else 1
            
            while pos >= min_train_len - 1:
                cutoff_positions.append(pos)
                pos -= step_size
            cutoff_positions = sorted(cutoff_positions)
            
            for cutoff in cutoff_positions:
                if strategy == 'expanding':
                    train_start_idx = 0
                else:  # sliding
                    train_start_idx = max(0, cutoff - train_window + 1)
                
                for w in range(n_windows_per_train):
                    test_start = cutoff + 1 + w * test_step
                    test_end = test_start + h - 1
                    if test_end >= n:
                        break
                    all_splits.append({
                        'unique_id': uid,
                        'train_start': dates[train_start_idx],
                        'train_end': dates[cutoff],
                        'test_start': dates[test_start],
                        'test_end': dates[test_end]
                    })
    
    return pd.DataFrame(all_splits)


    

def plot_cv_splits(df, splits, unique_id, title="CV Splits"):
    """
    Визуализация CV-окон линиями.
    
    Параметры:
    - df: исходный DataFrame с ['ds', 'y']
    - splits: результат make_cv_splits
    - unique_id: строка
    - title: заголовок
    """
    series = df[df['unique_id'] == unique_id].sort_values('ds')
    windows = splits[splits['unique_id'] == unique_id]
    
    plt.figure(figsize=(10, 2 + len(windows) * 0.5))
    
    for i, (_, row) in enumerate(windows.iterrows()):
        y_level = i + 1
        
        # Обучение
        plt.plot([row['train_start'], row['train_end']], [y_level, y_level],
                 color='C0', linewidth=2, solid_capstyle='butt')
        # Тест
        plt.plot([row['test_start'], row['test_end']], [y_level, y_level],
                 color='C1', linestyle='--', linewidth=2, solid_capstyle='butt')
        # Граница
        plt.scatter([row['train_end']], [y_level], color='black', s=20, zorder=5)
    
    plt.yticks(range(1, len(windows) + 1), [f"Окно {i+1}" for i in range(len(windows))])
    plt.xlabel('Дата')
    plt.title(f"{title}: {unique_id}")
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

<<<<<<< HEAD

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
=======
>>>>>>> a2b734ea8c8c21003808f711d374b698f9cff950




def cv_evaluation(
    df: pd.DataFrame,
    cv_splits: pd.DataFrame,
    sf: StatsForecast,
    metrics: List[Callable],
    levels: Optional[List[int]] = None,
    id_col: str = 'unique_id',
    time_col: str = 'ds',
    target_col: str = 'y',
    aggregate: bool = True,
    refit: bool = True,
):
    """
    Корректная кросс-валидация для StatsForecast с поддержкой экзогенных переменных.

    ВАЖНО:
    - refit=True  → честный CV (fit + predict на каждом сплите)
    - refit=False → быстрый режим (forecast), без сохранения состояния

    Parameters
    ----------
    df : pd.DataFrame
        Данные в long-формате.
        Все колонки кроме id_col, time_col, target_col
        считаются экзогенными и автоматически передаются модели.
    cv_splits : pd.DataFrame
        Сплиты с колонками:
        [id_col, 'train_start', 'train_end', 'test_start', 'test_end']
    sf : StatsForecast
        Инициализированный объект StatsForecast (модели заданы заранее)
    metrics : list
        Метрики из utilsforecast.evaluation
    levels : list[int], optional
        Уровни prediction intervals (например, [80, 95])
    id_col, time_col, target_col : str
        Имена колонок
    aggregate : bool
        Агрегировать метрики по рядам (mean)
    refit : bool
        Переобучать модели на каждом сплите

    Returns
    -------
    eval_df : pd.DataFrame
        Факт + прогнозы для всех CV-окон
    metrics_df : pd.DataFrame
        Результаты оценки
    """

    # === Валидация входных данных ===
    required = {id_col, time_col, target_col}
    if not required.issubset(df.columns):
        raise ValueError(f"df must contain columns: {required}")

    split_cols = {id_col, 'train_start', 'train_end', 'test_start', 'test_end'}
    if not split_cols.issubset(cv_splits.columns):
        raise ValueError(f"cv_splits must contain columns: {split_cols}")

    # === Вспомогательные функции ===
    def get_train_test(df, split):
        mask = df[id_col] == split[id_col]

        train = df[
            mask &
            (df[time_col] >= split['train_start']) &
            (df[time_col] <= split['train_end'])
        ].copy()

        test = df[
            mask &
            (df[time_col] >= split['test_start']) &
            (df[time_col] <= split['test_end'])
        ].copy()

        return (
            train if len(train) > 0 else None,
            test if len(test) > 0 else None
        )

    def rename_for_sf(df):
        """
        Переименовываем ТОЛЬКО ключевые колонки,
        все остальные (экзогены) сохраняем.
        """
        return df.rename(
            columns={
                id_col: 'unique_id',
                time_col: 'ds',
                target_col: 'y'
            }
        )

    # === Основной цикл CV ===
    all_evals = []
    all_metrics = []

    # Имена моделей (используются evaluate)
    model_names = [m.__class__.__name__ for m in sf.models]

    for _, split in cv_splits.iterrows():
        train, test = get_train_test(df, split)

        if train is None or test is None or len(test) == 0:
            continue

        train_sf = rename_for_sf(train)
        h = len(test)

        # === КЛЮЧЕВОЙ МОМЕНТ: refit ===
        if refit:
            # Честный CV
            sf.fit(train_sf)
            fcst = sf.predict(h=h, level=levels)
        else:
            # Быстрый режим (без сохранения состояния)
            fcst = sf.forecast(df=train_sf, h=h, level=levels)

        fcst['cutoff'] = split['train_end']

        # Возвращаем оригинальные имена колонок
        fcst = fcst.rename(
            columns={
                'unique_id': id_col,
                'ds': time_col
            }
        )

        # === Объединяем прогноз с фактом ===
        eval_row = test.merge(
            fcst,
            on=[id_col, time_col],
            how='inner'
        )

        all_evals.append(eval_row)

        # === Оценка метрик ===
        eval_renamed = rename_for_sf(eval_row)
        train_renamed = rename_for_sf(train)

        metrics_res = evaluate(
            df=eval_renamed,
            metrics=metrics,
            models=model_names,
            train_df=train_renamed,
            level=levels
        )

        metrics_res[id_col] = split[id_col]
        metrics_res['cutoff'] = split['train_end']

        all_metrics.append(metrics_res)

    if not all_evals:
        raise ValueError("No valid CV splits produced forecasts")

    # === Сборка результатов ===
    eval_df = pd.concat(all_evals, ignore_index=True)
    metrics_df = pd.concat(all_metrics, ignore_index=True)

    if aggregate:
        metrics_df = (
            metrics_df
            .groupby(['metric', id_col])[model_names]
            .mean()
            .reset_index()
        )

    return eval_df, metrics_df


def plot_cv_metrics_boxplots(eval_df, metrics=None, models=None, figsize_per_row=(5,2), macro=False):
    """
    Boxplots по выбранным моделям для каждого временного ряда и каждой метрики.
    eval_df: DataFrame вида unique_id, cutoff, metric, <model1>, <model2>, ...
    metrics: список метрик для отображения
    models: список моделей для отображения
    macro: если True — строим агрегированный график по всем рядам
    """
    # Метрики
    if metrics is None:
        metrics = eval_df['metric'].unique()
    
    # Модели
    all_model_cols = eval_df.columns.difference(['unique_id','cutoff','metric'])
    if models is None:
        models = all_model_cols
    else:
        models = [m for m in models if m in all_model_cols]
    
    if macro:
        n_rows = 1
        n_cols = len(metrics)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_row[0]*n_cols, figsize_per_row[1]))
        axes = [axes] if n_cols==1 else axes
        for j, metric in enumerate(metrics):
            ax = axes[j]
            df_plot = eval_df[eval_df['metric']==metric]
            df_melted = df_plot.melt(id_vars=['cutoff','unique_id'], 
                                     value_vars=models,
                                     var_name='model', value_name='error')
            sns.boxplot(data=df_melted, x='model', y='error', ax=ax)
            ax.set_title(f'Macro - {metric.upper()}')
            ax.set_ylabel('Error')
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            ax.set_xlabel('')
        plt.tight_layout()
        plt.show()
        return
    
    # Микро (по рядам)
    unique_ids = eval_df['unique_id'].unique()
    n_rows = len(unique_ids)
    n_cols = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=(figsize_per_row[0]*n_cols, figsize_per_row[1]*n_rows),
                             squeeze=False)
    
    for i, uid in enumerate(unique_ids):
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            df_plot = eval_df[(eval_df['unique_id']==uid) & (eval_df['metric']==metric)]
            if df_plot.empty:
                continue
            df_melted = df_plot.melt(id_vars=['cutoff'], value_vars=models,
                                     var_name='model', value_name='error')
            sns.boxplot(data=df_melted, x='model', y='error', ax=ax)
            if i==0:
                ax.set_title(metric.upper())
            if j==0:
                ax.set_ylabel(uid)
            else:
                ax.set_ylabel('')
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            ax.set_xlabel('')
    
    plt.tight_layout()
    plt.show()

<<<<<<< HEAD
def bootstrap_rank_biserial_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: Optional[int] = None
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for rank-biserial correlation.

    Returns:
    --------
    (ci_low, ci_high)
    """
    rng = np.random.default_rng(random_state)

    idx = np.arange(len(x))
    boot_stats = []

    for _ in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        r_boot = rank_biserial_from_pairs(x[sample_idx], y[sample_idx])
        boot_stats.append(r_boot)

    alpha = 1 - ci
    return (
        np.quantile(boot_stats, alpha / 2),
        np.quantile(boot_stats, 1 - alpha / 2),
    )
    

def cv_model_stats(
    df: pd.DataFrame,
    metric_name: Optional[List[str]] = None,
    metric_col: str = 'metric',
    model_cols: Optional[List[str]] = None,
    block_cols: List[str] = ['unique_id', 'cutoff'],
    lower_is_better: bool = True,
    alpha: float = 0.05,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Statistical comparison of forecasting models on CV / backtesting.

    Pipeline:
    ---------
    1. Friedman test (global difference)
    2. Kendall’s W (global effect size)
    3. Average ranks
    4. Pairwise Wilcoxon signed-rank tests
    5. Holm correction for multiple comparisons
    6. Rank-biserial correlation (pairwise effect size)

    Returns:
    --------
    dict[metric] -> {
        'friedman_stat'      : float,
        'p_value'            : float,
        'kendall_w'          : float,
        'significant'        : bool,
        'ranks_df'           : DataFrame,
        'best_model'         : str,
        'pairwise_wilcoxon'  : DataFrame,
        'n_blocks'           : int
    }
    """

    # -------- метрики --------
    if metric_name is None:
        metrics = sorted(df[metric_col].unique())
    elif isinstance(metric_name, str):
        metrics = [metric_name]
    else:
        metrics = metric_name

    # -------- модели --------
    if model_cols is None:
        reserved = set(block_cols + [metric_col, 'ds', 'y'])
        model_cols = [c for c in df.columns if c not in reserved]

    if len(model_cols) < 3:
        raise ValueError("Friedman test requires at least 3 models")

    results = {}

    for m in metrics:
        df_m = df[df[metric_col] == m]

        # -------- blocks × models --------
        matrix = []
        for _, block in df_m.groupby(block_cols):
            row = block[model_cols].iloc[0]
            if row.isna().any():
                continue
            matrix.append(row.values)

        matrix = np.asarray(matrix)

        if matrix.shape[0] < 3:
            if verbose:
                print(f"⚠ Метрика '{m}' пропущена: блоков < 3")
            continue

        n_blocks, n_models = matrix.shape

        # -------- Friedman --------
        stat, p_val = friedmanchisquare(
            *[matrix[:, i] for i in range(n_models)]
        )

        # -------- Kendall’s W --------
        kendall_w = stat / (n_blocks * (n_models - 1))

        # -------- ranks --------
        ranks = np.vstack([
            pd.Series(row).rank(
                ascending=lower_is_better,
                method='average'
            ).values
            for row in matrix
        ])

        ranks_df = (
            pd.DataFrame({
                'model': model_cols,
                'avg_rank': ranks.mean(axis=0),
                'std_rank': ranks.std(axis=0)
            })
            .sort_values('avg_rank')
            .reset_index(drop=True)
        )

        # -------- pairwise Wilcoxon + Holm --------
        pairwise_df = None

        if p_val < alpha:
            rows = []

            for i in range(n_models):
                for j in range(i + 1, n_models):
                    x, y = matrix[:, i], matrix[:, j]
            
                    try:
                        _, p_raw = wilcoxon(x, y)
                        r = rank_biserial_from_pairs(x, y)
                        ci_low, ci_high = bootstrap_rank_biserial_ci(x, y)
                    except ValueError:
                        p_raw, r = np.nan, np.nan
                        ci_low, ci_high = np.nan, np.nan
            
                    rows.append({
                        'model_1': model_cols[i],
                        'model_2': model_cols[j],
                        'p_value_raw': p_raw,
                        'effect_r': r,
                        'ci_low': ci_low,
                        'ci_high': ci_high,
                        'abs_effect_r': abs(r)
                    })

            pairwise_df = pd.DataFrame(rows)

            _, pvals_holm, _, _ = multipletests(
                pairwise_df['p_value_raw'],
                alpha=alpha,
                method='holm'
            )
            pairwise_df['p_value_holm'] = pvals_holm

            pairwise_df['ci_excludes_zero'] = ~(
                (pairwise_df['ci_low'] <= 0) &
                (pairwise_df['ci_high'] >= 0)
            )
            
            pairwise_df['significant'] = pairwise_df['p_value_holm'] < alpha

            pairwise_df = pairwise_df.sort_values(
                ['significant', 'abs_effect_r'],
                ascending=[False, False]
            )
        pairwise_df = pairwise_df[[
                    'model_1', 'model_2',
                    'p_value_raw', 'p_value_holm',
                    'abs_effect_r',  'significant'
                ]].rename(columns={'abs_effect_r': 'r',
                                   'p_value_raw':'p_wilcoxon',
                                   'p_value_holm':'p_holm',
                                   # 'significant':'H1'
                                  })
                
                # Сортировка
        pairwise_df = pairwise_df.sort_values(
                    ['significant', 'r'],
                    ascending=[False, False]
                ).reset_index(drop=True)
        # -------- save --------
        results[m] = {
            'friedman_stat': stat,
            'p_value': p_val,
            'kendall_w': kendall_w,
            'significant': p_val < alpha,
            'ranks_df': ranks_df,
            'best_model': ranks_df.iloc[0]['model'],
            'pairwise_wilcoxon': pairwise_df,
            'n_blocks': n_blocks
        }

        # -------- verbose --------
        if verbose:
            print(f"\n{'='*70}")
            print(f"Метрика: {m.upper()} | блоков: {n_blocks}")
            print('='*70)
            print(f"Friedman χ² = {stat:.4f}")
            print(f"p-value     = {p_val:.3e}")
            print(f"Kendall’s W = {kendall_w:.3f}")
            print('-'*70)
            print("Средние ранги:")
            display(ranks_df.round(3))

            if p_val < alpha and pairwise_df is not None:
                print('-'*70)
                print("Pairwise Wilcoxon + Holm (+ effect size r):")
                display(pairwise_df.round(3))
            else:
                print('-'*70)
                print("✗ Значимых различий нет")

    return results


    
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
=======
# def make_cv_splits(
#     df, 
#     h, 
#     step_size, 
#     train_window=None, 
#     strategy='expanding',
#     n_windows_per_train=1,
#     test_step=None  # ← НОВЫЙ ПАРАМЕТР
>>>>>>> a2b734ea8c8c21003808f711d374b698f9cff950
# ):
#     """
#     Генерирует CV-окна.
    
#     Новые параметры:
#     - n_windows_per_train: сколько тестовых окон на каждую позицию обучения.
#     - test_step: шаг между тестовыми окнами (в шагах временной шкалы).
#                  Если None → равен h (тесты идут подряд без пропусков).
#     """
#     if test_step is None:
#         test_step = h
    
#     df = df.sort_values(['unique_id', 'ds'])
#     all_splits = []
    
#     for uid, group in df.groupby('unique_id'):
#         dates = group['ds'].tolist()
#         n = len(dates)
        
#         if strategy == 'backtest':
#             if train_window is None:
#                 raise ValueError("Для backtest нужен train_window")
#             train_start_idx = 0
#             train_end_idx = train_window - 1
            
#             # Первая позиция теста
#             base_test_start = train_end_idx + 1
#             pos = base_test_start
            
#             while pos + h - 1 < n:
#                 for w in range(n_windows_per_train):
#                     test_start = pos + w * test_step
#                     test_end = test_start + h - 1
#                     if test_end >= n:
#                         break
#                     all_splits.append({
#                         'unique_id': uid,
#                         'train_start': dates[train_start_idx],
#                         'train_end': dates[train_end_idx],
#                         'test_start': dates[test_start],
#                         'test_end': dates[test_end]
#                     })
#                 # Сдвигаемся к следующей позиции обучения
#                 pos += step_size
                
#         else:
#             # expanding / sliding
#             # Последняя возможная позиция cutoff'а
#             last_possible = n - 1 - (n_windows_per_train - 1) * test_step - h
#             if last_possible < 0:
#                 continue
                
#             cutoff_positions = []
#             pos = last_possible
#             min_train_len = train_window if strategy == 'sliding' else 1
            
#             while pos >= min_train_len - 1:
#                 cutoff_positions.append(pos)
#                 pos -= step_size
#             cutoff_positions = sorted(cutoff_positions)
            
#             for cutoff in cutoff_positions:
#                 if strategy == 'expanding':
#                     train_start_idx = 0
#                 else:  # sliding
#                     train_start_idx = max(0, cutoff - train_window + 1)
                
#                 for w in range(n_windows_per_train):
#                     test_start = cutoff + 1 + w * test_step
#                     test_end = test_start + h - 1
#                     if test_end >= n:
#                         break
#                     all_splits.append({
#                         'unique_id': uid,
#                         'train_start': dates[train_start_idx],
#                         'train_end': dates[cutoff],
#                         'test_start': dates[test_start],
#                         'test_end': dates[test_end]
#                     })
    
#     return pd.DataFrame(all_splits)

# def plot_cv_splits(df, splits, unique_id, title="CV Splits"):
#     """
#     Визуализация CV-окон линиями.
    
#     Параметры:
#     - df: исходный DataFrame с ['ds', 'y']
#     - splits: результат make_cv_splits
#     - unique_id: строка
#     - title: заголовок
#     """
#     series = df[df['unique_id'] == unique_id].sort_values('ds')
#     windows = splits[splits['unique_id'] == unique_id]
    
#     plt.figure(figsize=(10, 2 + len(windows) * 0.5))
    
#     for i, (_, row) in enumerate(windows.iterrows()):
#         y_level = i + 1
        
#         # Обучение
#         plt.plot([row['train_start'], row['train_end']], [y_level, y_level],
#                  color='C0', linewidth=2, solid_capstyle='butt')
#         # Тест
#         plt.plot([row['test_start'], row['test_end']], [y_level, y_level],
#                  color='C1', linestyle='--', linewidth=2, solid_capstyle='butt')
#         # Граница
#         plt.scatter([row['train_end']], [y_level], color='black', s=20, zorder=5)
    
#     plt.yticks(range(1, len(windows) + 1), [f"Окно {i+1}" for i in range(len(windows))])
#     plt.xlabel('Дата')
#     plt.title(f"{title}: {unique_id}")
#     plt.grid(True, axis='x', linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.show()    



# def cv_evaluation_with_forecasts(
#     df, 
#     splits, 
#     models, 
#     model_names,
#     metrics,
#     refit=True,
#     freq=FREQ
# ):
#     all_evals = []
#     all_forecasts = []  # ← НОВОЕ
#     sf_cache = {}
    
#     grouped = splits.groupby(['unique_id', 'train_start', 'train_end'])
    
#     for (uid, train_start, train_end), group in grouped:
#         test_rows = []
#         test_info = []  # чтобы знать, откуда какие даты
#         for _, row in group.iterrows():
#             mask = (
#                 (df['unique_id'] == uid) &
#                 (df['ds'] >= row['test_start']) &
#                 (df['ds'] <= row['test_end'])
#             )
#             subset = df[mask]
#             test_rows.append(subset)
#             test_info.append(row)
        
#         if not test_rows:
#             continue
            
#         test = pd.concat(test_rows).sort_values('ds').reset_index(drop=True)
#         train = df[
#             (df['unique_id'] == uid) &
#             (df['ds'] >= train_start) &
#             (df['ds'] <= train_end)
#         ].copy()
        
#         if len(train) == 0 or len(test) == 0:
#             continue
        
#         pred_cols = {'ds': test['ds'].values, 'unique_id': uid}
#         for model, name in zip(models, model_names):
#             cache_key = (name, uid, train_start, train_end)
#             if refit or cache_key not in sf_cache:
#                 sf = StatsForecast(models=[model], freq=freq)
#                 sf.fit(train)
#                 if not refit:
#                     sf_cache[cache_key] = sf
#             else:
#                 sf = sf_cache[cache_key]
            
#             h = len(test)
#             pred = sf.predict(h=h)
#             pred_cols[name] = pred[name].values[:h]
        
#         # Сохраняем прогнозы
#         forecast_df = pd.DataFrame(pred_cols)
#         forecast_df['train_start'] = train_start
#         forecast_df['train_end'] = train_end
#         all_forecasts.append(forecast_df)
        
#         # Оценка
#         eval_df = test[['unique_id', 'ds', 'y']].copy()
#         for name in model_names:
#             eval_df[name] = pred_cols[name]
        
#         eval_result = evaluate(df=eval_df, metrics=metrics, train_df=train)
#         eval_result['train_start'] = train_start
#         eval_result['train_end'] = train_end
#         eval_result['test_start'] = test['ds'].min()
#         eval_result['test_end'] = test['ds'].max()
#         all_evals.append(eval_result)
    
#     return pd.concat(all_evals, ignore_index=True), pd.concat(all_forecasts, ignore_index=True)
    
# # def plot_cv_splits(df, splits, unique_id, title="CV Splits"):
# #     """
# #     Визуализация CV-окон линиями.
    
# #     Параметры:
# #     - df: исходный DataFrame с ['ds', 'y']
# #     - splits: результат make_cv_splits
# #     - unique_id: строка
# #     - title: заголовок
# #     """
# #     series = df[df['unique_id'] == unique_id].sort_values('ds')
# #     windows = splits[splits['unique_id'] == unique_id]
    
# #     plt.figure(figsize=(10, 2 + len(windows) * 0.5))
    
# #     for i, (_, row) in enumerate(windows.iterrows()):
# #         y_level = i + 1
        
# #         # Обучение
# #         plt.plot([row['train_start'], row['train_end']], [y_level, y_level],
# #                  color='C0', linewidth=2, solid_capstyle='butt')
# #         # Тест
# #         plt.plot([row['test_start'], row['test_end']], [y_level, y_level],
# #                  color='C1', linestyle='--', linewidth=2, solid_capstyle='butt')
# #         # Граница
# #         plt.scatter([row['train_end']], [y_level], color='black', s=20, zorder=5)
    
# #     plt.yticks(range(1, len(windows) + 1), [f"Окно {i+1}" for i in range(len(windows))])
# #     plt.xlabel('Дата')
# #     plt.title(f"{title}: {unique_id}")
# #     plt.grid(True, axis='x', linestyle='--', alpha=0.5)
# #     plt.tight_layout()
# #     plt.show()

# def get_train_test_from_split(
#     df,
#     split,
#     id_col='unique_id',
#     time_col='ds'
# ):
#     mask_uid = df[id_col] == split[id_col]

#     df_train = df.loc[
#         mask_uid &
#         (df[time_col] >= split['train_start']) &
#         (df[time_col] <= split['train_end'])
#     ].copy()

#     df_test = df.loc[
#         mask_uid &
#         (df[time_col] >= split['test_start']) &
#         (df[time_col] <= split['test_end'])
#     ].copy()

#     if df_train.empty or df_test.empty:
#         return None, None

#     return df_train, df_test


# def forecast_one_split(
#     sf,
#     df,
#     split,
#     level=None
# ):
#     df_train, df_test = get_train_test_from_split(df, split)
    
#     h = len(df_test)  # ← ЕДИНСТВЕННЫЙ источник истины
    
#     if h == 0:
#         raise ValueError("Empty test window in CV split.")
    
#     fcst = sf.forecast(
#         df=df_train,
#         h=h,
#         level=level
#     )
    
#     fcst['cutoff'] = split['train_end']
    
#     return df_train, df_test, fcst



# def cv_evaluation(
#     df,
#     cv_splits,
#     sf,
#     metrics,
#     levels=None,
#     aggregate=True,
#     id_col='unique_id',
#     time_col='ds',
#     target_col='y',
# ):
#     """
#     Cross-validation для StatsForecast на пользовательских CV-окнах.
#     Корректно совместима с Nixtla (StatsForecast + utilsforecast).
#     """

#     all_eval = []
#     all_metrics = []

#     for _, split in cv_splits.iterrows():

#         df_train, df_test = get_train_test_from_split(
#             df, split, id_col, time_col
#         )
#         if df_train is None:
#             continue

#         # === StatsForecast формат ===
#         df_train_sf = df_train.rename(columns={
#             id_col: 'unique_id',
#             time_col: 'ds',
#             target_col: 'y'
#         })

#         h = len(df_test)
#         if h == 0:
#             continue

#         # === Прогноз ===
#         fcst = sf.forecast(
#             df=df_train_sf,
#             h=h,
#             level=levels
#         )
#         fcst['cutoff'] = split['train_end']

#         # === Возвращаем исходные имена ===
#         fcst = fcst.rename(columns={
#             'unique_id': id_col,
#             'ds': time_col
#         })

#         eval_df = df_test[[id_col, time_col, target_col]].merge(
#             fcst,
#             on=[id_col, time_col],
#             how='inner'
#         )
#         all_eval.append(eval_df)

#         # === evaluate требует локальный train_df ===
#         eval_renamed = eval_df.rename(columns={
#             id_col: 'unique_id',
#             time_col: 'ds',
#             target_col: 'y'
#         })
#         train_renamed = df_train.rename(columns={
#             id_col: 'unique_id',
#             time_col: 'ds',
#             target_col: 'y'
#         })

#         model_names = [
#             c for c in eval_renamed.columns
#             if c not in {'unique_id', 'ds', 'y', 'cutoff'}
#             and not c.endswith(tuple(['lo', 'hi']))
#         ]

#         metrics_df = evaluate(
#             df=eval_renamed,
#             train_df=train_renamed,
#             metrics=metrics,
#             models=model_names,
#             level=levels
#         )

#         all_metrics.append(metrics_df)

#     if not all_eval:
#         raise ValueError("CV did not produce any valid forecasts.")

#     eval_df = pd.concat(all_eval, ignore_index=True)
#     metrics_df = pd.concat(all_metrics, ignore_index=True)

#     if aggregate:
#         metrics_df = (
#             metrics_df
#             .groupby(['metric', 'unique_id'])[model_names]
#             .mean()
#         )

#     return eval_df, metrics_df

# # def make_cv_splits(
# #     df, 
# #     h, 
# #     step_size, 
# #     train_window=None, 
# #     strategy='expanding',
# #     n_windows_per_train=1,
# #     test_step=None  # ← НОВЫЙ ПАРАМЕТР
# # ):
# #     """
# #     Генерирует CV-окна.
    
# #     Новые параметры:
# #     - n_windows_per_train: сколько тестовых окон на каждую позицию обучения.
# #     - test_step: шаг между тестовыми окнами (в шагах временной шкалы).
# #                  Если None → равен h (тесты идут подряд без пропусков).
# #     """
# #     if test_step is None:
# #         test_step = h
    
# #     df = df.sort_values(['unique_id', 'ds'])
# #     all_splits = []
    
# #     for uid, group in df.groupby('unique_id'):
# #         dates = group['ds'].tolist()
# #         n = len(dates)
        
# #         if strategy == 'backtest':
# #             if train_window is None:
# #                 raise ValueError("Для backtest нужен train_window")
# #             train_start_idx = 0
# #             train_end_idx = train_window - 1
            
# #             # Первая позиция теста
# #             base_test_start = train_end_idx + 1
# #             pos = base_test_start
            
# #             while pos + h - 1 < n:
# #                 for w in range(n_windows_per_train):
# #                     test_start = pos + w * test_step
# #                     test_end = test_start + h - 1
# #                     if test_end >= n:
# #                         break
# #                     all_splits.append({
# #                         'unique_id': uid,
# #                         'train_start': dates[train_start_idx],
# #                         'train_end': dates[train_end_idx],
# #                         'test_start': dates[test_start],
# #                         'test_end': dates[test_end]
# #                     })
# #                 # Сдвигаемся к следующей позиции обучения
# #                 pos += step_size
                
# #         else:
# #             # expanding / sliding
# #             # Последняя возможная позиция cutoff'а
# #             last_possible = n - 1 - (n_windows_per_train - 1) * test_step - h
# #             if last_possible < 0:
# #                 continue
                
# #             cutoff_positions = []
# #             pos = last_possible
# #             min_train_len = train_window if strategy == 'sliding' else 1
            
# #             while pos >= min_train_len - 1:
# #                 cutoff_positions.append(pos)
# #                 pos -= step_size
# #             cutoff_positions = sorted(cutoff_positions)
            
# #             for cutoff in cutoff_positions:
# #                 if strategy == 'expanding':
# #                     train_start_idx = 0
# #                 else:  # sliding
# #                     train_start_idx = max(0, cutoff - train_window + 1)
                
# #                 for w in range(n_windows_per_train):
# #                     test_start = cutoff + 1 + w * test_step
# #                     test_end = test_start + h - 1
# #                     if test_end >= n:
# #                         break
# #                     all_splits.append({
# #                         'unique_id': uid,
# #                         'train_start': dates[train_start_idx],
# #                         'train_end': dates[cutoff],
# #                         'test_start': dates[test_start],
# #                         'test_end': dates[test_end]
# #                     })
    
# #     return pd.DataFrame(all_splits)
    
    
# # # import pandas as pd
# # # import numpy as np
# # # from joblib import Parallel, delayed


# # # def make_cv_splits(
# # #     df,
# # #     h,
# # #     step_size,
# # #     train_window=None,
# # #     strategy='expanding',
# # #     n_windows_per_train=1,
# # #     test_step=None,
# # #     freq=None,
# # #     id_col='unique_id',
# # #     time_col='ds',
# # #     target_col='y',
# # #     min_data_length=10,
# # #     cutoffs_df=None,
# # #     n_jobs=1,
# # # ):
# # #     """
# # #     Генерирует CV-окна для временных рядов в формате Nixtla (long format).
    
# # #     Поддерживает:
# # #       - Любые колонки через id_col/time_col/target_col
# # #       - Экзогенные переменные (игнорируются, но сохраняется структура)
# # #       - Готовые cutoff'ы из utilsforecast
# # #       - Параллелизацию по уникальным id
# # #       - Минимальную длину ряда
    
# # #     Parameters
# # #     ----------
# # #     df : pd.DataFrame
# # #         Данные в long-формате.
# # #     h : int
# # #         Горизонт прогноза (в шагах временной шкалы).
# # #     step_size : int or pd.Timedelta
# # #         Шаг между cutoff'ами.
# # #     train_window : int, optional
# # #         Размер обучающего окна. Обязателен для 'sliding' и 'fixed_train'.
# # #     strategy : str, default='expanding'
# # #         Одна из: 'expanding', 'sliding', 'fixed_train'.
# # #     n_windows_per_train : int, default=1
# # #         Сколько тестовых окон на один cutoff.
# # #     test_step : int or pd.Timedelta, optional
# # #         Шаг между тестовыми окнами. Если None → равен h.
# # #     freq : str or pd.Timedelta, optional
# # #         Частота временной шкалы (для datetime ds).
# # #     id_col : str, default='unique_id'
# # #         Колонка с идентификатором серии.
# # #     time_col : str, default='ds'
# # #         Колонка со временем.
# # #     target_col : str, default='y'
# # #         Колонка с целевой переменной (не используется, но проверяется наличие).
# # #     min_data_length : int, default=10
# # #         Минимальная длина ряда для обработки.
# # #     cutoffs_df : pd.DataFrame, optional
# # #         Готовые cutoff'ы с колонками [id_col, 'cutoff'].
# # #         Если задан — игнорирует step_size и генерирует окна только для этих cutoff'ов.
# # #     n_jobs : int, default=1
# # #         Число параллельных процессов. -1 = все ядра.
    
# # #     Returns
# # #     -------
# # #     pd.DataFrame
# # #         Таблица с колонками:
# # #             [id_col, 'train_start', 'train_end', 'test_start', 'test_end']
    
# # #     Notes
# # #     -----
# # #     **Экзогенные переменные**: 
# # #         Функция не использует их напрямую, но при использовании cutoffs_df
# # #         вы можете передать дополнительные колонки (например, 'X_cols'),
# # #         которые будут сохранены в выходном датафрейме.
        
# # #         Рекомендуемый workflow:
# # #           1. Сгенерировать cutoff'ы с помощью этой функции
# # #           2. Для каждого cutoff'а извлечь train/test данные с экзогенными переменными
# # #           3. Обучить модель на train, оценить на test
# # #     """
# # #     # === Валидация входных данных ===
# # #     required_cols = {id_col, time_col}
# # #     if not required_cols.issubset(df.columns):
# # #         raise ValueError(f"df must contain columns: {required_cols}")
    
# # #     allowed_strategies = {'expanding', 'sliding', 'fixed_train'}
# # #     if strategy not in allowed_strategies:
# # #         raise ValueError(
# # #             f"strategy must be one of: {sorted(allowed_strategies)}. Got '{strategy}'"
# # #         )
    
# # #     if df.empty:
# # #         cols = [id_col, 'train_start', 'train_end', 'test_start', 'test_end']
# # #         return pd.DataFrame(columns=cols)
    
# # #     # === Использование готовых cutoff'ов (если заданы) ===
# # #     if cutoffs_df is not None:
# # #         if 'cutoff' not in cutoffs_df.columns:
# # #             raise ValueError("cutoffs_df must contain 'cutoff' column")
# # #         if id_col not in cutoffs_df.columns:
# # #             raise ValueError(f"cutoffs_df must contain '{id_col}' column")
        
# # #         # Группируем cutoff'ы по id
# # #         cutoff_groups = cutoffs_df.groupby(id_col)['cutoff'].apply(list).to_dict()
# # #         df_for_cutoffs = df.set_index([id_col, time_col])
        
# # #         def process_uid_with_cutoffs(uid):
# # #             if uid not in cutoff_groups:
# # #                 return []
# # #             cutoffs = cutoff_groups[uid]
# # #             if uid not in df_for_cutoffs.index.get_level_values(id_col):
# # #                 return []
            
# # #             series = df_for_cutoffs.xs(uid, level=id_col).sort_index()
# # #             dates = series.index.tolist()
# # #             date_to_idx = {d: i for i, d in enumerate(dates)}
# # #             results = []
            
# # #             for cutoff_date in cutoffs:
# # #                 if cutoff_date not in date_to_idx:
# # #                     continue
# # #                 cutoff_idx = date_to_idx[cutoff_date]
                
# # #                 # Определяем train window
# # #                 if strategy == 'expanding':
# # #                     train_start_idx = 0
# # #                 elif strategy == 'sliding':
# # #                     if train_window is None:
# # #                         raise ValueError("train_window required for 'sliding' with cutoffs")
# # #                     train_start_idx = max(0, cutoff_idx - train_window + 1)
# # #                 else:  # fixed_train
# # #                     if train_window is None:
# # #                         raise ValueError("train_window required for 'fixed_train' with cutoffs")
# # #                     train_start_idx = 0
# # #                     if cutoff_idx != train_window - 1:
# # #                         continue  # пропускаем, если cutoff не соответствует фиксированному окну
                
# # #                 # Генерируем тестовые окна
# # #                 for w in range(n_windows_per_train):
# # #                     test_start_idx = cutoff_idx + 1 + w * (test_step if not isinstance(test_step, pd.Timedelta) else 1)
# # #                     test_end_idx = test_start_idx + h - 1
# # #                     if test_end_idx >= len(dates):
# # #                         break
                    
# # #                     results.append({
# # #                         id_col: uid,
# # #                         'train_start': dates[train_start_idx],
# # #                         'train_end': dates[cutoff_idx],
# # #                         'test_start': dates[test_start_idx],
# # #                         'test_end': dates[test_end_idx]
# # #                     })
# # #             return results
        
# # #         uids = cutoffs_df[id_col].unique()
# # #         if n_jobs == 1:
# # #             all_results = [r for uid in uids for r in process_uid_with_cutoffs(uid)]
# # #         else:
# # #             results_list = Parallel(n_jobs=n_jobs)(
# # #                 delayed(process_uid_with_cutoffs)(uid) for uid in uids
# # #             )
# # #             all_results = [r for sublist in results_list for r in sublist]
        
# # #         return pd.DataFrame(all_results)
    
# # #     # === Автоматическая генерация cutoff'ов ===
# # #     ds_sample = df[time_col].iloc[0]
# # #     is_datetime = pd.api.types.is_datetime64_any_dtype(df[time_col])
    
# # #     # Конвертация step_size и test_step для datetime
# # #     if is_datetime:
# # #         if isinstance(step_size, (int, float)):
# # #             if freq is None:
# # #                 raise ValueError("Для datetime time_col требуется freq или step_size как Timedelta")
# # #             step_size = pd.Timedelta(step_size, unit=freq) if isinstance(freq, str) else step_size * freq
# # #         if test_step is not None and isinstance(test_step, (int, float)):
# # #             if freq is None:
# # #                 raise ValueError("Для datetime time_col требуется freq или test_step как Timedelta")
# # #             test_step = pd.Timedelta(test_step, unit=freq) if isinstance(freq, str) else test_step * freq
# # #         if test_step is None:
# # #             test_step = pd.Timedelta(h, unit=freq) if freq else h
# # #     else:
# # #         if not isinstance(step_size, (int, np.integer)):
# # #             raise ValueError("Для не-datetime time_col step_size должен быть целым числом")
# # #         if test_step is None:
# # #             test_step = h
# # #         elif not isinstance(test_step, (int, np.integer)):
# # #             raise ValueError("Для не-datetime time_col test_step должен быть целым числом")

# # #     df_sorted = df[[id_col, time_col]].sort_values([id_col, time_col]).reset_index(drop=True)
    
# # #     def process_single_series(uid, group):
# # #         dates = group[time_col].tolist()
# # #         n = len(dates)
        
# # #         if n < min_data_length:
# # #             return []
        
# # #         results = []
        
# # #         if strategy == 'fixed_train':
# # #             if train_window is None:
# # #                 raise ValueError("Для fixed_train нужен train_window")
# # #             if train_window > n:
# # #                 return []
            
# # #             train_start_idx = 0
# # #             train_end_idx = train_window - 1
# # #             current_test_start_idx = train_end_idx + 1
            
# # #             while current_test_start_idx + h - 1 < n:
# # #                 for w in range(n_windows_per_train):
# # #                     if is_datetime:
# # #                         expected_test_start = dates[train_end_idx] + (w + 1) * test_step
# # #                         try:
# # #                             test_start_idx = next(i for i, d in enumerate(dates) if d >= expected_test_start)
# # #                         except StopIteration:
# # #                             break
# # #                         test_end_idx = test_start_idx + h - 1
# # #                         if test_end_idx >= n:
# # #                             break
# # #                     else:
# # #                         test_start_idx = current_test_start_idx + w * test_step
# # #                         test_end_idx = test_start_idx + h - 1
# # #                         if test_end_idx >= n:
# # #                             break
                    
# # #                     results.append({
# # #                         id_col: uid,
# # #                         'train_start': dates[train_start_idx],
# # #                         'train_end': dates[train_end_idx],
# # #                         'test_start': dates[test_start_idx],
# # #                         'test_end': dates[test_end_idx]
# # #                     })
                
# # #                 if is_datetime:
# # #                     next_cutoff_date = dates[train_end_idx] + step_size
# # #                     try:
# # #                         current_test_start_idx = next(i for i, d in enumerate(dates) if d >= next_cutoff_date) + 1
# # #                     except StopIteration:
# # #                         break
# # #                 else:
# # #                     current_test_start_idx += step_size
        
# # #         else:
# # #             # expanding / sliding
# # #             min_train_len = train_window if strategy == 'sliding' else 1
# # #             if n < min_train_len + h:
# # #                 return []
            
# # #             test_step_int = test_step if not is_datetime else 1
# # #             last_possible_idx = n - h - (n_windows_per_train - 1) * test_step_int - 1
# # #             if last_possible_idx < min_train_len - 1:
# # #                 return []
            
# # #             cutoff_positions = []
# # #             pos = last_possible_idx
# # #             while pos >= min_train_len - 1:
# # #                 cutoff_positions.append(pos)
# # #                 if is_datetime:
# # #                     target_date = dates[pos] - step_size
# # #                     candidates = [i for i, d in enumerate(dates[:pos+1]) if d <= target_date]
# # #                     if not candidates:
# # #                         break
# # #                     pos = candidates[-1]
# # #                 else:
# # #                     pos -= step_size
# # #                     if pos < min_train_len - 1:
# # #                         break
# # #             cutoff_positions = sorted(cutoff_positions)
            
# # #             for cutoff in cutoff_positions:
# # #                 if strategy == 'expanding':
# # #                     train_start_idx = 0
# # #                 else:  # sliding
# # #                     train_start_idx = max(0, cutoff - train_window + 1)
                
# # #                 for w in range(n_windows_per_train):
# # #                     if is_datetime:
# # #                         expected_test_start = dates[cutoff] + (w + 1) * test_step
# # #                         try:
# # #                             test_start_idx = next(i for i, d in enumerate(dates) if d >= expected_test_start)
# # #                         except StopIteration:
# # #                             break
# # #                         test_end_idx = test_start_idx + h - 1
# # #                         if test_end_idx >= n:
# # #                             break
# # #                     else:
# # #                         test_start_idx = cutoff + 1 + w * test_step
# # #                         test_end_idx = test_start_idx + h - 1
# # #                         if test_end_idx >= n:
# # #                             break
                    
# # #                     results.append({
# # #                         id_col: uid,
# # #                         'train_start': dates[train_start_idx],
# # #                         'train_end': dates[cutoff],
# # #                         'test_start': dates[test_start_idx],
# # #                         'test_end': dates[test_end_idx]
# # #                     })
        
# # #         return results
    
# # #     # === Параллелизация ===
# # #     grouped = list(df_sorted.groupby(id_col))
# # #     if n_jobs == 1:
# # #         all_splits = [r for uid, group in grouped for r in process_single_series(uid, group)]
# # #     else:
# # #         results_list = Parallel(n_jobs=n_jobs)(
# # #             delayed(process_single_series)(uid, group) for uid, group in grouped
# # #         )
# # #         all_splits = [r for sublist in results_list for r in sublist]
    
# # #     return pd.DataFrame(all_splits)
