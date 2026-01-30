import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from typing import Optional, List, Tuple, Union
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tsa.stattools import kpss
import statsmodels.api as sm
def plot_model_diagnostics(
    df_resid, 
    resid_col="resid", 
    title_suffix="", 
    season_length=None,
    figsize=(9, 6)
):
    """
    Визуализация и статистическая диагностика остатков модели для одной временной серии.
    
    Выполняет 4 теста:
        - KPSS: стационарность остатков
        - Ljung-Box: автокорреляция
        - Breusch-Pagan: гетероскедастичность
        - Jarque-Bera: нормальность распределения
    
    Параметры
    ----------
    df_resid : pd.DataFrame
        Датафрейм с колонками ['unique_id', 'ds', resid_col]
    resid_col : str, default='resid'
        Имя колонки с остатками
    title_suffix : str, optional
        Дополнение к заголовку графика
    season_length : int, optional
        Сезонность ряда. Если None — определяется как min(10, T//5)
    figsize : tuple, default=(9, 6)
        Размер фигуры
    
    Возвращает
    ----------
    None
        Отображает график через plt.show()
    """
    uid = df_resid['unique_id'].iloc[0]
    resid = df_resid[resid_col].dropna()
    
    # Защита от коротких рядов
    if len(resid) < 10:
        print(f"Пропущено: {uid} — недостаточно данных (T={len(resid)} < 10)")
        return
    
    T = len(resid)
    
    # Определение лагов
    if season_length is not None:
        lags = min(2 * season_length, T // 5)
    else:
        lags = min(10, T // 5)
    lags = max(lags, 1)
    lags = min(lags, T // 2)  # ← критическое исправление
    
    # Тесты
    lb_pval = acorr_ljungbox(resid, lags=lags, return_df=True)['lb_pvalue'].iloc[-1]
    jb_stat, jb_pval = stats.jarque_bera(resid)
    
    exog = sm.add_constant(np.arange(len(resid)))
    _, bp_pval, _, _ = het_breuschpagan(resid, exog)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # ← критическое исправление: nlags вместо 'auto'
            kpss_stat, kpss_pval, _, _ = kpss(resid, regression='c', nlags=min(12, T//4))
        except Exception:
            kpss_pval = np.nan

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Диагностика остатков: {uid} {title_suffix}', fontsize=12)

    # [0,0] — Остатки + KPSS
    kpss_txt = f"Residuals (KPSS p={kpss_pval:.3f})" if not np.isnan(kpss_pval) else "Residuals"
    axes[0, 0].plot(df_resid['ds'], df_resid[resid_col], linewidth=0.8)
    axes[0, 0].set_title(kpss_txt)
    axes[0, 0].set_ylabel('Остатки')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    # [0,1] — ACF + Ljung-Box
    plot_acf(resid, ax=axes[0, 1], zero=False, auto_ylims=True)
    axes[0, 1].set_title(f'ACF (Ljung-Box p={lb_pval:.3f})')

    # [1,0] — Гистограмма + Breusch-Pagan
    axes[1, 0].hist(resid, bins=15, edgecolor='k', alpha=0.7)
    axes[1, 0].set_xlabel('Остатки')
    axes[1, 0].set_ylabel('Частота')
    axes[1, 0].set_title(f"Histogram (BP p={bp_pval:.3f})")

    # [1,1] — Q-Q + Jarque-Bera
    stats.probplot(resid, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f'Q-Q (JB p={jb_pval:.3f})')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()




def block_bootstrap_forecast(
    in_sample_sf: pd.DataFrame,
    fcst_sf: pd.DataFrame,
    target_col: str = 'y',
    models: Optional[List[str]] = None,
    levels: Tuple[int, ...] = (90, 95),
    n_sim: int = 5000,
    block_size: Optional[int] = None,
    point_estimator: str = 'mean',  # 'mean', 'median', 'base'
    suffix: str = 'boot',
    use_model_sigma: bool = False,
    hybrid_weight: Optional[float] = None,
    center_residuals: bool = True,
    random_state: int = 42,
    max_memory_mb: int = 1000,
    return_simulations: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """
    Генерирует робастные интервалы прогноза методом блокового бутстрапа (moving block bootstrap)
    на основе эмпирических остатков моделей. Сохраняет временную зависимость в остатках за счёт
    выборки блоков вместо независимых наблюдений.

    Алгоритм:
        1. Вычисление остатков (факт - прогноз) на in-sample данных.
           При `center_residuals=True` остатки центрируются (среднее = 0).
        2. Для каждой серии и модели:
           - Генерация `n_sim` траекторий путём случайной выборки перекрывающихся блоков остатков.
           - При `use_model_sigma=True` к остаткам добавляется гауссовский шум с дисперсией,
             оценённой из исходных интервалов модели ({model}-hi-95 / lo-95).
           - При `hybrid_weight=w` применяется взвешенная комбинация:
             шум = w * эмпирические_остатки + (1-w) * модельный_шум.
        3. Агрегация симуляций:
           - Точечный прогноз: среднее / медиана / исходный прогноз.
           - Интервалы покрытия: квантили симуляций для заданных уровней `levels`.

    Параметры
    ----------
    in_sample_sf : pd.DataFrame
        In-sample данные с колонками ['unique_id', 'ds', target_col] и прогнозами моделей.
    fcst_sf : pd.DataFrame
        Out-of-sample прогнозы на горизонте с колонками ['unique_id', 'ds'] и прогнозами моделей.
    target_col : str, optional (default='y')
        Имя колонки с целевой переменной.
    models : list of str, optional (default=None)
        Список моделей для обработки. Если None — используются все колонки кроме
        ['unique_id', 'ds', target_col].
    levels : tuple of int, optional (default=(90, 95))
        Уровни покрытия для построения интервалов (в процентах).
    n_sim : int, optional (default=5000)
        Количество симуляций бутстрапа.
    block_size : int, optional (default=None)
        Размер блока для выборки остатков. Если None — вычисляется как
        max(2, min(sqrt(T), H, T // 2)), где T — длина in-sample, H — горизонт прогноза.
    point_estimator : str, optional (default='mean')
        Метод агрегации симуляций: 'mean' (среднее), 'median' (медиана), 'base' (исходный прогноз).
    suffix : str, optional (default='boot')
        Суффикс для имён генерируемых колонок.
    use_model_sigma : bool, optional (default=False)
        Если True — к остаткам добавляется гауссовский шум с дисперсией, оценённой из
        исходных интервалов модели ({model}-hi-95 / lo-95).
    hybrid_weight : float, optional (default=None)
        Вес эмпирических остатков при комбинации с модельным шумом:
        - None: только эмпирические остатки (рекомендуется)
        - 1.0: только эмпирические остатки
        - 0.0: только модельный шум
        - 0.5: равная смесь (требует `use_model_sigma=True`)
    center_residuals : bool, optional (default=True)
        Центрировать ли остатки (вычитать среднее). Отключите, если смещение систематическое.
    random_state : int, optional (default=42)
        Seed для воспроизводимости генерации случайных чисел.
    max_memory_mb : int, optional (default=1000)
        Максимальный объём памяти (в МБ) для векторизованной генерации.
        При превышении используется пошаговая генерация.
    return_simulations : bool, optional (default=False)
        Если True, возвращает также словарь симуляций для диагностики:
        {(unique_id, model): np.ndarray(shape=(n_sim, H))}.

    Возвращает
    ----------
    pd.DataFrame или tuple
        - DataFrame: копия `fcst_sf` с добавленными колонками для каждой модели `m`:
          * `{m}_{suffix}`: точечный прогноз после бутстрапа
          * `{m}_{suffix}-lo-{lvl}`: нижняя граница интервала покрытия уровня `lvl`
          * `{m}_{suffix}-hi-{lvl}`: верхняя граница интервала покрытия уровня `lvl`
        - При `return_simulations=True`: кортеж (DataFrame, dict_simulations)

    Примечания
    ----------
    - Блоковый бутстрап корректно обрабатывает автокорреляцию в остатках.
    - Для коротких рядов (T < 5) возвращается базовый прогноз без бутстрапа.
    - При `use_model_sigma=True` требуется наличие колонок `{model}-hi-95` и `{model}-lo-95`.
    - Векторизованная генерация автоматически отключается при риске переполнения памяти.
    """
    # === Валидация входных данных ===
    required_cols_in = {'unique_id', 'ds', target_col}
    if not required_cols_in.issubset(in_sample_sf.columns):
        raise ValueError(f"in_sample_sf must contain columns: {required_cols_in}")
    if not {'unique_id', 'ds'}.issubset(fcst_sf.columns):
        raise ValueError("fcst_sf must contain 'unique_id' and 'ds' columns")
    
    if models is None:
        models = [c for c in in_sample_sf.columns if c not in required_cols_in]
    if not models:
        raise ValueError("No forecast models found in in_sample_sf")
    if not all(m in fcst_sf.columns for m in models):
        missing = set(models) - set(fcst_sf.columns)
        raise ValueError(f"Models missing in fcst_sf: {missing}")
    
    if use_model_sigma:
        for m in models:
            for bound in [f'{m}-hi-95', f'{m}-lo-95']:
                if bound not in fcst_sf.columns:
                    raise ValueError(
                        f"Column '{bound}' required when use_model_sigma=True. "
                        f"Consider setting use_model_sigma=False or providing interval columns."
                    )
    
    if hybrid_weight is not None and not use_model_sigma:
        raise ValueError("hybrid_weight requires use_model_sigma=True")
    
    if hybrid_weight is not None and not (0.0 <= hybrid_weight <= 1.0):
        raise ValueError("hybrid_weight must be in [0.0, 1.0] or None")
    
    if point_estimator not in ('mean', 'median', 'base'):
        raise ValueError("point_estimator must be one of: 'mean', 'median', 'base'")
    
    # === Инициализация ===
    rng = np.random.default_rng(random_state)
    fcst = fcst_sf.copy()
    
    # === Вычисление остатков ===
    resid_df = in_sample_sf[['unique_id', target_col] + models].copy()
    for m in models:
        resid = resid_df[target_col] - resid_df[m]
        if center_residuals:
            resid = resid - np.mean(resid)  # центрирование
        resid_df[f'{m}_resid'] = resid
    
    # === Создание колонок для результатов ===
    for m in models:
        fcst[f'{m}_{suffix}'] = np.nan
        for lvl in levels:
            fcst[f'{m}_{suffix}-lo-{lvl}'] = np.nan
            fcst[f'{m}_{suffix}-hi-{lvl}'] = np.nan
    
    # === Сбор симуляций для отладки (опционально) ===
    simulations_dict = {} if return_simulations else None
    
    # === Цикл по сериям ===
    for uid in fcst['unique_id'].unique():
        idx = fcst.index[fcst['unique_id'] == uid]
        H = len(idx)
        
        for m in models:
            # Извлечение остатков для серии
            resid = resid_df.loc[resid_df['unique_id'] == uid, f'{m}_resid'].dropna().values
            T = len(resid)
            
            # === Обработка коротких рядов ===
            if T < 5:
                warnings.warn(
                    f"Series '{uid}' model '{m}': insufficient residuals (T={T} < 5). "
                    f"Returning base forecast without bootstrap."
                )
                base_path = fcst.loc[idx, m].values
                fcst.loc[idx, f'{m}_{suffix}'] = base_path
                for lvl in levels:
                    fcst.loc[idx, f'{m}_{suffix}-lo-{lvl}'] = base_path
                    fcst.loc[idx, f'{m}_{suffix}-hi-{lvl}'] = base_path
                if return_simulations:
                    simulations_dict[(uid, m)] = np.tile(base_path, (n_sim, 1))
                continue
            
            # === Определение размера блока ===
            if block_size is None:
                bs = max(2, min(int(np.sqrt(T)), H, T // 2))
            else:
                bs = min(block_size, max(2, T // 2))
            
            max_start = T - bs + 1
            if max_start <= 0:
                warnings.warn(
                    f"Series '{uid}' model '{m}': block_size={bs} > residual length={T}. "
                    f"Reducing block_size to {T // 2}."
                )
                bs = max(2, T // 2)
                max_start = T - bs + 1
            
            base_path = fcst.loc[idx, m].values.copy()
            
            # === Оценка модельной дисперсии (опционально) ===
            if use_model_sigma:
                sigma_est = (fcst.loc[idx, f'{m}-hi-95'] - fcst.loc[idx, f'{m}-lo-95']) / (2 * 1.96)
                sigma_est = sigma_est.fillna(0).values
                sigma_est = np.maximum(sigma_est, 0)  # защита от отрицательных значений
            else:
                sigma_est = np.zeros(H)
            
            # === Генерация симуляций ===
            # Оценка памяти для векторизованной генерации
            mem_required_mb = (n_sim * H * 8) / (1024 ** 2)  # 8 байт на float64
            
            if mem_required_mb <= max_memory_mb and H * n_sim <= 10_000_000:
                # --- Векторизованная генерация (быстро) ---
                n_blocks_needed = int(np.ceil(H / bs))
                starts = rng.integers(0, max_start, size=(n_sim, n_blocks_needed))
                block_offsets = np.arange(bs)[None, None, :]  # shape: (1, 1, bs)
                block_indices = starts[:, :, None] + block_offsets  # shape: (n_sim, n_blocks, bs)
                block_indices = block_indices.reshape(n_sim, -1)[:, :H]  # shape: (n_sim, H)
                eps_arr = resid[block_indices]  # shape: (n_sim, H)
                
                # Гибридная комбинация шумов
                if use_model_sigma and hybrid_weight is not None and hybrid_weight < 1.0:
                    model_noise = rng.normal(0, sigma_est, size=(n_sim, H))
                    noise = hybrid_weight * eps_arr + (1.0 - hybrid_weight) * model_noise
                elif use_model_sigma and hybrid_weight is None:
                    # Только эмпирические остатки (рекомендуется)
                    noise = eps_arr
                elif use_model_sigma and hybrid_weight == 1.0:
                    noise = eps_arr
                elif use_model_sigma:  # hybrid_weight == 0.0
                    noise = rng.normal(0, sigma_est, size=(n_sim, H))
                else:
                    noise = eps_arr
                
                sims = base_path + noise
            else:
                # --- Пошаговая генерация (экономия памяти) ---
                sims = np.zeros((n_sim, H))
                for i in range(n_sim):
                    eps = []
                    while len(eps) < H:
                        start = rng.integers(0, max_start)
                        eps.extend(resid[start:start + bs])
                    eps_arr = np.array(eps[:H])
                    
                    if use_model_sigma and hybrid_weight is not None and hybrid_weight < 1.0:
                        model_noise = rng.normal(0, sigma_est)
                        noise = hybrid_weight * eps_arr + (1.0 - hybrid_weight) * model_noise
                    elif use_model_sigma and hybrid_weight is None:
                        noise = eps_arr
                    elif use_model_sigma and hybrid_weight == 0.0:
                        noise = rng.normal(0, sigma_est)
                    else:
                        noise = eps_arr
                    
                    sims[i] = base_path + noise
            
            # === Сохранение симуляций для отладки ===
            if return_simulations:
                simulations_dict[(uid, m)] = sims.copy()
            
            # === Агрегация: точечный прогноз ===
            if point_estimator == 'mean':
                point_array = sims.mean(axis=0)
            elif point_estimator == 'median':
                point_array = np.quantile(sims, 0.5, axis=0)
            else:  # 'base'
                point_array = base_path
            
            fcst.loc[idx, f'{m}_{suffix}'] = point_array
            
            # === Агрегация: интервалы покрытия ===
            for lvl in levels:
                alpha = (100 - lvl) / 200.0  # например, 0.05 для 90%
                lo = np.quantile(sims, alpha, axis=0)
                hi = np.quantile(sims, 1 - alpha, axis=0)
                fcst.loc[idx, f'{m}_{suffix}-lo-{lvl}'] = lo
                fcst.loc[idx, f'{m}_{suffix}-hi-{lvl}'] = hi
    
    
    fcst = fcst.drop(columns = models)
    # === Возврат результата ===
    if return_simulations:
        return fcst, simulations_dict
    return fcst


# def block_bootstrap_forecast(
#     in_sample_sf,       # in-sample прогноз: unique_id, ds, target_col, модели
#     fcst_sf,            # предсказания на горизонте: unique_id, ds, модели
#     target_col='y',
#     models=None,
#     levels=(90, 95),
#     n_sim=5000,
#     block_size=None,
#     point_estimator='mean',  # 'mean', 'median', 'base'
#     suffix='boot',
#     use_model_sigma=False,   # если True, учитываем интервалы модели
#     random_state=42,
# ):
#     """
#     Функция строит робастные интервалы прогноза для временных рядов с помощью moving block bootstrap на основе остатков моделей. 
#     block bootstrap Учитывает автокорреляцию во временных рядах за счёт выборки блоков остатков вместо отдельных наблюдений.

#     Особенности работы:
#     - Центрирование остатков: для каждой модели вычисляются и центрируются (среднее = 0) остатки 
#         остатки = тру − прогноз на in-sample .
#     - Генерация симуляций шумов (для каждого ВР и модели):
#         - Случайно выбираются перекрывающиеся блоки остатков длиной block_size.
#         - Блоки конкатенируются до достижения длины горизонта прогноза H.
#         - При use_model_sigma=True к остаткам добавляется гауссовский шум с дисперсией, оценённой из исходных интервалов модели.
#     - Симуляция: базовый прогноз + скорректированные остатки.
#     - Точечный прогноз: среднее / медиана / исходный прогноз по n_sim симуляциям.
#     - Интервалы: квантили симуляций для заданных уровней покрытия (levels).

#     Обратите внимание, что шумы генерируются по in-sample для прогноза!

#     Параметры
#     ----------
#     in_sample_sf : pd.DataFrame
#         In-sample данные с колонками ['unique_id', 'ds', target_col] и прогнозами моделей.
#     fcst_sf : pd.DataFrame
#         Out-of-sample прогнозы на горизонте с колонками ['unique_id', 'ds'] и прогнозами моделей.
#     target_col : str, optional (default='y')
#         Имя колонки с целевой переменной.
#     models : list of str, optional (default=None)
#         Список моделей для обработки. Если None — используются все колонки кроме
#         ['unique_id', 'ds', target_col].
#     levels : tuple of int, optional (default=(90, 95))
#         Уровни покрытия для построения интервалов (в процентах).
#     n_sim : int, optional (default=5000)
#         Количество симуляций бутстрапа.
#     block_size : int, optional (default=None)
#         Размер блока для выборки остатков. Если None — вычисляется как
#         max(2, min(sqrt(T), H)), где T — длина in-sample, H — горизонт прогноза.
#     point_estimator : str, optional (default='mean')
#         Метод агрегации симуляций: 'mean' (среднее), 'median' (медиана), 'base' (исходный прогноз).
#     suffix : str, optional (default='boot')
#         Суффикс для имён генерируемых колонок.
#     use_model_sigma : bool, optional (default=False)
#         Если True — к остаткам добавляется гауссовский шум с дисперсией, оценённой из
#         исходных интервалов модели ({model}-hi-95 / lo-95).
#     random_state : int, optional (default=42)
#         Seed для воспроизводимости генерации случайных чисел.

#     Возвращает
#     ----------
#     pd.DataFrame
#         Копия `fcst_sf` с добавленными колонками для каждой модели `m`:
#         - `{m}_{suffix}`: точечный прогноз после бутстрапа
#         - `{m}_{suffix}-lo-{lvl}`: нижняя граница интервала покрытия уровня `lvl`
#         - `{m}_{suffix}-hi-{lvl}`: верхняя граница интервала покрытия уровня `lvl`

#     Примечания
#     ----------
#     - Блоковый бутстрап корректно обрабатывает автокорреляцию в остатках, что критично
#       для временных рядов.
#     - При `use_model_sigma=True` комбинируется эмпирическая неопределённость (остатки)
#       и параметрическая (модельная дисперсия).
#     - Адаптивный размер блока по умолчанию балансирует между сохранением зависимости
#       и вариативностью выборки.
      
#     """
    
#     rng = np.random.default_rng(random_state)
#     fcst = fcst_sf.copy()
    
#     if models is None:
#         models = [c for c in in_sample_sf.columns if c not in ['unique_id','ds',target_col]]
    
#     # --- вычисляем остатки in-sample ---
#     resid_df = in_sample_sf[['unique_id', target_col] + models].copy()
#     for m in models:
#         resid = resid_df[target_col] - resid_df[m]
#         resid_df[f'{m}_resid'] = resid - np.mean(resid)  # центровка
    
#     # --- создаём колонки для bootstrap ---
#     for m in models:
#         fcst[f'{m}_{suffix}'] = np.nan
#         for lvl in levels:
#             fcst[f'{m}_{suffix}-lo-{lvl}'] = np.nan
#             fcst[f'{m}_{suffix}-hi-{lvl}'] = np.nan
    
#     # --- цикл по сериям и моделям ---
#     for uid in fcst['unique_id'].unique():
#         idx = fcst.index[fcst['unique_id']==uid]
#         H = len(idx)
        
#         for m in models:
#             resid = resid_df.loc[resid_df['unique_id']==uid, f'{m}_resid'].dropna().values
#             T = len(resid)
            
#             if block_size is None:
#                 bs = max(2, min(int(np.sqrt(T)), H))
#             else:
#                 bs = block_size
            
#             max_start = T - bs + 1
#             base_path = fcst.loc[idx, m].values.copy()
            
#             # --- если учитываем sigma модели, прибавляем дисперсию ---
#             if use_model_sigma:
#                 sigma_est = (fcst.loc[idx, f'{m}-hi-95'] - fcst.loc[idx, f'{m}-lo-95']) / 2
#                 sigma_est = sigma_est.fillna(0).values
#             else:
#                 sigma_est = np.zeros(H)
            
#             # --- симуляции ---
#             sims = np.zeros((n_sim, H))
#             for i in range(n_sim):
#                 eps = []
#                 while len(eps) < H:
#                     start = rng.integers(0, max_start)
#                     eps.extend(resid[start:start+bs])
#                 eps_arr = np.array(eps[:H])
                
#                 # добавляем модельную дисперсию
#                 noise = eps_arr + rng.normal(0, sigma_est)
#                 sims[i] = base_path + noise
            
#             # --- точечный прогноз ---
#             if point_estimator == 'mean':
#                 point_array = sims.mean(axis=0)
#             elif point_estimator == 'median':
#                 point_array = np.quantile(sims, 0.5, axis=0)
#             else:
#                 point_array = base_path
            
#             fcst.loc[idx, f'{m}_{suffix}'] = point_array
            
#             # --- интервалы ---
#             for lvl in levels:
#                 alpha = (100 - lvl) / 2 / 100
#                 lo = np.quantile(sims, alpha, axis=0)
#                 hi = np.quantile(sims, 1 - alpha, axis=0)
#                 fcst.loc[idx, f'{m}_{suffix}-lo-{lvl}'] = lo
#                 fcst.loc[idx, f'{m}_{suffix}-hi-{lvl}'] = hi
    
#     return fcst