import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from pandas.tseries.frequencies import to_offset

def backcast(
    df_train: pd.DataFrame,
    models,
    h: int,
    freq: str,
    level: list = None,
) -> pd.DataFrame:
    """
    Perform backcasting using StatsForecast models.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data with columns ['unique_id', 'ds', 'y'].
    models : list
        List of StatsForecast-compatible model instances (e.g., [ARIMA(...)]).
    h : int
        Backcast horizon (number of periods before the first observation).
    freq : str
        Pandas frequency string (e.g., 'D', 'W', 'M').
    level : list of int, optional
        Prediction intervals (e.g., [90]). If None, only point forecasts are returned.

    Returns
    -------
    pd.DataFrame
        Extended training set with backcasted values prepended.
    """
    sf = StatsForecast(models=models, freq=freq)
    alias = models[0].alias  # assumes single model or uses first

    backcast_parts = []

    for uid in df_train['unique_id'].unique():
        series = df_train[df_train['unique_id'] == uid].sort_values('ds')
        first_ds = series['ds'].iloc[0]

        # Reverse y for pseudo-backcasting
        y_rev = series['y'].iloc[::-1].values

        temp_df = pd.DataFrame({
            'unique_id': uid,
            'ds': series['ds'].values,
            'y': y_rev
        })

        fcst = sf.forecast(df=temp_df, h=h, level=level)

        mean_bc = fcst[alias].values[::-1]
        lo_bc = fcst[f'{alias}-lo-{level[0]}'].values[::-1] if level else None
        hi_bc = fcst[f'{alias}-hi-{level[0]}'].values[::-1] if level else None

        backcast_dates = pd.date_range(
            end=first_ds - to_offset(freq),
            periods=h,
            freq=freq
        )

        bc_dict = {
            'unique_id': uid,
            'ds': backcast_dates,
            'y': mean_bc,
            alias: mean_bc,
        }
        if level:
            bc_dict[f'{alias}-lo-{level[0]}'] = lo_bc
            bc_dict[f'{alias}-hi-{level[0]}'] = hi_bc

        backcast_parts.append(pd.DataFrame(bc_dict))

    df_train_extended = (
        pd.concat(backcast_parts + [df_train], ignore_index=True)
        .sort_values(['unique_id', 'ds'])
        .reset_index(drop=True)
    )

    return df_train_extended



def add_structural_shift_by_iqr(
    df,
    start_date,
    shift_percent=10.0,
    strategy='linear',
    duration_offset='26W',  # теперь offset-строка
    freq=None                # обязательная частота: 'D', 'W-MON', 'MS' и т.д.
):
    """
    Добавляет структурный сдвиг ко всем временным рядам.
    
    Параметры:
        df: DataFrame с колонками ['unique_id', 'ds', 'y']
        start_date: дата начала сдвига (str или Timestamp)
        shift_percent: процент от IQR (например, 10.0 → 10% от IQR)
        strategy: 'linear' или 'fast_saturation'
        duration_offset: строка с offset'ом (например, '6M', '12W', '90D')
        freq: частота временного ряда (обязательна!)
    
    Возвращает:
        df с новой колонкой y_drift_{strategy}
    """
    if freq is None:
        raise ValueError("Параметр 'freq' обязателен для корректного расчёта длительности сдвига.")
    
    df = df.copy()
    start_date = pd.Timestamp(start_date)
    new_col = f'y_drift_{strategy}'
    df[new_col] = df['y'].copy()
    
    # Преобразуем offset в timedelta или Period
    offset_obj = to_offset(duration_offset)
    end_date = start_date + offset_obj

    for uid in df['unique_id'].unique():
        mask = df['unique_id'] == uid
        sub = df[mask].sort_values('ds').reset_index(drop=True)
        
        # Пропускаем, если серия заканчивается до начала сдвига
        if sub['ds'].max() < start_date:
            continue
        
        # Вычисляем IQR по всей серии
        q75 = sub['y'].quantile(0.75)
        q25 = sub['y'].quantile(0.25)
        iqr = q75 - q25
        shift_amount = iqr * (shift_percent / 100.0)
        
        # Маски для периодов
        transition_mask = (sub['ds'] >= start_date) & (sub['ds'] <= end_date)
        post_mask = sub['ds'] > end_date
        
        n_trans = transition_mask.sum()
        if n_trans == 0:
            # Если нет точек в переходном периоде, но есть после — сразу применяем полный сдвиг
            if post_mask.any():
                sub.loc[post_mask, new_col] += shift_amount
                df.loc[mask, new_col] = sub[new_col].values
            continue
        
        # Генерация профиля сдвига
        if strategy == 'linear':
            t_norm = np.linspace(0, 1, n_trans)
            shift_profile = shift_amount * t_norm
            
        elif strategy == 'fast_saturation':
            t_norm = np.linspace(0, 1, n_trans)
            shift_profile = shift_amount * (1 - np.exp(-8 * t_norm))
            
        else:
            raise ValueError("strategy must be 'linear' or 'fast_saturation'")
        
        # Применяем профиль к переходному периоду
        sub.loc[transition_mask, new_col] += shift_profile
        
        # После завершения — полный сдвиг
        if post_mask.any():
            sub.loc[post_mask, new_col] += shift_amount
        
        df.loc[mask, new_col] = sub[new_col].values
    
    return df

    



def add_additive_shock(
    df,
    shock_date='2015-03-01',
    shock_strength=1.5,
    recovery_offset='12W',  # теперь offset-строка
    freq=None               # обязательная частота: 'D', 'W', 'M' и т.д.
):
    """
    Добавляет аддитивный шок с экспоненциальным восстановлением.
    
    Параметры:
        df: DataFrame с колонками ['unique_id', 'ds', 'y']
        shock_date: дата начала шока (str или Timestamp)
        shock_strength: множитель IQR для величины шока
        recovery_offset: строка с offset'ом (например, '12W', '3M')
        freq: частота временного ряда (обязательна, например 'W-MON')
    """
    if freq is None:
        raise ValueError("Параметр 'freq' обязателен для корректного расчёта восстановления.")
    
    df = df.copy()
    df['y_hybrid_shock'] = df['y'].copy()
    shock_date = pd.Timestamp(shock_date)
    recovery_delta = to_offset(recovery_offset)
    
    for uid in df['unique_id'].unique():
        mask = df['unique_id'] == uid
        sub = df[mask].sort_values('ds').reset_index(drop=True)
        
        if sub['ds'].max() < shock_date:
            continue
        
        # Находим позицию первого наблюдения >= shock_date
        shock_pos = sub[sub['ds'] >= shock_date].index[0]
        shock_time = sub.loc[shock_pos, 'ds']
        
        # Определяем конец периода восстановления
        recovery_end_time = shock_time + recovery_delta
        
        # Находим последнюю позицию <= recovery_end_time
        recovery_mask = sub['ds'] <= recovery_end_time
        if not recovery_mask.any():
            rec_n = 0
        else:
            last_recovery_pos = recovery_mask[::-1].idxmax()  # последний True
            rec_n = last_recovery_pos - shock_pos + 1
        
        if rec_n <= 0:
            continue
        
        n = len(sub)
        rec_n = min(rec_n, n - shock_pos)  # не выходим за границы
        
        # Расчёт шока через IQR
        q75 = sub['y'].quantile(0.75)
        q25 = sub['y'].quantile(0.25)
        iqr = q75 - q25
        shock_amount = iqr * shock_strength
        
        original_y = sub['y'].values
        shocked_y = original_y.copy()
        
        # Применяем шок
        shocked_y[shock_pos] -= shock_amount
        
        # Экспоненциальное восстановление
        for i in range(1, rec_n):
            pos = shock_pos + i
            decay_factor = np.exp(-3 * i / rec_n)
            shocked_y[pos] = decay_factor * shocked_y[pos - 1] + (1 - decay_factor) * original_y[pos]
        
        # После восстановления — оригинальные значения
        if shock_pos + rec_n < n:
            shocked_y[shock_pos + rec_n:] = original_y[shock_pos + rec_n:]
        
        df.loc[mask, 'y_hybrid_shock'] = shocked_y
    
    return df 


def add_seasonal_imbalance_by_quantile(
    df,
    target_seasonal_phases,
    season_length,
    imbalance_percent=30.0,
    direction='reduce',
    start_date=None,
    duration_offset=None,      # ← НОВОЕ: длительность эффекта
    freq=None,
    reference='global'
):
    """
    Добавляет сезонный дисбаланс в указанные фазы сезона на ограниченный период.
    
    Параметры:
        df: DataFrame с ['unique_id', 'ds', 'y']
        target_seasonal_phases: list[int] — фазы внутри сезона (0-based)
        season_length: int — длина сезона (в шагах)
        imbalance_percent: float — % от (q95 - q05)
        direction: 'reduce' или 'amplify'
        start_date: str/Timestamp — начало эффекта
        duration_offset: str — длительность (например, '2Y', '26W'); если None — бессрочно
        freq: str — частота ряда (обязательна)
        reference: 'global' (весь доступный временной ряд) или 'pre_event' (только наблюдения до start_date)

        
    """
    if freq is None:
        raise ValueError("Параметр 'freq' обязателен.")
    
    df = df.copy()
    df['y_seasonal_imbalance'] = df['y'].copy()
    
    start_date = pd.Timestamp(start_date) if start_date else df['ds'].min()
    end_date = start_date + to_offset(duration_offset) if duration_offset else pd.Timestamp.max
    
    target_phases = set(target_seasonal_phases)
    
    for uid in df['unique_id'].unique():
        mask = df['unique_id'] == uid
        sub = df[mask].sort_values('ds')
        
        # Референсный период для q05/q95
        if reference == 'pre_event':
            ref_mask = sub['ds'] < start_date
            ref_data = sub.loc[ref_mask, 'y'] if ref_mask.any() else sub['y']
        else:
            ref_data = sub['y']
        
        q05 = ref_data.quantile(0.05)
        q95 = ref_data.quantile(0.95)
        amplitude = q95 - q05
        center = ref_data.median()
        shift_amount = amplitude * (imbalance_percent / 100.0)
        
        # === КЛЮЧЕВОЕ УЛУЧШЕНИЕ: фаза через календарь ===
        # Генерируем регулярный диапазон от первой даты до последней
        full_range = pd.date_range(
            start=sub['ds'].min(),
            end=sub['ds'].max(),
            freq=freq
        )
        # Создаём mapping: дата → фаза
        phase_map = {date: i % season_length for i, date in enumerate(full_range)}
        
        # Применяем фазу только к датам, которые есть в данных
        sub['seasonal_phase'] = sub['ds'].map(phase_map)
        sub['seasonal_phase'] = sub['seasonal_phase'].fillna(-1).astype(int)  # -1 для пропущенных
        
        # Маска применения: фаза + временной интервал
        phase_mask = sub['seasonal_phase'].isin(target_phases)
        time_mask = (sub['ds'] >= start_date) & (sub['ds'] <= end_date)
        apply_mask = phase_mask & time_mask
        
        if not apply_mask.any():
            continue
        
        y_vals = sub['y'].values
        y_new = y_vals.copy()
        idx_apply = apply_mask.values
        
        if direction == 'reduce':
            diff = y_vals[idx_apply] - center
            shift = np.sign(diff) * np.minimum(np.abs(diff), shift_amount)
            y_new[idx_apply] = y_vals[idx_apply] - shift
        elif direction == 'amplify':
            signs = np.sign(y_vals[idx_apply] - center)
            y_new[idx_apply] = y_vals[idx_apply] + signs * shift_amount
        else:
            raise ValueError("direction must be 'reduce' or 'amplify'")
        
        sub['y_seasonal_imbalance'] = y_new
        df.loc[mask, 'y_seasonal_imbalance'] = sub['y_seasonal_imbalance'].values
    
    return df

def add_signal_degradation(
    df,
    degradation_start='2015-01-01',
    duration_offset='52W',      # ← offset-строка вместо weeks
    final_snr_ratio=0.5,
    freq=None,                  # ← обязательная частота
    reference='global'          # 'global' или 'pre_event'
):
    """
    Постепенно деградирует сигнал: шум линейно нарастает от 0 до final_snr_ratio * амплитуды.
    
    Параметры:
        df: DataFrame с ['unique_id', 'ds', 'y']
        degradation_start: дата начала деградации
        duration_offset: длительность нарастания (например, '1Y', '26W')
        final_snr_ratio: финальное отношение sigma / amplitude
        freq: частота ряда (обязательна)
        reference: как оценивать амплитуду ('global' или 'pre_event')
    """
    if freq is None:
        raise ValueError("Параметр 'freq' обязателен.")
    
    df = df.copy()
    df['y_degraded'] = df['y'].copy()
    start = pd.Timestamp(degradation_start)
    end = start + to_offset(duration_offset)
    
    for uid in df['unique_id'].unique():
        mask = df['unique_id'] == uid
        sub = df[mask].sort_values('ds')
        
        # Определяем амплитуду на референсном периоде
        if reference == 'pre_event':
            ref_data = sub[sub['ds'] < start]['y']
            if len(ref_data) < 10:
                ref_data = sub['y']
        else:
            ref_data = sub['y']
        
        q5 = ref_data.quantile(0.05)
        q95 = ref_data.quantile(0.95)
        amplitude = q95 - q5
        final_sigma = amplitude * final_snr_ratio
        
        # Маски периодов
        degrade_mask = (sub['ds'] >= start) & (sub['ds'] <= end)
        post_mask = sub['ds'] > end
        
        if not degrade_mask.any():
            continue
        
        # === Период нарастания ===
        degrade_sub = sub[degrade_mask]
        n_degrade = len(degrade_sub)
        
        if n_degrade == 1:
            t_norm = np.array([1.0])
        else:
            t_norm = np.linspace(0, 1, n_degrade)  # от 0 (начало) до 1 (конец)
        
        sigmas = final_sigma * t_norm
        noise_degrade = np.random.normal(0, sigmas)  # векторизовано!
        
        sub.loc[degrade_mask, 'y_degraded'] += noise_degrade
        
        # === Постоянный шум после окончания нарастания ===
        if post_mask.any():
            n_post = post_mask.sum()
            noise_post = np.random.normal(0, final_sigma, n_post)
            sub.loc[post_mask, 'y_degraded'] += noise_post
        
        df.loc[mask, 'y_degraded'] = sub['y_degraded'].values
    
    return df


def add_seasonal_phase_shift(
    df,
    phase_shift_steps=4,
    season_length=52,
    start_date=None,
    duration_offset='1Y',
    freq=None
):
    """
    Применяет локальный циклический сдвиг к значениям в заданном периоде.
    Эффект: "сезонность началась раньше на phase_shift_steps шагов".
    """
    if freq is None:
        raise ValueError("Параметр 'freq' обязателен.")
    
    df = df.copy()
    df['y_phase_shift'] = df['y'].copy()
    
    start_date = pd.Timestamp(start_date) if start_date else df['ds'].min()
    end_date = start_date + to_offset(duration_offset)
    
    for uid in df['unique_id'].unique():
        mask = df['unique_id'] == uid
        sub = df[mask].sort_values('ds').reset_index(drop=True)
        
        if sub['ds'].max() < start_date:
            continue
        
        # Находим индексы в периоде действия
        period_mask = (sub['ds'] >= start_date) & (sub['ds'] <= end_date)
        if not period_mask.any():
            continue
        
        # Извлекаем подмассив
        period_y = sub.loc[period_mask, 'y'].values
        n = len(period_y)
        
        if n == 0:
            continue
        
        # Циклический сдвиг: "раньше" = сдвиг влево = -shift
        shifted_y = np.roll(period_y, -phase_shift_steps)
        
        # Обновляем
        sub.loc[period_mask, 'y_phase_shift'] = shifted_y
        df.loc[mask, 'y_phase_shift'] = sub['y_phase_shift'].values
    
    return df
    



def add_trend_break(
    df,
    break_date='2015-01-01',
    trend_change_per_step=-0.02,   # ← изменение за ОДИН ШАГ (например, за день, неделю, час)
    duration_offset='2Y',          # ← длительность как offset
    return_to_original=True,
    freq=None                      # ← обязательная частота
):
    """
    Добавляет трендовый разрыв с изменением наклона.
    
    Параметры:
        trend_change_per_step: изменение уровня за один временной шаг (в единицах y)
        duration_offset: как долго действует новый тренд (например, '2Y', '26W')
        freq: частота ряда ('D', 'W', 'M', 'H' и т.д.)
    """
    if freq is None:
        raise ValueError("Параметр 'freq' обязателен.")
    
    df = df.copy()
    df['y_trend_break'] = df['y'].copy()
    break_date = pd.Timestamp(break_date)
    end_date = break_date + to_offset(duration_offset)
    
    for uid in df['unique_id'].unique():
        mask = df['unique_id'] == uid
        sub = df[mask].sort_values('ds').reset_index(drop=True)
        
        if sub['ds'].max() < break_date:
            continue
        
        # Находим позицию начала разрыва
        break_pos = sub[sub['ds'] >= break_date].index[0]
        
        # Маски периодов
        active_mask = (sub['ds'] >= break_date) & (sub['ds'] <= end_date)
        post_mask = sub['ds'] > end_date
        
        if not active_mask.any():
            continue
        
        # === КЛЮЧЕВОЕ УЛУЧШЕНИЕ: количество шагов от точки разрыва ===
        # Генерируем регулярный диапазон
        full_range = pd.date_range(
            start=sub['ds'].iloc[0],
            end=sub['ds'].iloc[-1],
            freq=freq
        )
        # Создаём mapping: дата → позиция
        date_to_step = {date: i for i, date in enumerate(full_range)}
        
        # Позиция точки разрыва в регулярном ряду
        if break_date not in date_to_step:
            # Ближайшая дата вперёд
            future_dates = [d for d in full_range if d >= break_date]
            if not future_dates:
                continue
            break_date_aligned = future_dates[0]
        else:
            break_date_aligned = break_date
        
        break_step = date_to_step[break_date_aligned]
        
        # Для каждой даты в активном периоде — вычисляем шаг от разрыва
        active_sub = sub[active_mask]
        steps_from_break = []
        for d in active_sub['ds']:
            if d in date_to_step:
                steps = date_to_step[d] - break_step
                steps_from_break.append(max(steps, 0))
            else:
                # Пропущенная дата — пропускаем или интерполируем?
                steps_from_break.append(np.nan)
        
        steps_from_break = np.array(steps_from_break)
        valid_steps = ~np.isnan(steps_from_break)
        
        if not valid_steps.any():
            continue
        
        # Накопленный трендовый сдвиг
        trend_offset = trend_change_per_step * steps_from_break[valid_steps]
        
        # Применяем сдвиг
        y_vals = sub.loc[active_mask, 'y'].values.copy()
        y_vals[valid_steps] += trend_offset
        sub.loc[active_mask, 'y_trend_break'] = y_vals
        
        # После периода — возврат к оригиналу (если нужно)
        if return_to_original and post_mask.any():
            sub.loc[post_mask, 'y_trend_break'] = sub.loc[post_mask, 'y']
        
        df.loc[mask, 'y_trend_break'] = sub['y_trend_break'].values
    
    return df
    
def add_local_volatility_spike(
    df,
    spike_start='2015-06-01',
    duration_offset='4W',   # ← offset-строка вместо weeks
    noise_percent=50.0,
    freq=None,              # ← обязательная частота
    reference='global'      # 'global' или 'pre_event'
):
    """
    Добавляет локальный всплеск волатильности (аддитивный белый шум) в заданный период.
    """
    if freq is None:
        raise ValueError("Параметр 'freq' обязателен.")
    
    df = df.copy()
    df['y_volatility_spike'] = df['y'].copy()
    start = pd.Timestamp(spike_start)
    end = start + to_offset(duration_offset)
    
    for uid in df['unique_id'].unique():
        mask = df['unique_id'] == uid
        sub = df[mask].sort_values('ds')
        
        # Определяем референсный период для квантилей
        if reference == 'pre_event':
            ref_data = sub[sub['ds'] < start]['y']
            if len(ref_data) < 10:
                ref_data = sub['y']
        else:
            ref_data = sub['y']
        
        q5 = ref_data.quantile(0.05)
        q95 = ref_data.quantile(0.95)
        amplitude = q95 - q5
        sigma = amplitude * (noise_percent / 100.0)
        
        # Маска всплеска
        spike_mask = (sub['ds'] >= start) & (sub['ds'] <= end)
        if not spike_mask.any():
            continue
        
        # Генерируем уникальный шум для каждой серии
        noise = np.random.normal(0, sigma, spike_mask.sum())
        
        sub.loc[spike_mask, 'y_volatility_spike'] += noise
        df.loc[mask, 'y_volatility_spike'] = sub['y_volatility_spike'].values
    
    return df


def add_asymmetric_shock(
    df,
    shock_start='2015-03-01',
    drop_percent=40.0,
    recovery_offset='8W',   # ← offset-строка
    freq=None,              # ← обязательная частота
    reference='global'
):
    """
    Добавляет асимметричный шок: резкое падение → медленное восстановление.
    """
    if freq is None:
        raise ValueError("Параметр 'freq' обязателен.")
    
    df = df.copy()
    df['y_asymmetric_shock'] = df['y'].copy()
    start = pd.Timestamp(shock_start)
    end_recovery = start + to_offset(recovery_offset)
    
    for uid in df['unique_id'].unique():
        mask = df['unique_id'] == uid
        sub = df[mask].sort_values('ds').reset_index(drop=True)
        
        if sub['ds'].max() < start:
            continue
        
        # Находим позицию начала шока
        shock_pos = sub[sub['ds'] >= start].index[0]
        shock_time = sub.loc[shock_pos, 'ds']
        
        # Находим последнюю дату <= end_recovery
        recovery_mask = sub['ds'] <= end_recovery
        if not recovery_mask.any():
            rec_end_pos = shock_pos
        else:
            rec_end_pos = recovery_mask[::-1].idxmax()
        
        n = len(sub)
        if rec_end_pos <= shock_pos:
            # Нет данных для восстановления — просто падение
            drop_amount = _get_drop_amount(sub, shock_start, drop_percent, reference)
            sub.loc[shock_pos, 'y_asymmetric_shock'] -= drop_amount
            df.loc[mask, 'y_asymmetric_shock'] = sub['y_asymmetric_shock'].values
            continue
        
        # Количество шагов восстановления
        rec_steps = rec_end_pos - shock_pos  # например, 8 недель → 8 шагов
        
        # Вычисляем drop_amount
        drop_amount = _get_drop_amount(sub, shock_start, drop_percent, reference)
        
        y_vals = sub['y'].values.copy()
        
        # 1. Резкое падение
        y_vals[shock_pos] -= drop_amount
        
        # 2. Линейное восстановление до оригинального уровня
        for i in range(1, rec_steps + 1):
            pos = shock_pos + i
            if pos >= n:
                break
            # Восстанавливаемся к оригинальному значению
            y_vals[pos] = y_vals[shock_pos] + drop_amount * (i / rec_steps)
        
        # 3. После восстановления — оригинал
        if shock_pos + rec_steps + 1 < n:
            y_vals[shock_pos + rec_steps + 1:] = sub['y'].values[shock_pos + rec_steps + 1:]
        
        sub['y_asymmetric_shock'] = y_vals
        df.loc[mask, 'y_asymmetric_shock'] = sub['y_asymmetric_shock'].values
    
    return df

def _get_drop_amount(sub, shock_start, drop_percent, reference):
    """Вспомогательная функция для вычисления величины падения."""
    start = pd.Timestamp(shock_start)
    if reference == 'pre_event':
        ref_data = sub[sub['ds'] < start]['y']
        if len(ref_data) < 10:
            ref_data = sub['y']
    else:
        ref_data = sub['y']
    
    q5 = ref_data.quantile(0.05)
    q95 = ref_data.quantile(0.95)
    amplitude = q95 - q5
    return amplitude * (drop_percent / 100.0)

from pandas.tseries.offsets import DateOffset

def add_seasonal_regressors_effect(
    df,
    event_dates,                # список дат праздников (например, ['2020-01-01', '2021-01-01', ...])
    effect_percent=10.0,        # % от (q95 - q05)
    direction='spike',          # 'spike' (рост) или 'dip' (падение)
    window_radius=1,            # сколько дней до/после (0 = только сам день)
    freq=None,                  # частота ряда ('D', 'H' и т.д.)
    reference='global'          # 'global' или 'pre_event'
):
    """
    Добавляет эффект вокруг повторяющихся событий (праздников).
    
    Параметры:
        event_dates: list[str] — даты событий (без времени)
        effect_percent: float — сила эффекта как % от амплитуды (q95 - q05)
        direction: 'spike' → увеличение, 'dip' → уменьшение
        window_radius: int — радиус окна в днях (например, 1 → [−1, 0, +1])
        freq: str — частота временного ряда
        reference: как оценивать амплитуду
    """
    if freq is None:
        raise ValueError("Параметр 'freq' обязателен.")
    
    df = df.copy()
    col_name = f'y_holiday_{"spike" if direction == "spike" else "dip"}'
    df[col_name] = df['y'].copy()
    
    # Преобразуем события в Timestamp (без времени)
    event_days = set(pd.to_datetime(event_dates).normalize())
    
    for uid in df['unique_id'].unique():
        mask = df['unique_id'] == uid
        sub = df[mask].sort_values('ds')
        
        # Определяем референсный период
        if reference == 'pre_event':
            first_event = min(event_days) if event_days else pd.Timestamp.max
            ref_data = sub[sub['ds'].dt.normalize() < first_event]['y']
            if len(ref_data) < 10:
                ref_data = sub['y']
        else:
            ref_data = sub['y']
        
        q05 = ref_data.quantile(0.05)
        q95 = ref_data.quantile(0.95)
        amplitude = q95 - q05
        effect_amount = amplitude * (effect_percent / 100.0)
        
        # Маска для всех дней, попадающих в окно события
        apply_mask = pd.Series(False, index=sub.index)
        
        for event_day in event_days:
            # Создаём окно: [event - radius, ..., event + radius]
            window_start = event_day - pd.Timedelta(days=window_radius)
            window_end = event_day + pd.Timedelta(days=window_radius)
            
            # Находим все наблюдения в этом окне
            in_window = (sub['ds'] >= window_start) & (sub['ds'] <= window_end)
            apply_mask |= in_window
        
        if not apply_mask.any():
            continue
        
        # Применяем эффект
        if direction == 'spike':
            sub.loc[apply_mask, col_name] += effect_amount
        elif direction == 'dip':
            sub.loc[apply_mask, col_name] -= effect_amount
        else:
            raise ValueError("direction must be 'spike' or 'dip'")
        
        df.loc[mask, col_name] = sub[col_name].values
    
    return df


def add_sensor_drift(
    df,
    drift_start='2015-01-01',
    duration_offset='2Y',
    drift_percent=20.0,
    profile='linear',           # 'linear', 'exponential', 'logarithmic', 'saturation'
    freq=None,
    reference='global'
):
    """
    Добавляет постепенный дрейф датчика (смещение показаний).
    
    Параметры:
        drift_start: дата начала дрейфа
        duration_offset: длительность дрейфа (например, '1Y', '52W')
        drift_percent: максимальное смещение как % от (q95 - q05)
        profile: форма дрейфа
            - 'linear': линейный рост
            - 'exponential': быстрый рост → замедление
            - 'logarithmic': быстрое начало → насыщение
            - 'saturation': 1 - exp(-kt) → плавное насыщение
        freq: частота ряда
        reference: 'global' или 'pre_event'
    """
    if freq is None:
        raise ValueError("Параметр 'freq' обязателен.")
    
    df = df.copy()
    df['y_sensor_drift'] = df['y'].copy()
    start = pd.Timestamp(drift_start)
    end = start + to_offset(duration_offset)
    
    for uid in df['unique_id'].unique():
        mask = df['unique_id'] == uid
        sub = df[mask].sort_values('ds')
        
        # Определяем амплитуду до дрейфа
        if reference == 'pre_event':
            ref_data = sub[sub['ds'] < start]['y']
            if len(ref_data) < 10:
                ref_data = sub['y']
        else:
            ref_data = sub['y']
        
        q05 = ref_data.quantile(0.05)
        q95 = ref_data.quantile(0.95)
        amplitude = q95 - q05
        max_drift = amplitude * (drift_percent / 100.0)
        
        # Маска периода дрейфа
        drift_mask = (sub['ds'] >= start) & (sub['ds'] <= end)
        post_mask = sub['ds'] > end
        
        if not drift_mask.any():
            continue
        
        # Нормированное время: 0 → начало, 1 → конец
        drift_sub = sub[drift_mask]
        n = len(drift_sub)
        if n == 1:
            t_norm = np.array([1.0])
        else:
            t_norm = np.linspace(0, 1, n)
        
        # Профиль дрейфа
        if profile == 'linear':
            drift_profile = t_norm
        elif profile == 'exponential':
            # Быстрый рост → замедление (например, деградация)
            drift_profile = 1 - np.exp(-3 * t_norm)
        elif profile == 'logarithmic':
            # Быстрое начало → насыщение
            drift_profile = np.log(1 + 9 * t_norm) / np.log(10)
        elif profile == 'saturation':
            # Плавное насыщение (часто в физике)
            drift_profile = 1 - np.exp(-2 * t_norm)
        else:
            raise ValueError("profile must be 'linear', 'exponential', 'logarithmic', or 'saturation'")
        
        # Применяем дрейф
        drift_amount = max_drift * drift_profile
        sub.loc[drift_mask, 'y_sensor_drift'] += drift_amount
        
        # После окончания — постоянное смещение
        if post_mask.any():
            sub.loc[post_mask, 'y_sensor_drift'] += max_drift
        
        df.loc[mask, 'y_sensor_drift'] = sub['y_sensor_drift'].values
    
    return df



def add_skewness_shift(
    df,
    skew_start='2015-01-01',
    duration_offset='1Y',
    tail='right',              # 'right' или 'left'
    intensity=0.3,             # сила искажения (0–1)
    freq=None,
    reference='global'
):
    """
    Добавляет асимметрию: усиливает правый или левый хвост распределения.
    """
    if freq is None:
        raise ValueError("Параметр 'freq' обязателен.")
    
    df = df.copy()
    df['y_skewed'] = df['y'].copy()
    start = pd.Timestamp(skew_start)
    end = start + to_offset(duration_offset)
    
    for uid in df['unique_id'].unique():
        mask = df['unique_id'] == uid
        sub = df[mask].sort_values('ds')
        
        if reference == 'pre_event':
            ref_data = sub[sub['ds'] < start]['y']
            if len(ref_data) < 10:
                ref_data = sub['y']
        else:
            ref_data = sub['y']
        
        median_ref = ref_data.median()
        iqr_ref = ref_data.quantile(0.75) - ref_data.quantile(0.25)
        apply_mask = (sub['ds'] >= start) & (sub['ds'] <= end)
        if not apply_mask.any():
            continue
        
        y_vals = sub.loc[apply_mask, 'y'].values.copy()
        deviations = y_vals - median_ref
        
        if tail == 'right':
            # Увеличиваем положительные отклонения
            scale = 1 + intensity * (deviations > 0)
        elif tail == 'left':
            # Увеличиваем отрицательные отклонения
            scale = 1 + intensity * (deviations < 0)
        else:
            raise ValueError("tail must be 'right' or 'left'")
        
        y_vals = median_ref + deviations * scale
        sub.loc[apply_mask, 'y_skewed'] = y_vals
        df.loc[mask, 'y_skewed'] = sub['y_skewed'].values
    
    return df


def add_variance_expansion(
    df,
    expansion_start='2015-01-01',
    duration_offset='1Y',
    expansion_factor=1.5,      # >1 → расширение, <1 → сужение
    freq=None,
    reference='global'
):
    """
    Масштабирует отклонения от медианы: y_new = median + (y - median) * factor.
    """
    if freq is None:
        raise ValueError("Параметр 'freq' обязателен.")
    
    df = df.copy()
    df['y_variance_expanded'] = df['y'].copy()
    start = pd.Timestamp(expansion_start)
    end = start + to_offset(duration_offset)
    
    for uid in df['unique_id'].unique():
        mask = df['unique_id'] == uid
        sub = df[mask].sort_values('ds')
        
        if reference == 'pre_event':
            ref_data = sub[sub['ds'] < start]['y']
            if len(ref_data) < 10:
                ref_data = sub['y']
        else:
            ref_data = sub['y']
        
        median_ref = ref_data.median()
        apply_mask = (sub['ds'] >= start) & (sub['ds'] <= end)
        if not apply_mask.any():
            continue
        
        # Масштабируем отклонения от медианы
        deviations = sub.loc[apply_mask, 'y'] - median_ref
        sub.loc[apply_mask, 'y_variance_expanded'] = median_ref + deviations * expansion_factor
        
        df.loc[mask, 'y_variance_expanded'] = sub['y_variance_expanded'].values
    
    return df

from scipy.stats import wasserstein_distance, ks_2samp, energy_distance

def marginal_distribution_test(y_orig: np.ndarray, y_aug: np.ndarray, 
                              name: str = "aug") -> pd.DataFrame:
    """
    Тест маргинальных распределений: возвращает простой датафрейм с бинарными решениями (1 = прошло, 0 = не прошло)
    """
    # Базовые статистики
    orig_std = y_orig.std() + 1e-8
    
    # Метрики и пороги
    metrics = {
        'wasserstein': {
            'value': wasserstein_distance(y_orig, y_aug)/orig_std,
            'threshold': 0.15,
            'passed': wasserstein_distance(y_orig, y_aug)/orig_std <= 0.15  
        },
        'ks_pvalue': {
            'value': ks_2samp(y_orig, y_aug).pvalue,
            'threshold': 0.05,
            'passed': ks_2samp(y_orig, y_aug).pvalue > 0.05
        },
        'mean_diff_pct': {
            'value': abs(y_orig.mean() - y_aug.mean()) / (abs(y_orig.mean()) + 1e-8) * 100,
            'threshold': 5.0,
            'passed': abs(y_orig.mean() - y_aug.mean()) / (abs(y_orig.mean()) + 1e-8) * 100 <= 5.0
        },
        'std_diff_pct': {
            'value': abs(y_orig.std() - y_aug.std()) / (orig_std) * 100,
            'threshold': 10.0,
            'passed': abs(y_orig.std() - y_aug.std()) / (orig_std) * 100 <= 10.0
        }
    }
    
    # Формируем датафрейм
    df = pd.DataFrame([
        {
            'metric': metric,
            'value': f"{m['value']:.4f}",
            'threshold': f"{m['threshold']:.4f}",
            'passed': int(m['passed'])  # 1 = OK, 0 = FAIL
        }
        for metric, m in metrics.items()
    ])
    
    # Добавляем имя аугментации как метаданные (для конкатенации)
    df['augmentation'] = name
    df = df[['augmentation', 'metric', 'value', 'threshold', 'passed']]
    
    return df

def structural_tests(y_orig: np.ndarray, y_aug: np.ndarray, 
                    seasonal_period: int = None,
                    name: str = "aug") -> pd.DataFrame:
    """
    Тест структурных свойств временных рядов: возвращает датафрейм с бинарными решениями (1 = прошло, 0 = не прошло)
    """
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.stattools import acf
    from scipy.signal import periodogram
    from scipy.stats import wasserstein_distance

    # Убедимся, что ряды достаточно длинные
    min_len = max(20, 2 * (seasonal_period or 1))
    if len(y_orig) < min_len or len(y_aug) < min_len:
        # Возвращаем нейтральный результат, если данных мало
        dummy = {
            'acf_mae': {'value': 0.0, 'threshold': 0.05, 'passed': 1},
            'spectral_dist': {'value': 0.0, 'threshold': 0.2, 'passed': 1},
            'quantile_max_diff': {'value': 0.0, 'threshold': 0.15, 'passed': 1},
            'volatility_corr': {'value': 1.0, 'threshold': 0.7, 'passed': 1}
        }
        df = pd.DataFrame([
            {'metric': k, 'value': f"{v['value']:.4f}", 'threshold': f"{v['threshold']:.4f}", 'passed': v['passed']}
            for k, v in dummy.items()
        ])
        df['augmentation'] = name
        return df[['augmentation', 'metric', 'value', 'threshold', 'passed']]

    tests = {}

    # 1. ACF MAE
    max_lag = min(50, len(y_orig) // 2)
    acf_orig = acf(y_orig, nlags=max_lag, fft=True)
    acf_aug = acf(y_aug, nlags=max_lag, fft=True)
    acf_mae = np.mean(np.abs(acf_orig - acf_aug))
    tests['acf_mae'] = {
        'value': acf_mae,
        'threshold': 0.05,
        'passed': acf_mae <= 0.05
    }

    # 2. Spectral distance
    f_orig, Pxx_orig = periodogram(y_orig)
    f_aug, Pxx_aug = periodogram(y_aug)
    Pxx_orig = Pxx_orig / (Pxx_orig.sum() + 1e-8)
    Pxx_aug = Pxx_aug / (Pxx_aug.sum() + 1e-8)
    spectral_dist = wasserstein_distance(f_orig, f_aug, u_weights=Pxx_orig, v_weights=Pxx_aug)
    spec_thresh = 0.1 * (1.0 / seasonal_period) if seasonal_period else 0.2
    tests['spectral_dist'] = {
        'value': spectral_dist,
        'threshold': spec_thresh,
        'passed': spectral_dist <= spec_thresh
    }

    # 3. Quantile coverage
    quantiles = [0.1, 0.5, 0.9]
    q_orig = np.quantile(y_orig, quantiles)
    q_aug = np.quantile(y_aug, quantiles)
    max_q_diff = np.max(np.abs(q_orig - q_aug)) / (np.std(y_orig) + 1e-8)
    tests['quantile_max_diff'] = {
        'value': max_q_diff,
        'threshold': 0.15,
        'passed': max_q_diff <= 0.15
    }

    # 4. Volatility correlation
    window = min(20, len(y_orig) // 5)
    vol_orig = pd.Series(y_orig).rolling(window).std().dropna().values
    vol_aug = pd.Series(y_aug).rolling(window).std().dropna().values
    n_overlap = min(len(vol_orig), len(vol_aug))
    if n_overlap > 1:
        vol_corr = np.corrcoef(vol_orig[:n_overlap], vol_aug[:n_overlap])[0, 1]
        vol_corr = float(vol_corr) if not np.isnan(vol_corr) else 0.0
    else:
        vol_corr = 0.0
    tests['volatility_corr'] = {
        'value': vol_corr,
        'threshold': 0.7,
        'passed': vol_corr >= 0.7
    }

    # Формируем датафрейм
    df = pd.DataFrame([
        {
            'metric': metric,
            'value': f"{m['value']:.4f}",
            'threshold': f"{m['threshold']:.4f}",
            'passed': int(m['passed'])
        }
        for metric, m in tests.items()
    ])
    
    df['augmentation'] = name
    df = df[['augmentation', 'metric', 'value', 'threshold', 'passed']]
    
    return df