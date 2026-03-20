import numpy as np
import pandas as pd
from typing import Tuple, Optional
from statsforecast import StatsForecast



def stationary_bootstrap(
    resid: np.ndarray,
    h: int,
    n_sim: int,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Stationary bootstrap for dependent residuals.

    resid : (T,)
    returns : (n_sim, h)
    """
    T = len(resid)
    sims = np.zeros((n_sim, h))

    for i in range(n_sim):
        t = rng.integers(T)
        for j in range(h):
            sims[i, j] = resid[t]
            if rng.random() < p:
                t = rng.integers(T)
            else:
                t = (t + 1) % T

    return sims

class BootstrapForecastSF:
    """
    Residual-based stationary bootstrap for StatsForecast.

    Produces:
    - median / mean forecast
    - prediction intervals
    - nixtla-compatible output
    """

    def __init__(
        self,
        sf: StatsForecast,
        n_sim: int = 2000,
        block_size: int = 24,
        levels: Tuple[int, ...] = (80, 90),
        point: str = "median",  # "median" | "mean" | "base"
        random_state: int = 42,
    ):
        self.sf = sf
        self.n_sim = n_sim
        self.block_size = block_size
        self.p = 1.0 / block_size
        self.levels = levels
        self.point = point
        self.rng = np.random.default_rng(random_state)

        if point not in {"median", "mean", "base"}:
            raise ValueError("point must be 'median', 'mean', or 'base'")

    def forecast(
            self,
            df: pd.DataFrame,
            h: int,
            X_df: Optional[pd.DataFrame] = None,
        ) -> pd.DataFrame:
        """
        Multi-model residual stationary bootstrap forecast.
    
        Returns nixtla-style DataFrame with:
        - {model}_y_hat
        - {model}_lo-{lvl}
        - {model}_hi-{lvl}
        """
    
        # 1. Forecast + fitted values (StatsForecast API)
        fcst = self.sf.forecast(
            df=df,
            h=h,
            X_df=X_df,
            fitted=True,
        )
    
        fitted = self.sf.forecast_fitted_values()
    
        # 2. Определяем модели
        model_cols = [c for c in fitted.columns if c not in ("unique_id", "ds")]
    
        # 3. Готовим выход
        out = fcst.copy()
    
        for m in model_cols:
            out.rename(columns={m: f"{m}_y_hat"}, inplace=True)
            for lvl in self.levels:
                out[f"{m}_lo-{lvl}"] = np.nan
                out[f"{m}_hi-{lvl}"] = np.nan
    
        # 4. Bootstrap: model × series
        for m in model_cols:
            # residuals для конкретной модели
            resid_df = (
                df.merge(
                    fitted[["unique_id", "ds", m]],
                    on=["unique_id", "ds"],
                    how="inner",
                )
                .rename(columns={m: "y_hat"})
            )
            resid_df["resid"] = resid_df["y"] - resid_df["y_hat"]
    
            for uid in out["unique_id"].unique():
                idx = out.index[out["unique_id"] == uid]
                base_path = out.loc[idx, f"{m}_y_hat"].values
    
                resid = (
                    resid_df.loc[resid_df["unique_id"] == uid, "resid"]
                    .dropna()
                    .values
                )
    
                T = len(resid)
                if T < 5:
                    # fallback: без бутстрэпа
                    for lvl in self.levels:
                        out.loc[idx, f"{m}_lo-{lvl}"] = base_path
                        out.loc[idx, f"{m}_hi-{lvl}"] = base_path
                    continue
    
                # 5. Stationary bootstrap
                eps = stationary_bootstrap(
                    resid=resid,
                    h=h,
                    n_sim=self.n_sim,
                    p=self.p,
                    rng=self.rng,
                )
    
                sims = base_path + eps  # shape: (n_sim, h)
    
                # 6. Point forecast
                if self.point == "median":
                    out.loc[idx, f"{m}_y_hat"] = np.median(sims, axis=0)
                elif self.point == "mean":
                    out.loc[idx, f"{m}_y_hat"] = sims.mean(axis=0)
                # else: base forecast already in place
    
                # 7. Prediction intervals
                for lvl in self.levels:
                    alpha = (100 - lvl) / 200
                    out.loc[idx, f"{m}_lo-{lvl}"] = np.quantile(sims, alpha, axis=0)
                    out.loc[idx, f"{m}_hi-{lvl}"] = np.quantile(
                        sims, 1 - alpha, axis=0
                    )
    
            return out
