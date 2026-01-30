from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from prophet import Prophet
from statsforecast.models import _TS
from statsforecast.utils import ConformalIntervals


class ProphetAdapter(_TS):
    """Native StatsForecast _TS-compatible Prophet adapter."""

    uses_exog = True  # позволяет подавать X / X_future

    def __init__(
        self,
        growth="linear",
        changepoints=None,
        n_changepoints=25,
        changepoint_range=0.8,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        holidays: Optional[pd.DataFrame] = None,
        seasonality_mode="additive",
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        changepoint_prior_scale=0.05,
        mcmc_samples=0,
        interval_width=0.80,
        uncertainty_samples=0,
        stan_backend=None,
        alias: str = "Prophet",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ):
        # Сохраняем kwargs Prophet
        self.prophet_kwargs = dict(
            growth=growth,
            changepoints=changepoints,
            n_changepoints=n_changepoints,
            changepoint_range=changepoint_range,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            holidays=holidays,
            seasonality_mode=seasonality_mode,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            changepoint_prior_scale=changepoint_prior_scale,
            mcmc_samples=mcmc_samples,
            interval_width=interval_width,
            uncertainty_samples=uncertainty_samples,
            stan_backend=stan_backend,
        )
        self.model_: Optional[Prophet] = None
        self.alias = alias
        self.prediction_intervals = prediction_intervals

    def fit(self, y: np.ndarray, X: Optional[pd.DataFrame] = None):
        """Fit Prophet. X can contain 'ds' column for panel alignment; ignored otherwise."""
        if X is not None and "ds" in X:
            ds = pd.to_datetime(X["ds"])
        else:
            # создаем последовательные даты
            ds = pd.date_range(start="2000-01-01", periods=len(y))
        df = pd.DataFrame({"ds": ds, "y": y})
        self.model_ = Prophet(**self.prophet_kwargs)
        self.model_.fit(df)
        self._store_cs(y=y, X=X)
        return self

    def predict(self, h: int, X: Optional[pd.DataFrame] = None, level: Optional[List[int]] = None):
        """Forecast h steps using fitted Prophet."""
        if self.model_ is None:
            raise Exception("Fit the model first!")

        if X is not None and "ds" in X:
            ds_future = pd.to_datetime(X["ds"])
        else:
            # если X нет — создаем даты после последнего ds
            last_ds = self.model_.history["ds"].max()
            freq = pd.infer_freq(self.model_.history["ds"])
            if freq is None:
                freq = "D"
            ds_future = pd.date_range(start=last_ds + pd.Timedelta(1, unit=freq[0]), periods=h, freq=freq)

        df_future = pd.DataFrame({"ds": ds_future})
        fcst = self.model_.predict(df_future)
        res = {"mean": fcst["yhat"].to_numpy()}

        if level is not None:
            level = sorted(level)
            if self.prediction_intervals is not None:
                res = self._add_predict_conformal_intervals(res, level)
            else:
                for l in level:
                    res[f"lo-{l}"] = fcst[f"yhat_lower"].to_numpy()
                    res[f"hi-{l}"] = fcst[f"yhat_upper"].to_numpy()
        return res

    def predict_in_sample(self, level: Optional[List[int]] = None):
        """Return in-sample fitted values."""
        if self.model_ is None:
            raise Exception("Fit the model first!")
        fitted = self.model_.history["y"].values
        res = {"fitted": fitted}
        return res

    def forecast(
        self,
        y: np.ndarray,
        h: int,
        X: Optional[pd.DataFrame] = None,
        X_future: Optional[pd.DataFrame] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ):
        """Fit and forecast (memory-efficient)."""
        self.fit(y=y, X=X)
        fcst = self.predict(h=h, X=X_future, level=level)
        if fitted:
            fcst_in = self.predict_in_sample(level=level)
            fcst.update(fcst_in)
        return fcst

    def forward(
        self,
        y: np.ndarray,
        h: int,
        X: Optional[pd.DataFrame] = None,
        X_future: Optional[pd.DataFrame] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ):
        """Apply fitted model to new series."""
        if self.model_ is None:
            return self.forecast(y=y, h=h, X=X, X_future=X_future, level=level, fitted=fitted)
        else:
            return self.predict(h=h, X=X_future, level=level)

    def __repr__(self):
        return self.alias
