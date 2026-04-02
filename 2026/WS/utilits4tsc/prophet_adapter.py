import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Any
from statsforecast.models import _TS
from prophet import Prophet as FBProphet
from prophet.make_holidays import make_holidays_df

from statsforecast.utils import ConformalIntervals

from statsforecast.arima import (
    Arima,
    auto_arima_f,
    fitted_arima,
    forecast_arima,
    forward_arima,
    is_constant,
)

from statsforecast.models import _add_fitted_pi


class TSProphet(_TS):
    """
    Кастомный Prophet wrapper для StatsForecast с поддержкой:
    - кастомных сезонностей (hourly/daily/weekly/yearly)
    - prediction intervals (level)
    - country holidays и пользовательских праздников
    """

    _tags = {
        "python_dependencies": ["prophet"],
        "capability:pred_int": True,
    }
    uses_exog = True
    def __init__(
        self,
        freq: str = "H",
        growth: str = "linear",
        yearly_seasonality = False, # : bool|int
        weekly_seasonality = False, #: bool|int
        daily_seasonality = False,# bool|int
        custom_seasonalities: Optional[List[Dict]] = None,
        seasonality_mode: str = "additive",
        seasonality_prior_scale: float = 10.0,
        holidays: Optional[pd.DataFrame] = None,
        country_holidays: Optional[str] = None,
        alias: str = "Prophet",
        **kwargs,
    ):
        super().__init__()

        # Перехват устаревших аргументов
        if "add_country_holidays" in kwargs:
            country_holidays = kwargs.pop("add_country_holidays")

        self.freq = freq
        self.growth = growth
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.custom_seasonalities = custom_seasonalities or []
        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays = holidays
        self.country_holidays = country_holidays
        self.kwargs = kwargs
        self.alias = alias

    # --------------------------------------------------
    def _get_interval_width(self, level: Optional[List[int]]) -> float:
        return max(level) / 100.0 if level else 0.8

    def _merge_holidays(self, years):
        dfs = []
        if self.country_holidays:
            dfs.append(make_holidays_df(year_list=years, country=self.country_holidays))
        if self.holidays is not None:
            dfs.append(self.holidays.copy())
        if not dfs:
            return None
        return pd.concat(dfs, ignore_index=True)

    # --------------------------------------------------
    def _add_custom_seasonalities(self, model: FBProphet):
        for s in self.custom_seasonalities:
            model.add_seasonality(
                name=s["name"],
                period=s["period"],
                fourier_order=s.get("fourier_order", 5),
            )

    def _create_model(self, interval_width: float, years=None):
        holidays_df = self._merge_holidays(years)

        model = FBProphet(
            growth=self.growth,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            seasonality_prior_scale=self.seasonality_prior_scale,
            interval_width=interval_width,
            holidays=holidays_df,
            **self.kwargs,
        )

        self._add_custom_seasonalities(model)
        return model

    # --------------------------------------------------
    def fit(self, y: np.ndarray, X=None, level=None):
        ds = self._dates if hasattr(self, "_dates") else pd.date_range(
            "1970-01-01", periods=len(y), freq=self.freq
        )
        years = sorted(set(pd.to_datetime(ds).year))
        interval_width = self._get_interval_width(level)

        df = pd.DataFrame({"ds": ds, "y": y})
        self.model_ = self._create_model(interval_width, years)
        self.model_.fit(df)
        self._store_cs(y=y, X=X)
        return self

    def predict(self, h: int, X=None, level=None):
        future = self.model_.make_future_dataframe(periods=h, freq=self.freq, include_history=False)
        fcst = self.model_.predict(future)

        res = {"mean": fcst["yhat"].to_numpy()}
        if level:
            for lv in level:
                res[f"lo-{lv}"] = fcst["yhat_lower"].to_numpy()
                res[f"hi-{lv}"] = fcst["yhat_upper"].to_numpy()
        return res

    def predict_in_sample(self, level: Optional[List[int]] = None):
        fcst = self.model_.predict(self.model_.history)
        values = [fcst["yhat"].to_numpy()]
        cols = ["fitted"]
        if level:
            for lv in level:
                values.extend([fcst["yhat_lower"].to_numpy(), fcst["yhat_upper"].to_numpy()])
                cols.extend([f"lo-{lv}", f"hi-{lv}"])
        values = np.column_stack(values)
        return {"values": values, "cols": cols}

    # --------------------------------------------------
    def forecast(
        self,
        y: np.ndarray,
        h: int,
        X: Optional[np.ndarray] = None,
        X_future: Optional[np.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ):
        self.fit(y=y, X=X, level=level)
        out = self.predict(h=h, X=X_future, level=level)

        if fitted:
            ins = self.model_.predict(self.model_.history)
            out["fitted"] = ins["yhat"].to_numpy()
            if level:
                for lv in level:
                    out[f"fitted-lo-{lv}"] = ins["yhat_lower"].to_numpy()
                    out[f"fitted-hi-{lv}"] = ins["yhat_upper"].to_numpy()
        return out

    def forward(
        self,
        y: np.ndarray,
        h: int,
        X: Optional[np.ndarray] = None,
        X_future: Optional[np.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ):
        out = self.predict(h=h, X=X_future, level=level)
        if fitted:
            ins = self.model_.predict(self.model_.history)
            out["fitted"] = ins["yhat"].to_numpy()
            if level:
                for lv in level:
                    out[f"fitted-lo-{lv}"] = ins["yhat_lower"].to_numpy()
                    out[f"fitted-hi-{lv}"] = ins["yhat_upper"].to_numpy()
        return out
class ARIMAProphet(_TS):
    r"""ARIMAProphet model.

    AutoRegressive Integrated Moving Average model.

    References:
        - [Rob J. Hyndman, Yeasmin Khandakar (2008). "Automatic Time Series Forecasting: The forecast package for R"](https://www.jstatsoft.org/article/view/v027i03).

    Args:
        order (tuple, default=(0, 0, 0)): A specification of the non-seasonal part of the ARIMA model: the three components (p, d, q) are the AR order, the degree of differencing, and the MA order.
        season_length (int, default=1): Number of observations per unit of time. Ex: 24 Hourly data.
        seasonal_order (tuple, default=(0, 0, 0)): A specification of the seasonal part of the ARIMA model. (P, D, Q) for the  AR order, the degree of differencing, the MA order.
        include_mean (bool, default=True): Should the ARIMA model include a mean term? The default is True for undifferenced series, False for differenced ones (where a mean would not affect the fit nor predictions).
        include_drift (bool, default=False): Should the ARIMA model include a linear drift term? (i.e., a linear regression with ARIMA errors is fitted.)
        include_constant (bool, optional, default=None): If True, then includ_mean is set to be True for undifferenced series and include_drift is set to be True for differenced series. Note that if there is more than one difference taken, no constant is included regardless of the value of this argument. This is deliberate as otherwise quadratic and higher order polynomial trends would be induced.
        blambda (float, optional, default=None): Box-Cox transformation parameter.
        biasadj (bool, default=False): Use adjusted back-transformed mean Box-Cox.
        method (str, default='CSS-ML'): Fitting method: maximum likelihood or minimize conditional sum-of-squares. The default (unless there are missing values) is to use conditional-sum-of-squares to find starting values, then maximum likelihood.
        fixed (dict, optional, default=None): Dictionary containing fixed coefficients for the arima model. Example: `{'ar1': 0.5, 'ma2': 0.75}`. For autoregressive terms use the `ar{i}` keys. For its seasonal version use `sar{i}`. For moving average terms use the `ma{i}` keys. For its seasonal version use `sma{i}`. For intercept and drift use the `intercept` and `drift` keys. For exogenous variables use the `ex_{i}` keys.
        alias (str): Custom name of the model.
        prediction_intervals (Optional[ConformalIntervals]): Information to compute conformal prediction intervals. By default, the model will compute the native prediction intervals.
    """

    uses_exog = True

    def __init__(
        self,
        order: Tuple[int, int, int] = (0, 0, 0),
        season_length: int = 1,
        seasonal_order: Tuple[int, int, int] = (0, 0, 0),
        include_mean: bool = True,
        include_drift: bool = False,
        include_constant: Optional[bool] = None,
        blambda: Optional[float] = None,
        biasadj: bool = False,
        method: str = "CSS-ML",
        fixed: Optional[dict] = None,
        alias: str = "ARIMAProphet",
        prediction_intervals: Optional[ConformalIntervals] = None,
        prophet: Optional[Any] = None,   # ← сюда передаем Prophet объект
        # #Prophet
        freq: str = "H",
        growth: str = "linear",
        yearly_seasonality  = False, #: bool|int
        weekly_seasonality = False, #: bool|int
        daily_seasonality = False, #: bool|int
        custom_seasonalities: Optional[List[Dict]] = None,
        seasonality_mode: str = "additive",
        seasonality_prior_scale: float = 10.0,
        holidays: Optional[pd.DataFrame] = None,
        country_holidays: Optional[str] = None,

    ):
        self.order = order
        self.season_length = season_length
        self.seasonal_order = seasonal_order
        self.include_mean = include_mean
        self.include_drift = include_drift
        self.include_constant = include_constant
        self.blambda = blambda
        self.biasadj = biasadj
        self.method = method
        self.fixed = fixed
        self.alias = alias
        self.prediction_intervals = prediction_intervals
        self.prophet = prophet,
        #Prophet
        self.freq = freq
        self.growth = growth
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.custom_seasonalities = custom_seasonalities or []
        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays = holidays
        self.country_holidays = country_holidays

    def _merge_holidays(self, years):
        dfs = []
        if self.country_holidays:
            dfs.append(make_holidays_df(year_list=years, country=self.country_holidays))
        if self.holidays is not None:
            dfs.append(self.holidays.copy())
        if not dfs:
            return None
        return pd.concat(dfs, ignore_index=True)

    # -----------------------------
    def _add_custom_seasonalities(self, model: FBProphet):
        for s in self.custom_seasonalities:
            model.add_seasonality(
                name=s["name"],
                period=s["period"],
                fourier_order=s.get("fourier_order", 5),
            )

    # -----------------------------
    def _create_model(self, years=None):
        """Создаёт объект Prophet с учётом всех параметров"""
        holidays_df = self._merge_holidays(years)
        model = FBProphet(
            growth=self.growth,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            seasonality_prior_scale=self.seasonality_prior_scale,
            # interval_width=self.interval_width,
            holidays=holidays_df,
            # **self.kwargs,
        )
        self._add_custom_seasonalities(model)
        return model

   
    
    def fit(
        self,
        y: np.ndarray,
        X: Optional[np.ndarray] = None,
    ):
        r"""
        Fit the model to a time series (numpy array) `y`
        and optionally exogenous variables (numpy array) `X`.

        Args:
            y (numpy.array): Clean time series of shape (t, ).
            X (array-like): Optional exogenous of shape (t, n_x).

        Returns:
            self: Fitted model.
        """
        # y = _ensure_float(y)
        with np.errstate(invalid="ignore"):

            X_full = self._create_exog_train(y, X)
            
            self.model_ = Arima(
                x=y,
                order=self.order,
                seasonal={"order": self.seasonal_order, "period": self.season_length},
                xreg=X_full,
                include_mean=self.include_mean,
                include_constant=self.include_constant,
                include_drift=self.include_drift,
                blambda=self.blambda,
                biasadj=self.biasadj,
                method=self.method,
                fixed=self.fixed,
            )

        
        self._store_cs(y=y, X=X)
        return self
     
        
    def predict(
        self,
        h: int,
        X: Optional[np.ndarray] = None,
        level: Optional[List[int]] = None,
    ):
        r"""Predict with fitted model.

        Args:
            h (int): Forecast horizon.
            X (array-like): Optional exogenous of shape (h, n_x).
            level (List[float]): Confidence levels (0-100) for prediction intervals.

        Returns:
            forecasts (dict): Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """

        X_full = self._create_exog_future(y, X, h=h)
        
        fcst = forecast_arima(self.model_, h=h, xreg=X_full, level=level)
        mean = fcst["mean"]
        res = {"mean": mean}
        if level is None:
            return res
        level = sorted(level)
        if self.prediction_intervals is not None:
            res = self._add_predict_conformal_intervals(res, level)
        else:
            res = {
                "mean": mean,
                **{f"lo-{l}": fcst["lower"][f"{l}%"] for l in reversed(level)},
                **{f"hi-{l}": fcst["upper"][f"{l}%"] for l in level},
            }
        return res

    def predict_in_sample(self, level: Optional[List[int]] = None):
        r"""Access fitted insample predictions.

        Args:
            level (List[float]): Confidence levels (0-100) for prediction intervals.

        Returns:
            forecasts (dict): Dictionary with entries `fitted` for point predictions and `level_*` for probabilistic predictions.
        """
        mean = fitted_arima(self.model_)
        res = {"fitted": mean}
        if level is not None:
            se = np.sqrt(self.model_["sigma2"])
            res = _add_fitted_pi(res=res, se=se, level=level)
        return res

    def forecast(
        self,
        y: np.ndarray,
        h: int,
        X: Optional[np.ndarray] = None,
        X_future: Optional[np.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ):
        r"""Memory efficient predictions.

        This method avoids memory burden due from object storage.
        It is analogous to `fit_predict` without storing information.
        It assumes you know the forecast horizon in advance.

        Args:
            y (numpy.array): Clean time series of shape (n, ).
            h (int): Forecast horizon.
            X (array-like): Optional insample exogenous of shape (t, n_x).
            X_future (array-like): Optional exogenous of shape (h, n_x) optional exogenous.
            level (List[float]): Confidence levels (0-100) for prediction intervals.
            fitted (bool): Whether or not returns insample predictions.

        Returns:
            forecasts (dict): Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        # y = _ensure_float(y)
        X_train = self._create_exog_train(y, X)
        X_fut   = self._create_exog_future(y, X_future, h=h)
        
        with np.errstate(invalid="ignore"):

            mod = Arima(
                x=y,
                order=self.order,
                seasonal={"order": self.seasonal_order, "period": self.season_length},
                xreg=X_train,
                include_mean=self.include_mean,
                include_constant=self.include_constant,
                include_drift=self.include_drift,
                blambda=self.blambda,
                biasadj=self.biasadj,
                method=self.method,
                fixed=self.fixed,
            )
            
        fcst = forecast_arima(mod, h, xreg=X_fut, level=level)
        res = {"mean": fcst["mean"]}
        if fitted:
            res["fitted"] = fitted_arima(mod)
        if level is not None:
            level = sorted(level)
            if self.prediction_intervals is not None:
                res = self._add_conformal_intervals(fcst=res, y=y, X=X, level=level)
            else:
                res = {
                    **res,
                    **{f"lo-{l}": fcst["lower"][f"{l}%"] for l in reversed(level)},
                    **{f"hi-{l}": fcst["upper"][f"{l}%"] for l in level},
                }
            if fitted:
                # add prediction intervals for fitted values
                se = np.sqrt(mod["sigma2"])
                res = _add_fitted_pi(res=res, se=se, level=level)
        return res

    def forward(
        self,
        y: np.ndarray,
        h: int,
        X: Optional[np.ndarray] = None,
        X_future: Optional[np.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ):
        r"""Apply fitted model to a new time series.

        Args:
            y (numpy.array): Clean time series of shape (n, ).
            h (int): Forecast horizon.
            X (array-like): Optional insample exogenous of shape (t, n_x).
            X_future (array-like): Optional exogenous of shape (h, n_x).
            level (List[float]): Confidence levels for prediction intervals.
            fitted (bool): Whether or not returns insample predictions.

        Returns:
            forecasts (dict): Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        if not hasattr(self, "model_"):
            raise Exception("You have to use the `fit` method first")
        # y = _ensure_float(y)
        
        X_train = self._create_exog_train(y, X)
        X_fut   = self._create_exog_future(y, X_future, h=h)
        
        with np.errstate(invalid="ignore"):
            mod = forward_arima(self.model_, y=y, xreg=X_train, method=self.method)
        fcst = forecast_arima(mod, h, xreg=X_fut, level=level)
        res = {"mean": fcst["mean"]}
        if fitted:
            res["fitted"] = fitted_arima(mod)
        if level is not None:
            level = sorted(level)
            if self.prediction_intervals is not None:
                res = self._add_conformal_intervals(fcst=res, y=y, X=X_train, level=level)
            else:
                res = {
                    **res,
                    **{f"lo-{l}": fcst["lower"][f"{l}%"] for l in reversed(level)},
                    **{f"hi-{l}": fcst["upper"][f"{l}%"] for l in level},
                }
            if fitted:
                # add prediction intervals for fitted values
                se = np.sqrt(mod["sigma2"])
                res = _add_fitted_pi(res=res, se=se, level=level)
        return res


    def _create_exog_train(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Экзогены только для обучающего ряда"""
        import pandas as pd
        import numpy as np
        from prophet import Prophet
    
        ds_train = pd.date_range("1970-01-01", periods=len(y), freq=self.freq)
        df_train = pd.DataFrame({"ds": ds_train, "y": y})
    
        prophet = self._create_model()
        prophet.fit(df_train)
        seasonal_train, _, component_cols, _ = prophet.make_all_seasonality_features(df_train)
        self.xreg_cols = component_cols
        X_seasonal = seasonal_train.values if not seasonal_train.empty else None
    
        if X is not None:
            X_full = np.hstack([X_seasonal, X]) if X_seasonal is not None else X
        else:
            X_full = X_seasonal
    
        return X_full
    
    
    def _create_exog_future(self, y: np.ndarray, X_future: Optional[np.ndarray] = None, h: int = 0) -> Optional[np.ndarray]:
        """Экзогены только для будущих шагов"""
        import pandas as pd
        import numpy as np
        from prophet import Prophet
    
        ds_train = pd.date_range("1970-01-01", periods=len(y), freq=self.freq)
        ds_future = pd.date_range(
            start=ds_train[-1] + pd.tseries.frequencies.to_offset(self.freq),
            periods=h,
            freq=self.freq
        )
        df_future = pd.DataFrame({"ds": ds_future, "y": np.zeros(h)})
    
        prophet = self._create_model()
        prophet.fit(pd.DataFrame({"ds": ds_train, "y": y}))  # Prophet должен знать тренд
    
        seasonal_future, _, _, _ = prophet.make_all_seasonality_features(df_future)
        X_seasonal_future = seasonal_future.values if not seasonal_future.empty else None
    
        if X_future is not None:
            X_full_future = np.hstack([X_seasonal_future, X_future]) if X_seasonal_future is not None else X_future
        else:
            X_full_future = X_seasonal_future
    
        return X_full_future