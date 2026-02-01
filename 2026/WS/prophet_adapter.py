import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from statsforecast.models import _TS
from prophet import Prophet as FBProphet
from prophet.make_holidays import make_holidays_df

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

    def __init__(
        self,
        freq: str = "H",
        growth: str = "linear",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
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
