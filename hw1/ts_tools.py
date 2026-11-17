"""Анализ временных рядов продаж Online Retail."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_transactions(path: str) -> pd.DataFrame:
    """Читает Excel и добавляет дату без времени и строковый код товара."""
    df = pd.read_excel(path)
    df["date"] = df["InvoiceDate"].dt.normalize()
    df["code"] = df["StockCode"].astype(str)
    return df


def trading_days(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(df["date"].dropna().unique()))


def product_sales(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Отбирает продажи товара без отмен, возвратов и строк с нулевой ценой."""
    product = df[df["code"] == code]
    is_return = product["InvoiceNo"].astype(str).str.startswith("C") | (product["Quantity"] <= 0)
    return product[~is_return & (product["UnitPrice"] > 0)]


def make_daily(df: pd.DataFrame, code: str, value: str = "Quantity") -> pd.Series:
    """Суммирует продажи по дням. Субботы исключает, дни без транзакций оставляет как NaN."""
    sales = product_sales(df, code)
    by_day = sales.groupby("date")[value].sum()
    open_days = trading_days(df)
    calendar = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    calendar = calendar[calendar.dayofweek != 5]
    daily = by_day.reindex(calendar)
    daily.loc[daily.index.isin(open_days) & daily.isna()] = 0.0
    daily.name = f"{code}: {value}"
    return daily


def aggregate(daily: pd.Series, freq: str) -> pd.Series:
    """Суммирует продажи за полные недели или месяцы. Пустые периоды оставляет как NaN."""
    if freq == "D":
        return daily
    if freq == "W":
        out = daily.resample("W-SUN").sum(min_count=1)
        first_ok = daily.index.min() <= out.index[0] - pd.Timedelta(days=6)
        last_ok = daily.index.max() >= out.index[-1]
    elif freq == "M":
        out = daily.resample("MS").sum(min_count=1)
        first_ok = daily.index.min() <= out.index[0]
        last_ok = daily.index.max() >= out.index[-1] + pd.offsets.MonthEnd(0)
    else:
        raise ValueError("freq должен быть 'D', 'W' или 'M'")
    if not first_ok:
        out = out.iloc[1:]
    if not last_ok:
        out = out.iloc[:-1]
    out.name = daily.name
    return out


def closed_days_per_period(daily: pd.Series, freq: str) -> pd.Series:
    rule = {"W": "W-SUN", "M": "MS"}[freq]
    return daily.isna().resample(rule).sum().astype(int)


def missing_report(s: pd.Series) -> pd.DataFrame:
    """Находит начало, конец и длину каждого блока пропусков."""
    na = s.isna()
    block = (na != na.shift()).cumsum()
    rows = []
    for _, g in s[na].groupby(block[na]):
        rows.append({"start": g.index[0].date(), "end": g.index[-1].date(), "n_days": len(g)})
    return pd.DataFrame(rows, columns=["start", "end", "n_days"])


def impute_closed_days(s: pd.Series, method: str = "weekday", window_weeks: int = 4) -> pd.Series:
    """Заполняет пропуски средним по тому же дню недели или интерполяцией."""
    if method == "linear":
        return s.interpolate(method="time", limit_direction="both")
    out = s.copy()
    for d in s.index[s.isna()]:
        neigh = [d + pd.Timedelta(weeks=k) for k in range(-window_weeks, window_weeks + 1) if k]
        vals = s.reindex(neigh).dropna()
        out[d] = vals.mean() if len(vals) else s.mean()
    return out


def detect_outliers(s: pd.Series, method: str = "iqr", k: float = 1.5, thr: float = 3.5) -> pd.DataFrame:
    """Ищет выбросы по IQR, MAD или z-оценке."""
    x = s.dropna()
    if method == "iqr":
        q1, q3 = x.quantile([0.25, 0.75])
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        mask = (x < lo) | (x > hi)
        score = (x - x.median()) / (q3 - q1)
    elif method == "mad":
        med = x.median()
        mad = stats.median_abs_deviation(x, scale="normal")
        score = (x - med) / mad if mad > 0 else pd.Series(0.0, index=x.index)
        mask = score.abs() > thr
    elif method == "zscore":
        score = (x - x.mean()) / x.std()
        mask = score.abs() > 3
    else:
        raise ValueError(method)
    return pd.DataFrame({"value": x[mask], "score": score[mask]}).sort_values("value", ascending=False)


def winsorize(s: pd.Series, upper_q: float = 0.99) -> pd.Series:
    return s.clip(upper=s.quantile(upper_q))


def hurst_rs(x: np.ndarray) -> float:
    """Оценивает показатель Херста методом R/S."""
    x = np.asarray(x, float)
    n_max = len(x)
    sizes = np.unique(np.floor(np.logspace(np.log10(8), np.log10(n_max // 2), 12)).astype(int))
    rs = []
    for n in sizes:
        vals = []
        for i in range(0, n_max - n + 1, n):
            seg = x[i:i + n]
            dev = np.cumsum(seg - seg.mean())
            r, sd = dev.max() - dev.min(), seg.std()
            if sd > 0:
                vals.append(r / sd)
        rs.append(np.mean(vals))
    return float(np.polyfit(np.log(sizes), np.log(rs), 1)[0])


def spectral_analysis(s: pd.Series, fs: float = 1.0, nperseg: int | None = None) -> dict:
    """Считает периодограмму, спектр Уэлча и их характеристики."""
    x = s.dropna().to_numpy(float)
    x = x - x.mean()
    f_p, p_p = signal.periodogram(x, fs=fs, detrend=False)
    nperseg = nperseg or min(len(x), 128)
    f_w, p_w = signal.welch(x, fs=fs, nperseg=nperseg, detrend=False)
    pos = f_p > 0
    k = np.argmax(p_p[pos])
    psd = p_w[f_w > 0] / p_w[f_w > 0].sum()
    entropy = float(-(psd * np.log(psd + 1e-15)).sum() / np.log(len(psd)))
    return {"f_periodogram": f_p, "p_periodogram": p_p, "f_welch": f_w, "p_welch": p_w,
            "dominant_period": float(1 / f_p[pos][k]), "dominant_power_share": float(p_p[pos][k] / p_p[pos].sum()),
            "centroid": float((f_w * p_w).sum() / p_w.sum()), "spectral_entropy": entropy,
            "nyquist": fs / 2, "freq_resolution": fs / len(x)}


def describe_series(s: pd.Series) -> pd.Series:
    x = s.dropna()
    q1, q3 = x.quantile([0.25, 0.75])
    out = {
        "n": len(x), "mean": x.mean(), "median": x.median(), "std": x.std(),
        "cv": x.std() / x.mean() if x.mean() else np.nan,
        "min": x.min(), "max": x.max(), "range": x.max() - x.min(),
        "q1": q1, "q3": q3, "iqr": q3 - q1,
        "skew": stats.skew(x), "kurtosis": stats.kurtosis(x),
        "share_zeros": float((x == 0).mean()),
        "peak_factor": x.max() / np.sqrt((x ** 2).mean()),
        "energy": float((x ** 2).sum()),
        "hurst_rs": hurst_rs(x.to_numpy()) if len(x) >= 40 else np.nan,
    }
    return pd.Series(out)


def ljung_box(s: pd.Series, lags: list[int]) -> pd.DataFrame:
    return acorr_ljungbox(s.dropna(), lags=lags)


def weekday_profile(daily_raw: pd.Series) -> pd.DataFrame:
    x = daily_raw.dropna()
    g = x.groupby(x.index.dayofweek)
    prof = pd.DataFrame({"mean": g.mean(), "median": g.median(), "n": g.size()})
    names = {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 6: "Вс"}
    prof.index = [names[i] for i in prof.index]
    return prof


def kruskal_by_weekday(daily_raw: pd.Series) -> tuple[float, float]:
    x = daily_raw.dropna()
    groups = [v.values for _, v in x.groupby(x.index.dayofweek)]
    h, p = stats.kruskal(*groups)
    return float(h), float(p)


def trend_test(s: pd.Series) -> dict:
    """Оценивает линейный тренд и корреляцию Спирмена со временем."""
    x = s.dropna()
    t = np.flatnonzero(s.notna())
    lr = stats.linregress(t, x.values)
    rho, p_rho = stats.spearmanr(t, x.values)
    return {"slope_per_step": lr.slope, "slope_p": lr.pvalue, "spearman_rho": rho, "spearman_p": p_rho}


def decompose(s: pd.Series, period: int, method: str = "stl", model: str = "additive", seasonal: int = 13):
    """Разлагает ряд. В режиме multiplicative применяет STL к log1p ряда."""
    if method == "stl":
        y = np.log1p(s) if model == "multiplicative" else s
        return STL(y, period=period, seasonal=seasonal, robust=True).fit()
    return seasonal_decompose(s, model=model, period=period, extrapolate_trend="freq")


def seasonal_strength(res) -> float:
    """Оценивает силу сезонности по доле дисперсии остатка."""
    seas = res.seasonal
    resid = res.resid
    return float(max(0.0, 1 - np.var(resid) / np.var(seas + resid)))


def trend_strength(res) -> float:
    return float(max(0.0, 1 - np.var(res.resid) / np.var(res.trend + res.resid)))


@dataclass
class StationarityResult:
    name: str
    adf_stat: float
    adf_p: float
    adf_lags: int
    kpss_stat: float
    kpss_p: float
    verdict: str


def verdict_adf_kpss(adf_p: float, kpss_p: float, alpha: float = 0.05) -> str:
    adf_rej, kpss_rej = adf_p < alpha, kpss_p < alpha
    if adf_rej and not kpss_rej:
        return "в пользу стационарности"
    if not adf_rej and kpss_rej:
        return "в пользу нестационарности"
    if not adf_rej and not kpss_rej:
        return "неоднозначно (ни один тест не отверг H0)"
    return "противоречие (оба теста отвергли H0)"


def stationarity_tests(
    s: pd.Series, name: str = "", regression: str = "c", autolag: str = "AIC",
) -> StationarityResult:
    """Проверяет ряд тестами ADF и KPSS."""
    x = s.dropna()
    adf = adfuller(x, regression=regression, autolag=autolag)
    with warnings.catch_warnings():
        # KPSS предупреждает, когда p-значение выходит за границы таблицы.
        warnings.simplefilter("ignore", InterpolationWarning)
        kp = kpss(x, regression=regression, nlags="auto")
    verdict = verdict_adf_kpss(adf[1], kp[1])
    return StationarityResult(name, adf[0], adf[1], adf[2], kp[0], kp[1], verdict)


def stationarity_table(series: dict[str, pd.Series], regression: str = "c", autolag: str = "AIC") -> pd.DataFrame:
    rows = [stationarity_tests(s, n, regression, autolag).__dict__ for n, s in series.items()]
    return pd.DataFrame(rows).set_index("name").round({"adf_stat": 4, "kpss_stat": 4})


def difference(s: pd.Series, lag: int = 1, order: int = 1) -> pd.Series:
    out = s.copy()
    for _ in range(order):
        out = out.diff(lag)
    return out.dropna()


def plot_series(s: pd.Series, title: str, ylabel: str = "Продано, шт.", ax=None,
                mark: pd.Series | None = None, label: str = "ряд"):
    ax = ax or plt.subplots(figsize=(11, 3.6))[1]
    ax.plot(s.index, s.values, lw=1, marker="o" if len(s) < 80 else None, ms=3, label=label)
    if mark is not None and len(mark):
        ax.scatter(mark.index, mark.values, c="crimson", zorder=3, label="выбросы (IQR)")
    ax.set(title=title, xlabel="Дата", ylabel=ylabel)
    ax.legend()
    return ax


def plot_missing(s: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(s.index, s.values, lw=0.8, label="продажи")
    na = s[s.isna()]
    ax.vlines(na.index, 0, np.nanmax(s.values), colors="orange", alpha=0.6, lw=1.2, label="нет транзакций (NaN)")
    ax.set(title=title, xlabel="Дата", ylabel="Продано, шт.")
    ax.legend()
    return ax


def plot_rolling(s: pd.Series, window: int, title: str):
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    axes[0].plot(s.index, s, lw=0.6, alpha=0.5, label="ряд")
    axes[0].plot(s.index, s.rolling(window, center=True).mean(), c="C1", label=f"скользящее среднее ({window})")
    axes[0].set(title=title, ylabel="Продано, шт.")
    axes[0].legend()
    axes[1].plot(s.index, s.rolling(window, center=True).std(), c="C2", label=f"скользящее СКО ({window})")
    axes[1].set(xlabel="Дата", ylabel="СКО, шт.")
    axes[1].legend()
    fig.tight_layout()
    return axes


def plot_acf_pacf(s: pd.Series, lags: int, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    plot_acf(s.dropna(), lags=lags, ax=axes[0], title=f"ACF: {title}")
    plot_pacf(s.dropna(), lags=lags, ax=axes[1], title=f"PACF: {title}", method="ywm")
    axes[0].set(xlabel="Лаг", ylabel="Автокорреляция")
    axes[1].set(xlabel="Лаг", ylabel="Частичная автокорреляция")
    fig.tight_layout()
    return axes


def plot_spectrum(sp: dict, title: str, xlabel="Частота (1/сутки)"):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.semilogy(sp["f_periodogram"][1:], sp["p_periodogram"][1:], lw=0.8, alpha=0.6, label="периодограмма")
    ax.semilogy(sp["f_welch"][1:], sp["p_welch"][1:], lw=1.8, label="Уэлч")
    ax.set(title=title, xlabel=xlabel, ylabel="Спектральная плотность мощности")
    ax.legend()
    return ax


def plot_decomposition(res, title: str, ylabel: str = "шт."):
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    obs = res.observed if hasattr(res, "observed") else None
    for ax, (data, name) in zip(axes, [(obs, "Наблюдаемый ряд"), (res.trend, "Тренд"),
                                       (res.seasonal, "Сезонность"), (res.resid, "Остаток")]):
        ax.plot(data.index, data.values, lw=0.9)
        ax.set_ylabel(f"{name}\n({ylabel})", fontsize=8)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Дата")
    fig.tight_layout()
    return axes


def run_pipeline(df: pd.DataFrame, code: str) -> dict:
    """Возвращает сводку по продажам одного товара."""
    daily = make_daily(df, code)
    filled = impute_closed_days(daily)
    weekly, monthly = aggregate(daily, "W"), aggregate(daily, "M")
    weekly_filled = impute_closed_days(weekly, method="linear")
    res = decompose(filled, period=6, model="multiplicative")
    return {
        "code": code,
        "описание": df.loc[df["code"] == code, "Description"].dropna().iloc[0],
        "дней в ряду / без транзакций (NaN)": f"{len(daily)} / {int(daily.isna().sum())}",
        "доля нулевых торговых дней": round(float((daily.dropna() == 0).mean()), 3),
        "выбросов IQR (день)": len(detect_outliers(daily, "iqr")),
        "CV (день)": round(float(describe_series(daily)["cv"]), 2),
        "p Краскела-Уоллиса по дням недели": float(f"{kruskal_by_weekday(daily)[1]:.2e}"),
        "сила сезонности STL (log1p)": round(seasonal_strength(res), 3),
        "стационарность (день)": stationarity_tests(filled, "d").verdict,
        "стационарность (неделя)": stationarity_tests(weekly_filled, "w").verdict,
        "точек в недельном / месячном ряду": f"{len(weekly)} / {len(monthly)}",
    }
