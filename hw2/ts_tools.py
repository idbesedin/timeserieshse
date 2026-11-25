"""Модели и оценка прогнозов для ДЗ 2."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


MODEL_COLORS = {"Наивная": "#79828e", "SES": "#d27a21", "CatBoost": "#256da6"}
LAGS = {"День": (1, 6, 12), "Неделя": (1, 2, 4), "Месяц": (1, 2)}


def fill_history(raw, frequency):
    """Заполняет пропуски по наблюдениям, известным до пропущенной даты."""
    past = raw.ffill()
    if frequency == "День":
        weekday_mean = raw.groupby(raw.index.dayofweek).transform(
            lambda values: values.expanding().mean().shift()
        )
        return raw.fillna(weekday_mean).fillna(past)
    return past


def naive_forecast(history, steps, frequency):
    """Повторяет последнюю торговую неделю для дней или последнее значение для других частот."""
    period = 6 if frequency == "День" else 1
    return np.resize(history.iloc[-period:].to_numpy(), steps)


def ses_forecast(history, steps):
    """Оценивает уровень продаж по обучающему ряду и сохраняет его на всём горизонте."""
    model = SimpleExpSmoothing(history.to_numpy(), initialization_method="estimated")
    return model.fit(optimized=True).forecast(steps)


def features(history, date, frequency):
    """Берёт лаги и среднее из истории перед прогнозной датой."""
    lags = LAGS[frequency]
    row = [history[-lag] for lag in lags]
    row.append(np.mean(history[-lags[-1]:]))
    if frequency == "День":
        row.extend(int(date.dayofweek == day) for day in (0, 1, 2, 3, 4, 6))
    return row


def boosting_forecast(raw, history, dates, frequency):
    """Обучает CatBoost на наблюдаемых продажах и добавляет каждый прогноз в историю лагов."""
    values = history.tolist()
    warmup = max(LAGS[frequency])
    positions = [t for t in range(warmup, len(raw)) if pd.notna(raw.iloc[t])]
    train_features = [features(values[:t], raw.index[t], frequency) for t in positions]
    model = CatBoostRegressor(
        iterations=120, depth=2, learning_rate=0.05, loss_function="RMSE",
        random_seed=42, thread_count=1, verbose=False, allow_writing_files=False,
    )
    model.fit(train_features, raw.iloc[positions])
    predictions = []
    for date in dates:
        estimate = max(0.0, float(model.predict(features(values, date, frequency))))
        predictions.append(estimate)
        values.append(estimate)
    return predictions


def training_sizes(length, max_horizon, n_origins):
    """Возвращает длины обучающих окон, после которых остаётся весь горизонт прогноза."""
    last = length - max_horizon
    return range(last - n_origins + 1, last + 1)


def backtest(series, frequency, sizes, steps):
    """Строит прогнозы трёх моделей после каждой границы обучения."""
    frames = []
    for cut in sizes:
        raw = series.iloc[:cut]
        history = fill_history(raw, frequency)
        actual = series.iloc[cut:cut + steps]
        predictions = {
            "Наивная": naive_forecast(history, steps, frequency),
            "SES": ses_forecast(history, steps),
            "CatBoost": boosting_forecast(raw, history, actual.index, frequency),
        }
        for model, forecast in predictions.items():
            frames.append(pd.DataFrame({
                "частота": frequency, "модель": model, "конец обучения": raw.index[-1],
                "шаг": np.arange(1, steps + 1), "дата": actual.index,
                "факт": actual.to_numpy(), "прогноз": forecast,
            }))
    return pd.concat(frames, ignore_index=True)


def score_forecasts(paths, horizons_by_frequency):
    """Считает ошибки отдельно для каждого шага по датам с известными продажами."""
    rows = []
    for frequency, horizons in horizons_by_frequency.items():
        subset = paths[(paths["частота"] == frequency) & paths["шаг"].isin(horizons)]
        for (step, model), group in subset.groupby(["шаг", "модель"]):
            observed = group.dropna(subset=["факт"])
            error = observed["прогноз"] - observed["факт"]
            volume = observed["факт"].abs().sum()
            rows.append({
                "частота": frequency, "шаг": step, "модель": model, "наблюдений": len(error),
                "MAE": error.abs().mean(), "RMSE": np.sqrt((error**2).mean()),
                "WAPE, %": 100 * error.abs().sum() / volume if volume else np.nan,
            })
    return pd.DataFrame(rows)


def plot_forecasts(series_by_frequency, forecasts):
    """Показывает продажи и прогнозы из последнего окна для каждой частоты."""
    history_lengths = {"День": 18, "Неделя": 8, "Месяц": 6}
    fig, axes = plt.subplots(len(series_by_frequency), 1, figsize=(11, 10), layout="constrained")
    for ax, (frequency, series) in zip(axes, series_by_frequency.items()):
        paths = forecasts[forecasts["частота"] == frequency]
        origin = paths["конец обучения"].max()
        last_window = paths[paths["конец обучения"] == origin]
        start = series.loc[:origin].tail(history_lengths[frequency]).index[0]
        ax.plot(series.loc[start:], color="black", marker=".", label="Факт")
        for model, color in MODEL_COLORS.items():
            line = last_window[last_window["модель"] == model]
            ax.plot(line["дата"], line["прогноз"], color=color, marker=".", label=model)
        ax.axvline(origin, color="#666666", linestyle=":", label="Конец обучения")
        ax.set(title=f"{frequency}. Прогноз после {origin:%d.%m.%Y}", xlabel="Дата",
               ylabel="Продажи, шт.")
        ax.legend(ncol=5, fontsize=8)
    plt.show()


def plot_relative_errors(scores, horizons_by_frequency):
    """Сопоставляет WAPE моделей на общей процентной шкале."""
    fig, axes = plt.subplots(1, len(horizons_by_frequency), figsize=(12, 3.7),
                             sharey=True, layout="constrained")
    for ax, (frequency, horizons) in zip(axes, horizons_by_frequency.items()):
        for model, color in MODEL_COLORS.items():
            rows = scores[(scores["частота"] == frequency) & (scores["модель"] == model)]
            ax.plot(rows["шаг"], rows["WAPE, %"], color=color, marker="o", label=model)
        ax.set(title=frequency, xticks=list(horizons), xlabel="Горизонт, шаги")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("WAPE, %")
    fig.suptitle("Относительная ошибка по всем окнам")
    plt.show()
