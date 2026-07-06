import re
import warnings
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_squared_error

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor
    LGBM_AVAILABLE = False

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ドコモでんきコール予測アプリVer1",
    page_icon="📞",
    layout="wide"
)

# ============ 環境判定 ============
def detect_environment():
    """ローカル環境かクラウド環境かを自動判定"""
    local_path = Path(r"C:\Users\1991000")
    if local_path.exists() and os.name == 'nt':
        return "local"
    return "cloud"

ENV = detect_environment()
IS_LOCAL = ENV == "local"
IS_CLOUD = ENV == "cloud"


# ============ 定数 ============
NEW_SALES_CALL_RATE = 0.08
SUBSCRIBER_CALL_RATE = 0.01
ANCHOR_YEAR = 2025
ANCHOR_MONTH = 5

GW_START_MONTH = 5
GW_START_DAY = 3
GW_END_MONTH = 5
GW_END_DAY = 6

DEFAULT_RECENCY_HALFLIFE_DAYS = 180
DEFAULT_OLD_DATA_CUTOFF = "2025-01-01"
DEFAULT_OLD_DATA_WEIGHT = 0.30
DEFAULT_2025_MAY_WEIGHT = 5.00
DEFAULT_SAME_MONTH_WEIGHT = 1.50

DEFAULT_GW_REL_SHAPE_STRENGTH = 0.65
DEFAULT_GW_REL_SHAPE_MIN = 0.85
DEFAULT_GW_REL_SHAPE_MAX = 1.60
DEFAULT_MAY_PRE_GW_LAST2_LIFT = 1.20
DEFAULT_MAY_GW_CORE_DAMPEN = 1.00
DEFAULT_MAY_POST_1_3_LIFT = 1.08
DEFAULT_MAY_END_DAMPEN = 0.90

CATEGORY_MODEL_PARAMS = {
    "料金・解約・その他": {"num_leaves": 63, "learning_rate": 0.02, "min_child_samples": 5, "n_estimators": 1200},
    "料金": {"num_leaves": 63, "learning_rate": 0.02, "min_child_samples": 5, "n_estimators": 1200},
    "解約": {"num_leaves": 63, "learning_rate": 0.02, "min_child_samples": 5, "n_estimators": 1200},
    "新規関連": {"num_leaves": 31, "learning_rate": 0.025, "min_child_samples": 8, "n_estimators": 900},
    "引っ越し・プラン変更": {"num_leaves": 31, "learning_rate": 0.03, "min_child_samples": 8, "n_estimators": 900},
    "default": {"num_leaves": 31, "learning_rate": 0.025, "min_child_samples": 8, "n_estimators": 950},
}

WEEKDAY_MAP = {0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"}

# ============ 保存設定（環境で分岐） ============
if IS_LOCAL:
    SAVE_DIR = Path(r"C:\Users\1991000\docomo_call_forecast")
else:
    SAVE_DIR = Path("./saved_data")

try:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_ENABLED = True
except Exception:
    SAVE_ENABLED = False

HISTORY_LATEST = SAVE_DIR / "latest_history.tsv"
MASTER_LATEST = SAVE_DIR / "latest_master.tsv"
MAX_BACKUPS = 10


# ============ セッション初期化 ============
if "master_df" not in st.session_state:
    st.session_state["master_df"] = None
if "history_df" not in st.session_state:
    st.session_state["history_df"] = None
if "history_source" not in st.session_state:
    st.session_state["history_source"] = None
if "auto_loaded" not in st.session_state:
    st.session_state["auto_loaded"] = False


# ============ 基本関数 ============
def month_position(day: int) -> str:
    if day <= 10:
        return "初"
    if day <= 20:
        return "中"
    return "末"


def month_range(start_date, end_date):
    start = pd.Timestamp(start_date).to_period("M").to_timestamp()
    end = pd.Timestamp(end_date).to_period("M").to_timestamp()
    return list(pd.date_range(start=start, end=end, freq="MS"))


def make_target_month_frame(target_month, category_name):
    start = pd.Timestamp(target_month).to_period("M").to_timestamp()
    end = start + pd.offsets.MonthEnd(1)
    out = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    out["category"] = category_name
    return out


def normalize_boost(x: float) -> float:
    return 1.0 if float(x) == 0 else float(x)


def parse_event_dates(text: str) -> List[pd.Timestamp]:
    if not text or not text.strip():
        return []
    out = []
    for s in re.split(r"[,\n、\s]+", text.strip()):
        if not s:
            continue
        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            out.append(pd.Timestamp(dt).normalize())
    return sorted(list(set(out)))


def split_line(line: str):
    line = line.strip()
    if "\t" in line:
        return [x.strip() for x in line.split("\t") if x.strip()]
    if "," in line:
        return [x.strip() for x in line.split(",") if x.strip()]
    parts = [x.strip() for x in re.split(r"\s{2,}", line) if x.strip()]
    if len(parts) >= 2:
        return parts
    return [x.strip() for x in re.split(r"\s+", line) if x.strip()]


def is_history_header(parts):
    text = "".join(parts).lower()
    return any(k in text for k in ["日付", "カテゴリー", "着信数", "応答数", "date", "category", "call_count", "answered_count"])


def parse_history_text(text: str) -> pd.DataFrame:
    if not text or not text.strip():
        raise ValueError("データが空です。")
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [ln for ln in lines if ln.strip()]
    rows = [split_line(ln) for ln in lines]
    rows = [r for r in rows if r]
    if not rows:
        raise ValueError("行を解釈できませんでした。")

    if is_history_header(rows[0]):
        header = rows[0]
        data_rows = rows[1:]
        cols = []
        for h in header:
            hl = h.lower()
            if h == "日付" or hl == "date":
                cols.append("date")
            elif h == "カテゴリー" or hl == "category":
                cols.append("category")
            elif h == "着信数" or hl == "call_count":
                cols.append("call_count")
            elif h == "応答数" or hl == "answered_count":
                cols.append("answered_count")
            else:
                cols.append(h)
        fixed = [(r + [None] * len(cols))[:len(cols)] for r in data_rows]
        df = pd.DataFrame(fixed, columns=cols)
    else:
        max_len = max(len(r) for r in rows)
        if max_len == 4:
            df = pd.DataFrame([r[:4] for r in rows if len(r) >= 4], columns=["date", "category", "call_count", "answered_count"])
        elif max_len == 3:
            df = pd.DataFrame([r[:3] for r in rows if len(r) >= 3], columns=["date", "call_count", "answered_count"])
        else:
            fixed = []
            for r in rows:
                if len(r) >= 4:
                    fixed.append(r[:4])
                elif len(r) == 3:
                    fixed.append([r[0], None, r[1], r[2]])
            if not fixed:
                raise ValueError("列数を判定できませんでした。")
            df = pd.DataFrame(fixed, columns=["date", "category", "call_count", "answered_count"])

    if "date" not in df.columns or "call_count" not in df.columns:
        raise ValueError("date / call_count 列が見つかりません。")
    df = df[df["date"].astype(str).str.lower() != "date"]
    df = df[df["date"].astype(str) != "日付"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["call_count"] = pd.to_numeric(df["call_count"], errors="coerce")
    if "answered_count" in df.columns:
        df["answered_count"] = pd.to_numeric(df["answered_count"], errors="coerce")
    df["category"] = df["category"].astype(str).replace("nan", "合計") if "category" in df.columns else "合計"
    df = df.dropna(subset=["date", "call_count"]).copy()
    if df.empty:
        raise ValueError("有効なデータがありません。")
    return df


def parse_master_text(text: str) -> pd.DataFrame:
    if not text or not text.strip():
        raise ValueError("マスタ入力が空です。")
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [ln for ln in lines if ln.strip()]
    rows = [split_line(ln) for ln in lines]
    header = [h.strip().lower() for h in rows[0]]
    if "yyyymm" not in header or "subscriber_base" not in header or "new_sales" not in header:
        raise ValueError("マスタには yyyymm, subscriber_base, new_sales が必要です。")
    df = pd.DataFrame(rows[1:], columns=header)
    if "cancel_count" not in df.columns:
        df["cancel_count"] = 0
    if "bill_issue_day" not in df.columns:
        df["bill_issue_day"] = 10
    df["yyyymm"] = df["yyyymm"].astype(str).str.replace("/", "", regex=False).str.replace("-", "", regex=False)
    for c in ["subscriber_base", "new_sales", "cancel_count", "bill_issue_day"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["cancel_count"] = df["cancel_count"].fillna(0)
    df["bill_issue_day"] = df["bill_issue_day"].fillna(10)
    df = df.dropna(subset=["subscriber_base", "new_sales"]).copy()
    df["subscriber_call_base"] = df["subscriber_base"] * SUBSCRIBER_CALL_RATE
    df["new_sales_call_base"] = df["new_sales"] * NEW_SALES_CALL_RATE
    df["log_subscriber_call_base"] = np.log1p(df["subscriber_call_base"])
    df["log_new_sales_call_base"] = np.log1p(df["new_sales_call_base"])
    return df.sort_values("yyyymm").reset_index(drop=True)


def get_jp_holiday_flag(dates: pd.Series) -> pd.Series:
    try:
        import jpholiday
        return dates.dt.date.apply(lambda x: 1 if jpholiday.is_holiday(x) else 0).astype(int)
    except Exception:
        return pd.Series(0, index=dates.index, dtype=int)


def add_fixed_gw_split_features(date_map: pd.DataFrame) -> pd.DataFrame:
    d = date_map.copy()
    gw_start = pd.to_datetime(d["year"].astype(str) + f"-{GW_START_MONTH:02d}-{GW_START_DAY:02d}")
    gw_end = pd.to_datetime(d["year"].astype(str) + f"-{GW_END_MONTH:02d}-{GW_END_DAY:02d}")
    days_to_start = (gw_start - d["date"]).dt.days.astype(int)
    days_from_end = (d["date"] - gw_end).dt.days.astype(int)
    rel_end = (d["date"] - gw_end).dt.days.astype(int)

    d["days_to_gw_start"] = days_to_start
    d["days_from_gw_end"] = days_from_end
    d["gw_relative_day"] = rel_end
    d["gw_relative_abs"] = rel_end.abs()

    d["gw_pre_last_2days_flag"] = ((days_to_start >= 1) & (days_to_start <= 2)).astype(int)
    d["gw_pre_3_5days_flag"] = ((days_to_start >= 3) & (days_to_start <= 5)).astype(int)
    d["gw_core_holiday_flag"] = ((days_to_start <= 0) & (days_from_end <= 0)).astype(int)
    d["gw_post_1_3_flag"] = ((days_from_end >= 1) & (days_from_end <= 3)).astype(int)
    d["gw_post_4_7_flag"] = ((days_from_end >= 4) & (days_from_end <= 7)).astype(int)
    d["gw_post_8_14_flag"] = ((days_from_end >= 8) & (days_from_end <= 14)).astype(int)
    d["gw_near_flag"] = ((days_to_start >= -14) & (days_from_end <= 14)).astype(int)

    for k in range(-14, 22):
        col = f"gw_rel_{k:+d}".replace("+", "p").replace("-", "m")
        d[col] = (rel_end == k).astype(int)
    return d


def month_feature_frame(dates: pd.Series, event_dates: Optional[List[pd.Timestamp]] = None) -> pd.DataFrame:
    date_map = pd.DataFrame({"date": pd.to_datetime(pd.Series(dates).dropna().dt.normalize().unique())})
    date_map = date_map.sort_values("date").reset_index(drop=True)
    date_map["year"] = date_map["date"].dt.year
    date_map["month"] = date_map["date"].dt.month
    date_map["day"] = date_map["date"].dt.day
    date_map["weekday"] = date_map["date"].dt.weekday
    date_map["weekday_name"] = date_map["weekday"].map(WEEKDAY_MAP)
    date_map["month_pos"] = date_map["day"].apply(month_position)
    date_map["month_start_flag"] = (date_map["day"] <= 5).astype(int)
    date_map["month_end_flag"] = (date_map["day"] >= 25).astype(int)
    date_map["month_start_peak"] = (date_map["day"] <= 3).astype(int)
    date_map["month_end_peak"] = (date_map["day"] >= 28).astype(int)
    date_map["billing_window_flag"] = ((date_map["day"] >= 10) & (date_map["day"] <= 20)).astype(int)
    date_map["holiday_flag"] = get_jp_holiday_flag(date_map["date"])
    date_map["month_sin"] = np.sin(2 * np.pi * date_map["month"] / 12)
    date_map["month_cos"] = np.cos(2 * np.pi * date_map["month"] / 12)
    date_map["fiscal_year_start_flag"] = ((date_map["month"] == 4) & (date_map["day"] <= 10)).astype(int)
    date_map["april_flag"] = (date_map["month"] == 4).astype(int)
    date_map["january_flag"] = (date_map["month"] == 1).astype(int)
    date_map["move_season_flag"] = ((date_map["month"] >= 3) & (date_map["month"] <= 5)).astype(int)
    date_map["may_flag"] = (date_map["month"] == 5).astype(int)
    date_map = add_fixed_gw_split_features(date_map)
    if event_dates:
        event_set = {pd.Timestamp(x).normalize() for x in event_dates}
        date_map["event_flag"] = date_map["date"].apply(lambda x: 1 if any(abs((x - ev).days) <= 2 for ev in event_set) else 0).astype(int)
    else:
        date_map["event_flag"] = 0
    return date_map


def attach_master_features(df: pd.DataFrame, master_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    d = df.copy()
    d["yyyymm"] = d["date"].dt.strftime("%Y%m")
    d["yyyymm"] = d["yyyymm"].astype(str)

    if master_df is None or master_df.empty:
        for c, v in {"subscriber_base": np.nan, "new_sales": np.nan, "cancel_count": 0, "bill_issue_day": 10,
                     "subscriber_call_base": 0.0, "new_sales_call_base": 0.0, "log_subscriber_call_base": 0.0, "log_new_sales_call_base": 0.0}.items():
            d[c] = v
        return d

    master_df = master_df.copy()
    master_df["yyyymm"] = master_df["yyyymm"].astype(str).str.replace(".0", "", regex=False)

    keep = ["yyyymm", "subscriber_base", "new_sales", "cancel_count", "bill_issue_day",
            "subscriber_call_base", "new_sales_call_base", "log_subscriber_call_base", "log_new_sales_call_base"]
    d = d.merge(master_df[keep], on="yyyymm", how="left")
    for c in ["cancel_count", "subscriber_call_base", "new_sales_call_base", "log_subscriber_call_base", "log_new_sales_call_base"]:
        d[c] = d[c].fillna(0)
    d["bill_issue_day"] = d["bill_issue_day"].fillna(10)
    return d


def build_features(df: pd.DataFrame, master_df: Optional[pd.DataFrame] = None, event_dates: Optional[List[pd.Timestamp]] = None) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["date_key"] = d["date"].dt.normalize()
    fmap = month_feature_frame(d["date"], event_dates=event_dates).rename(columns={"date": "date_key"})
    d = d.merge(fmap, on="date_key", how="left").drop(columns=["date_key"])
    d = attach_master_features(d, master_df)
    d["days_from_bill_issue"] = (d["day"] - d["bill_issue_day"]).astype(float)
    return d


def calc_2025_gw_relative_shape_map(train_df: pd.DataFrame) -> Dict[int, float]:
    anchor = train_df[(train_df["date"].dt.year == 2025) & (train_df["date"].dt.month == 5)].copy()
    if anchor.empty or "gw_relative_day" not in anchor.columns:
        return {}
    m = anchor["call_count"].mean()
    if pd.isna(m) or m <= 0:
        return {}
    shape = anchor.groupby("gw_relative_day")["call_count"].mean() / m
    return {int(k): float(v) for k, v in shape.to_dict().items()}


def apply_gw_relative_shape_control(df, shape_map, strength, min_factor, max_factor, pre_last2_lift, core_dampen, post13_lift, end_dampen):
    d = df.copy()
    for c in ["gwrel_shape_factor_raw", "gwrel_shape_factor", "pre_last2_lift_factor", "gw_core_dampen_factor", "post13_lift_factor", "may_end_dampen_factor"]:
        d[c] = 1.0
    is_may = d["date"].dt.month == 5
    if shape_map and strength > 0 and "gw_relative_day" in d.columns:
        raw = d["gw_relative_day"].map(shape_map).fillna(1.0).astype(float).clip(min_factor, max_factor)
        blended = 1.0 + strength * (raw - 1.0)
        d.loc[is_may, "gwrel_shape_factor_raw"] = raw[is_may]
        d.loc[is_may, "gwrel_shape_factor"] = blended[is_may]
        d.loc[is_may, "predicted"] *= d.loc[is_may, "gwrel_shape_factor"]

    pre_mask = is_may & (d.get("gw_pre_last_2days_flag", 0) == 1)
    d.loc[pre_mask, "pre_last2_lift_factor"] = pre_last2_lift
    d.loc[pre_mask, "predicted"] *= pre_last2_lift

    core_mask = is_may & (d.get("gw_core_holiday_flag", 0) == 1)
    d.loc[core_mask, "gw_core_dampen_factor"] = core_dampen
    d.loc[core_mask, "predicted"] *= core_dampen

    post_mask = is_may & (d.get("gw_post_1_3_flag", 0) == 1)
    d.loc[post_mask, "post13_lift_factor"] = post13_lift
    d.loc[post_mask, "predicted"] *= post13_lift

    end_mask = is_may & (d["date"].dt.day >= 25)
    d.loc[end_mask, "may_end_dampen_factor"] = end_dampen
    d.loc[end_mask, "predicted"] *= end_dampen
    return d


def make_lgbm_X(df):
    cols = [
        "year", "month", "day", "weekday", "month_start_flag", "month_end_flag", "month_start_peak", "month_end_peak",
        "holiday_flag", "billing_window_flag", "days_from_bill_issue",
        "log_subscriber_call_base", "log_new_sales_call_base", "cancel_count",
        "month_sin", "month_cos", "fiscal_year_start_flag", "april_flag", "january_flag", "move_season_flag", "may_flag", "event_flag",
        "days_to_gw_start", "days_from_gw_end", "gw_relative_day", "gw_relative_abs",
        "gw_pre_last_2days_flag", "gw_pre_3_5days_flag", "gw_core_holiday_flag", "gw_post_1_3_flag", "gw_post_4_7_flag", "gw_post_8_14_flag", "gw_near_flag",
    ]
    for k in range(-14, 22):
        cols.append(f"gw_rel_{k:+d}".replace("+", "p").replace("-", "m"))
    base = df.reindex(columns=cols, fill_value=0)
    wd = pd.get_dummies(df["weekday_name"], prefix="wd")
    mp = pd.get_dummies(df["month_pos"], prefix="mp")
    return pd.concat([base, wd, mp], axis=1).astype(float).fillna(0)


def make_weights(d, half_life_days, old_cutoff, old_weight, may2025_weight, same_month_weight, target_month):
    max_date = d["date"].max()
    age = (max_date - d["date"]).dt.days.clip(lower=0).astype(float)
    w = np.power(0.5, age / max(1, int(half_life_days)))
    if old_cutoff is not None:
        w = np.where(d["date"] < old_cutoff, w * old_weight, w)
    w = np.where((d["date"].dt.year == 2025) & (d["date"].dt.month == 5), w * may2025_weight, w)
    w = np.where(d["date"].dt.month == pd.Timestamp(target_month).month, w * same_month_weight, w)
    return np.asarray(w, dtype=float)


def get_category_params(category: str) -> dict:
    if category in CATEGORY_MODEL_PARAMS:
        return CATEGORY_MODEL_PARAMS[category]
    return CATEGORY_MODEL_PARAMS["default"]


def fit_model_v18(train_df, pred_df, category, half_life_days, old_cutoff, old_weight, may2025_weight, same_month_weight, target_month,
                  use_log_transform=True, use_category_params=True):
    d = train_df.copy().sort_values("date").reset_index(drop=True)
    if len(d) < 45:
        out = pred_df.copy()
        out["predicted"] = d["call_count"].mean() if not d.empty else 0
        out["model_name"] = "平均代替"
        return out

    X_train = make_lgbm_X(d)
    X_pred = make_lgbm_X(pred_df).reindex(columns=X_train.columns, fill_value=0)
    y_raw = d["call_count"].clip(lower=0).astype(float).values

    if use_log_transform:
        y = np.log1p(y_raw)
    else:
        y = y_raw

    w = make_weights(d, half_life_days, old_cutoff, old_weight, may2025_weight, same_month_weight, target_month)

    if use_category_params:
        params = get_category_params(category)
    else:
        params = CATEGORY_MODEL_PARAMS["default"]

    if LGBM_AVAILABLE:
        model = LGBMRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            max_depth=-1,
            min_child_samples=params["min_child_samples"],
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="regression",
            random_state=42,
            verbose=-1,
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_leaf_nodes=params["num_leaves"],
            l2_regularization=1.0,
            random_state=42,
        )

    try:
        model.fit(X_train, y, sample_weight=w)
    except TypeError:
        model.fit(X_train, y)

    pred_raw = np.asarray(model.predict(X_pred))
    if use_log_transform:
        pred = np.expm1(pred_raw).clip(min=0)
    else:
        pred = pred_raw.clip(min=0)

    out = pred_df.copy()
    out["predicted"] = pred
    out["model_name"] = f"LightGBM v18 [{category}]" if LGBM_AVAILABLE else f"HGB v18 [{category}]"
    return out


def compute_bias_from_prior_year_same_month(train_df, target_month):
    target_dt = pd.Timestamp(target_month)
    prior_year_dt = target_dt - pd.DateOffset(years=1)
    prior_data = train_df[(train_df["date"].dt.year == prior_year_dt.year) & (train_df["date"].dt.month == prior_year_dt.month)]
    if prior_data.empty:
        return 0.0
    return float(prior_data["call_count"].mean() * 0.05)


def apply_bias_correction(forecast_df, bias_value):
    d = forecast_df.copy()
    d["bias_correction"] = bias_value
    d["predicted"] = (d["predicted"] + bias_value).clip(lower=0)
    return d


def apply_manual_adjustment(df, month_start_boost, month_end_boost, event_boost):
    d = df.copy()
    d.loc[d["date"].dt.day <= 5, "predicted"] *= normalize_boost(month_start_boost)
    d.loc[d["date"].dt.day >= 25, "predicted"] *= normalize_boost(month_end_boost)
    if "event_flag" in d.columns:
        d.loc[d["event_flag"] == 1, "predicted"] *= normalize_boost(event_boost)
    return d


def build_forecast_by_category(train_cat_df, target_month, half_life_days, old_cutoff, old_weight, may2025_weight, same_month_weight,
                               gw_shape_strength, gw_shape_min, gw_shape_max, pre_last2_lift, core_dampen, post13_lift, end_dampen,
                               month_start_boost, month_end_boost, event_boost, event_dates,
                               use_log_transform=True, use_category_params=True,
                               use_bias_correction=True):
    cats = sorted(train_cat_df["category"].dropna().astype(str).unique().tolist())
    forecasts = []
    bias_info = []

    for cat in cats:
        train = train_cat_df[train_cat_df["category"] == cat].copy()
        pred = make_target_month_frame(target_month, cat)
        pred = build_features(pred, st.session_state.get("master_df"), event_dates)

        out = fit_model_v18(
            train, pred, cat,
            half_life_days, old_cutoff, old_weight, may2025_weight, same_month_weight, target_month,
            use_log_transform=use_log_transform,
            use_category_params=use_category_params,
        )
        out["used_model"] = out.get("model_name", f"LightGBM v18 [{cat}]")

        for c in pred.columns:
            if c not in out.columns and c not in ["call_count"]:
                out[c] = pred[c].values

        shape_map = calc_2025_gw_relative_shape_map(train)

        out = apply_gw_relative_shape_control(
            out, shape_map, gw_shape_strength, gw_shape_min, gw_shape_max,
            pre_last2_lift, core_dampen, post13_lift, end_dampen
        )

        if use_bias_correction:
            bias_val = compute_bias_from_prior_year_same_month(train, target_month)
            out = apply_bias_correction(out, bias_val)
            bias_info.append({"category": cat, "bias_correction": bias_val})

        out = apply_manual_adjustment(out, month_start_boost, month_end_boost, event_boost)
        out["category"] = cat
        out.attrs = {}
        forecasts.append(out)

    if not forecasts:
        return pd.DataFrame(), pd.DataFrame(columns=["date", "predicted"]), pd.DataFrame()

    for f in forecasts:
        f.attrs = {}
    forecast_cat = pd.concat(forecasts, ignore_index=True)
    forecast_cat.attrs = {}
    forecast_total = forecast_cat.groupby("date", as_index=False)["predicted"].sum().sort_values("date").reset_index(drop=True)
    forecast_total.attrs = {}
    return forecast_cat, forecast_total, pd.DataFrame(bias_info)


def compute_metrics(compare_df):
    d = compare_df.dropna(subset=["actual"]).copy()
    if d.empty:
        return None, None
    denom = d["actual"].replace(0, np.nan)
    d["abs_pct_error"] = (d["actual"] - d["predicted"]).abs() / denom * 100
    d["error"] = d["actual"] - d["predicted"]
    d["sq_error"] = d["error"] ** 2
    metrics = {
        "MAPE": float(d["abs_pct_error"].dropna().mean()),
        "RMSE": float(np.sqrt(mean_squared_error(d["actual"], d["predicted"]))),
        "Bias": float(d["error"].mean()),
    }
    running = d.sort_values("date").copy()
    running["cumulative_mape"] = [running.iloc[:i+1]["abs_pct_error"].dropna().mean() for i in range(len(running))]
    running["cumulative_rmse"] = [np.sqrt(running.iloc[:i+1]["sq_error"].mean()) for i in range(len(running))]
    running["cumulative_bias"] = [running.iloc[:i+1]["error"].mean() for i in range(len(running))]
    return metrics, running


def read_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    try:
        content = uploaded_file.read().decode("utf-8-sig")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        content = uploaded_file.read().decode("shift_jis")
    return content


# ============ ローカル保存機能 ============
def save_dataframe_to_local(df: pd.DataFrame, latest_path: Path, prefix: str) -> tuple:
    if not SAVE_ENABLED:
        return None, None
    df.to_csv(latest_path, sep="\t", index=False, encoding="utf-8-sig")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = SAVE_DIR / f"{prefix}_{timestamp}.tsv"
    df.to_csv(backup_path, sep="\t", index=False, encoding="utf-8-sig")
    backups = sorted(SAVE_DIR.glob(f"{prefix}_*.tsv"), reverse=True)
    for old in backups[MAX_BACKUPS:]:
        try:
            old.unlink()
        except Exception:
            pass
    return latest_path, backup_path


def load_dataframe_from_local(path: Path) -> Optional[pd.DataFrame]:
    if not SAVE_ENABLED or not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    except Exception as e:
        st.warning(f"読込エラー: {e}")
        return None


def get_backup_list(prefix: str) -> list:
    if not SAVE_ENABLED:
        return []
    backups = sorted(SAVE_DIR.glob(f"{prefix}_*.tsv"), reverse=True)
    return [b.name for b in backups]


def delete_local_data(path: Path) -> bool:
    if not SAVE_ENABLED:
        return False
    try:
        if path.exists():
            path.unlink()
        return True
    except Exception:
        return False


# ============ 起動時の自動読込 ============
if not st.session_state["auto_loaded"]:
    if st.session_state.get("history_df") is None:
        saved_history = load_dataframe_from_local(HISTORY_LATEST)
        if saved_history is not None and not saved_history.empty:
            if "date" in saved_history.columns:
                saved_history["date"] = pd.to_datetime(saved_history["date"], errors="coerce")
            if "call_count" in saved_history.columns:
                saved_history["call_count"] = pd.to_numeric(saved_history["call_count"], errors="coerce")
            if "answered_count" in saved_history.columns:
                saved_history["answered_count"] = pd.to_numeric(saved_history["answered_count"], errors="coerce")
            if "category" in saved_history.columns:
                saved_history["category"] = saved_history["category"].astype(str).replace("nan", "合計")
            saved_history = saved_history.dropna(subset=["date", "call_count"]).copy()

            st.session_state["history_df"] = saved_history
            st.session_state["history_source"] = "💾 前回保存データ（自動読込）"

    if st.session_state.get("master_df") is None:
        saved_master = load_dataframe_from_local(MASTER_LATEST)
        if saved_master is not None and not saved_master.empty:
            if "yyyymm" in saved_master.columns:
                saved_master["yyyymm"] = saved_master["yyyymm"].astype(str).str.replace(".0", "", regex=False)

            for c in ["subscriber_base", "new_sales", "cancel_count", "bill_issue_day"]:
                if c in saved_master.columns:
                    saved_master[c] = pd.to_numeric(saved_master[c], errors="coerce")
            if "cancel_count" in saved_master.columns:
                saved_master["cancel_count"] = saved_master["cancel_count"].fillna(0)
            if "bill_issue_day" in saved_master.columns:
                saved_master["bill_issue_day"] = saved_master["bill_issue_day"].fillna(10)

            if "subscriber_call_base" not in saved_master.columns:
                saved_master["subscriber_call_base"] = saved_master["subscriber_base"] * SUBSCRIBER_CALL_RATE
                saved_master["new_sales_call_base"] = saved_master["new_sales"] * NEW_SALES_CALL_RATE
                saved_master["log_subscriber_call_base"] = np.log1p(saved_master["subscriber_call_base"])
                saved_master["log_new_sales_call_base"] = np.log1p(saved_master["new_sales_call_base"])

            st.session_state["master_df"] = saved_master

    st.session_state["auto_loaded"] = True


# ============ サンプルテンプレート ============
SAMPLE_HISTORY_TSV = """date\tcategory\tcall_count
2024-04-01\t料金・解約・その他\t850
2024-04-01\t新規関連\t320
2024-04-01\t引っ越し・プラン変更\t180
2024-04-02\t料金・解約・その他\t880
2024-04-02\t新規関連\t340
2024-04-02\t引っ越し・プラン変更\t190
"""

SAMPLE_MASTER_TSV = """yyyymm\tsubscriber_base\tnew_sales\tcancel_count\tbill_issue_day
202404\t1200000\t55000\t0\t1
202405\t1250000\t58000\t0\t1
202406\t1300000\t60000\t0\t1
"""


# ============================================
# UI 開始
# ============================================
st.title("📞 ドコモでんきコール予測アプリVer1")
st.caption("3ステップで予測 → 結果確認 → CSVダウンロード")

# 環境情報表示
if IS_CLOUD:
    st.info("☁️ クラウド環境で稼働中。データはセッション中のみ保持されます。")

# ============ サイドバー ============
with st.sidebar:
    st.markdown("### ⚙️ 設定")

    mode = st.radio(
        "モード選択",
        ["🟢 かんたん", "🟡 標準", "🔴 エキスパート"],
        index=0,
        help="かんたん：推奨設定固定 / 標準：主要設定 / エキスパート：全設定"
    )

    st.markdown("---")

    if "🟡" in mode or "🔴" in mode:
        st.markdown("#### 🔧 モデル設定")
        use_bias_correction = st.checkbox("Bias補正", value=True, help="前年同月からの傾向補正")
        use_log_transform = st.checkbox("log変換", value=True, help="ピーク再現性が上がる")
        use_category_params = st.checkbox("カテゴリ別モデル", value=True, help="料金/新規/引越し別に最適化")
    else:
        use_bias_correction = True
        use_log_transform = True
        use_category_params = True

    if "🔴" in mode:
        st.markdown("---")
        st.markdown("#### 🎓 詳細パラメータ")

        with st.expander("直近・重点学習"):
            half_life_days = st.slider("直近重み半減期(日)", 30, 365, DEFAULT_RECENCY_HALFLIFE_DAYS)
            old_cutoff = pd.Timestamp(st.text_input("低ウェイト開始日", value=DEFAULT_OLD_DATA_CUTOFF))
            old_weight = st.slider("古い期間ウェイト", 0.05, 1.0, DEFAULT_OLD_DATA_WEIGHT)
            may2025_weight = st.slider("2025年5月重点", 1.0, 10.0, DEFAULT_2025_MAY_WEIGHT)
            same_month_weight = st.slider("同月過去ウェイト", 1.0, 5.0, DEFAULT_SAME_MONTH_WEIGHT)

        with st.expander("GW前後補正"):
            gw_shape_strength = st.slider("GW shape strength", 0.0, 1.0, DEFAULT_GW_REL_SHAPE_STRENGTH)
            gw_shape_min = st.slider("GW shape min", 0.5, 1.0, DEFAULT_GW_REL_SHAPE_MIN)
            gw_shape_max = st.slider("GW shape max", 1.0, 2.5, DEFAULT_GW_REL_SHAPE_MAX)
            pre_last2_lift = st.slider("GW直前2日lift", 1.0, 2.0, DEFAULT_MAY_PRE_GW_LAST2_LIFT)
            core_dampen = st.slider("GW中dampen", 0.5, 1.2, DEFAULT_MAY_GW_CORE_DAMPEN)
            post13_lift = st.slider("GW後1-3日lift", 1.0, 2.0, DEFAULT_MAY_POST_1_3_LIFT)
            end_dampen = st.slider("月末dampen", 0.5, 1.2, DEFAULT_MAY_END_DAMPEN)

        with st.expander("手動補正"):
            month_start_boost = st.slider("月初補正", 0.5, 2.0, 1.0)
            month_end_boost = st.slider("月末補正", 0.5, 2.0, 1.0)
            event_boost = st.slider("イベント補正", 0.5, 2.0, 1.0)
            event_dates_text = st.text_area("イベント日", value="")
            event_dates = parse_event_dates(event_dates_text)
    else:
        half_life_days = DEFAULT_RECENCY_HALFLIFE_DAYS
        old_cutoff = pd.Timestamp(DEFAULT_OLD_DATA_CUTOFF)
        old_weight = DEFAULT_OLD_DATA_WEIGHT
        may2025_weight = DEFAULT_2025_MAY_WEIGHT
        same_month_weight = DEFAULT_SAME_MONTH_WEIGHT
        gw_shape_strength = DEFAULT_GW_REL_SHAPE_STRENGTH
        gw_shape_min = DEFAULT_GW_REL_SHAPE_MIN
        gw_shape_max = DEFAULT_GW_REL_SHAPE_MAX
        pre_last2_lift = DEFAULT_MAY_PRE_GW_LAST2_LIFT
        core_dampen = DEFAULT_MAY_GW_CORE_DAMPEN
        post13_lift = DEFAULT_MAY_POST_1_3_LIFT
        end_dampen = DEFAULT_MAY_END_DAMPEN
        month_start_boost = 1.0
        month_end_boost = 1.0
        event_boost = 1.0
        event_dates = []

    st.markdown("---")
    if IS_LOCAL:
        st.markdown("#### 📁 データ保存先")
        st.caption(f"`{SAVE_DIR}`")
    else:
        st.markdown("#### ☁️ 環境情報")
        st.caption("クラウド環境（セッション保持のみ）")

    st.markdown("---")
    st.markdown("#### ℹ️ モデル情報")
    st.caption(f"予測手法: **LightGBM v18.2**" if LGBM_AVAILABLE else "予測手法: **HGB v18.2**")
    st.caption("学習: カテゴリ別 + Bias補正 + log変換")
    st.caption("特徴量: GW前後分割・請求サイクル・契約者数連携")


# ============================================
# メインエリア: ステップ形式
# ============================================

# ============ ステップ1: データ準備 ============
st.markdown("## 🔵 ステップ 1: データを準備する")

with st.container():
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("**📁 履歴データ（TSV/CSV）をアップロード**")

        input_method = st.radio(
            "入力方式",
            ["📁 ファイルアップロード", "📝 テキスト貼り付け"],
            horizontal=True,
            key="input_method"
        )

        history_text = ""

        if input_method == "📁 ファイルアップロード":
            uploaded_history = st.file_uploader(
                "履歴データ",
                type=["tsv", "csv", "txt"],
                key="history_upload",
                help="TSV（タブ区切り）推奨・CSVも対応"
            )
            if uploaded_history is not None:
                history_text = read_uploaded_file(uploaded_history)
                st.session_state["history_source"] = f"📎 {uploaded_history.name}"
        else:
            history_text = st.text_area("履歴データを貼り付け", height=200, key="history_paste")
            if history_text:
                st.session_state["history_source"] = "📝 貼り付け"

        if history_text:
            try:
                raw_df = parse_history_text(history_text)
                st.session_state["history_df"] = raw_df
                st.success(f"✅ 読み込み成功: {len(raw_df):,}行 / {raw_df['date'].nunique()}日分")
            except Exception as e:
                st.error(f"❌ 読み込み失敗: {e}")
        elif st.session_state.get("history_df") is not None:
            raw_df = st.session_state["history_df"]
            st.info(f"💾 {st.session_state.get('history_source', 'データ使用中')} - {len(raw_df):,}行")

        # 履歴データの保存/管理（ローカルのみ）
        if st.session_state.get("history_df") is not None and IS_LOCAL and SAVE_ENABLED:
            with st.expander("💾 履歴データの保存・バックアップ管理", expanded=False):
                col_save, col_del = st.columns(2)

                with col_save:
                    if st.button("💾 現在のデータを保存", key="save_history", use_container_width=True):
                        latest, backup = save_dataframe_to_local(
                            st.session_state["history_df"],
                            HISTORY_LATEST,
                            "history"
                        )
                        st.success(f"✅ 保存完了")
                        st.caption(f"最新: {latest.name}")
                        st.caption(f"バックアップ: {backup.name}")

                with col_del:
                    if st.button("🗑️ 保存データ削除", key="del_history", use_container_width=True):
                        if delete_local_data(HISTORY_LATEST):
                            st.warning("削除しました")

                backups = get_backup_list("history")
                if backups:
                    st.markdown(f"**📚 バックアップ ({len(backups)}件)**")
                    selected_backup = st.selectbox(
                        "過去のデータに戻す",
                        options=["（選択してください）"] + backups,
                        key="history_backup_select"
                    )
                    if selected_backup != "（選択してください）":
                        if st.button(f"📂 このデータを読み込む", key="load_history_backup"):
                            path = SAVE_DIR / selected_backup
                            restored = load_dataframe_from_local(path)
                            if restored is not None:
                                st.session_state["history_df"] = restored
                                st.session_state["history_source"] = f"💾 {selected_backup}"
                                st.rerun()

    with col_right:
        st.markdown("**🎁 テンプレート**")
        st.download_button(
            "📥 履歴データTSV",
            data=SAMPLE_HISTORY_TSV.encode("utf-8-sig"),
            file_name="history_template.tsv",
            mime="text/tab-separated-values",
        )
        st.download_button(
            "📥 契約者マスタTSV",
            data=SAMPLE_MASTER_TSV.encode("utf-8-sig"),
            file_name="master_template.tsv",
            mime="text/tab-separated-values",
        )

    # 契約者マスタ
    with st.expander("📊 契約者マスタ（任意）", expanded=False):
        master_input_method = st.radio(
            "入力方式",
            ["📁 ファイル", "📝 貼り付け"],
            horizontal=True,
            key="master_input_method"
        )

        master_text = ""
        if master_input_method == "📁 ファイル":
            uploaded_master = st.file_uploader("契約者マスタ", type=["tsv", "csv", "txt"], key="master_upload")
            if uploaded_master is not None:
                master_text = read_uploaded_file(uploaded_master)
        else:
            master_text = st.text_area("マスタを貼り付け", height=120, key="master_paste")

        if master_text:
            try:
                st.session_state["master_df"] = parse_master_text(master_text)
                st.success(f"✅ マスタ取込: {len(st.session_state['master_df'])}件")
            except Exception as e:
                st.error(f"❌ マスタ取込失敗: {e}")

        if st.session_state["master_df"] is not None:
            st.dataframe(st.session_state["master_df"], use_container_width=True, hide_index=True, height=180)

            # マスタ保存（ローカルのみ）
            if IS_LOCAL and SAVE_ENABLED:
                st.markdown("---")
                st.markdown("**💾 マスタの保存・管理**")
                col_ms, col_mdel = st.columns(2)

                with col_ms:
                    if st.button("💾 マスタを保存", key="save_master", use_container_width=True):
                        save_cols = ["yyyymm", "subscriber_base", "new_sales", "cancel_count", "bill_issue_day"]
                        master_to_save = st.session_state["master_df"][save_cols].copy()
                        latest, backup = save_dataframe_to_local(
                            master_to_save,
                            MASTER_LATEST,
                            "master"
                        )
                        st.success(f"✅ マスタ保存完了")

                with col_mdel:
                    if st.button("🗑️ マスタ削除", key="del_master", use_container_width=True):
                        if delete_local_data(MASTER_LATEST):
                            st.warning("マスタを削除しました")

                master_backups = get_backup_list("master")
                if master_backups:
                    st.caption(f"📚 マスタバックアップ: {len(master_backups)}件")
                    selected_master_backup = st.selectbox(
                        "過去のマスタに戻す",
                        options=["（選択してください）"] + master_backups,
                        key="master_backup_select"
                    )
                    if selected_master_backup != "（選択してください）":
                        if st.button(f"📂 このマスタを読み込む", key="load_master_backup"):
                            path = SAVE_DIR / selected_master_backup
                            restored = load_dataframe_from_local(path)
                            if restored is not None:
                                if "subscriber_call_base" not in restored.columns:
                                    restored["subscriber_call_base"] = restored["subscriber_base"] * SUBSCRIBER_CALL_RATE
                                    restored["new_sales_call_base"] = restored["new_sales"] * NEW_SALES_CALL_RATE
                                    restored["log_subscriber_call_base"] = np.log1p(restored["subscriber_call_base"])
                                    restored["log_new_sales_call_base"] = np.log1p(restored["new_sales_call_base"])
                                st.session_state["master_df"] = restored
                                st.rerun()


if st.session_state.get("history_df") is None:
    st.warning("👆 履歴データをアップロードまたは貼り付けてください")
    st.stop()

raw_df = st.session_state["history_df"]


# ============ ステップ2: 予測対象を選ぶ ============
st.markdown("---")
st.markdown("## 🔵 ステップ 2: 予測対象を選ぶ")

daily_check = raw_df.groupby("date", as_index=False)["call_count"].sum()
min_month = daily_check["date"].min().to_period("M").to_timestamp()
max_month = daily_check["date"].max().to_period("M").to_timestamp()
month_opts = month_range(min_month, max_month + pd.offsets.MonthBegin(12))
month_labels = [m.strftime("%Y-%m") for m in month_opts]
suggested = (max_month + pd.offsets.MonthBegin(1)).strftime("%Y-%m")
default_idx = month_labels.index(suggested) if suggested in month_labels else len(month_labels) - 1

col_month, col_info = st.columns([1, 2])
with col_month:
    selected_label = st.selectbox("🎯 予測対象年月", options=month_labels, index=default_idx)
    target_month = pd.to_datetime(selected_label + "-01")

with col_info:
    st.info(f"📅 履歴期間: {min_month.strftime('%Y-%m')} 〜 {max_month.strftime('%Y-%m')} ({(max_month - min_month).days // 30 + 1}ヶ月分)")


# ============ ステップ3: 予測結果 ============
st.markdown("---")
st.markdown("## 🔵 ステップ 3: 予測結果")

train_cat_df = build_features(raw_df, st.session_state.get("master_df"), event_dates)

with st.spinner("🔄 予測実行中..."):
    forecast_cat_df, forecast_total_df, bias_df = build_forecast_by_category(
        train_cat_df, target_month,
        half_life_days, old_cutoff, old_weight, may2025_weight, same_month_weight,
        gw_shape_strength, gw_shape_min, gw_shape_max,
        pre_last2_lift, core_dampen, post13_lift, end_dampen,
        month_start_boost, month_end_boost, event_boost, event_dates,
        use_log_transform=use_log_transform,
        use_category_params=use_category_params,
        use_bias_correction=use_bias_correction,
    )

st.success(f"✅ 予測完了: {target_month.strftime('%Y年%m月')}")

# 月次サマリー
st.markdown("### 📊 予測サマリー")

total_month = int(forecast_total_df["predicted"].sum())
avg_day = forecast_total_df["predicted"].mean()
max_row = forecast_total_df.loc[forecast_total_df["predicted"].idxmax()]
min_row = forecast_total_df.loc[forecast_total_df["predicted"].idxmin()]

tmp = forecast_total_df.copy()
tmp["weekday"] = tmp["date"].dt.weekday
weekday_avg = tmp[tmp["weekday"] < 5]["predicted"].mean()
weekend_avg = tmp[tmp["weekday"] >= 5]["predicted"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("📞 月合計", f"{total_month:,}件")
col2.metric("📅 日平均", f"{avg_day:,.0f}件")
col3.metric("🔺 最大日", f"{int(max_row['predicted']):,}件", help=f"{max_row['date'].strftime('%m/%d')}")
col4.metric("🔻 最小日", f"{int(min_row['predicted']):,}件", help=f"{min_row['date'].strftime('%m/%d')}")

col5, col6 = st.columns(2)
col5.metric("💼 平日平均", f"{weekday_avg:,.0f}件")
col6.metric("🏖️ 土日平均", f"{weekend_avg:,.0f}件")


# 日次グラフ
st.markdown("### 📈 日次予測")
fig = go.Figure()
fig.add_trace(go.Bar(
    x=forecast_total_df["date"],
    y=forecast_total_df["predicted"],
    name="予測",
    marker_color="lightblue"
))
fig.update_layout(
    xaxis_title="日付",
    yaxis_title="コール数",
    height=420,
    showlegend=False
)
st.plotly_chart(fig, use_container_width=True)


# カテゴリ別 & 曜日別
col_cat, col_wd = st.columns(2)

with col_cat:
    st.markdown("**📂 カテゴリ別月合計**")
    cat_summary = forecast_cat_df.groupby("category")["predicted"].sum().reset_index()
    cat_summary["predicted"] = cat_summary["predicted"].round(0).astype(int)
    cat_summary = cat_summary.sort_values("predicted", ascending=False)
    cat_summary.columns = ["カテゴリ", "月合計（件）"]
    cat_summary["月合計（件）"] = cat_summary["月合計（件）"].apply(lambda x: f"{x:,}")
    st.dataframe(cat_summary, use_container_width=True, hide_index=True)

with col_wd:
    st.markdown("**📆 曜日別平均**")
    wd_summary = tmp.groupby("weekday")["predicted"].mean().reset_index()
    wd_summary["曜日"] = wd_summary["weekday"].map(WEEKDAY_MAP)
    wd_summary["平均件数"] = wd_summary["predicted"].round(0).astype(int).apply(lambda x: f"{x:,}")
    wd_summary = wd_summary[["曜日", "平均件数"]]
    st.dataframe(wd_summary, use_container_width=True, hide_index=True)


# CSVダウンロード
st.markdown("### 💾 予測結果ダウンロード")

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    csv_daily = forecast_total_df[["date", "predicted"]].copy()
    csv_daily["predicted"] = csv_daily["predicted"].round(0).astype(int)
    csv_daily.columns = ["日付", "予測件数"]
    csv_bytes = csv_daily.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 日次予測（合計）CSV",
        data=csv_bytes,
        file_name=f"forecast_daily_{target_month.strftime('%Y%m')}.csv",
        mime="text/csv",
    )

with col_dl2:
    csv_cat = forecast_cat_df[["date", "category", "predicted"]].copy()
    csv_cat["predicted"] = csv_cat["predicted"].round(0).astype(int)
    csv_cat.columns = ["日付", "カテゴリ", "予測件数"]
    csv_cat_bytes = csv_cat.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 カテゴリ別予測CSV",
        data=csv_cat_bytes,
        file_name=f"forecast_category_{target_month.strftime('%Y%m')}.csv",
        mime="text/csv",
    )


# ============ 実績との比較 ============
target_period = target_month.to_period("M")
history_actual = raw_df[raw_df["date"].dt.to_period("M") == target_period].copy()

if not history_actual.empty:
    st.markdown("---")
    st.markdown("## 📊 実績との比較")

    actual_daily = history_actual.groupby("date", as_index=False)["call_count"].sum().rename(columns={"call_count": "actual"})
    compare_df = forecast_total_df.merge(actual_daily, on="date", how="left")

    actual_valid = compare_df.dropna(subset=["actual"])
    if not actual_valid.empty:
        actual_total = int(actual_valid["actual"].sum())
        actual_days = len(actual_valid)
        actual_avg = actual_valid["actual"].mean()
        pred_partial = int(actual_valid["predicted"].sum())
        diff = actual_total - pred_partial

        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("実績合計", f"{actual_total:,}件", help=f"{actual_days}日分")
        ac2.metric("実績日平均", f"{actual_avg:,.0f}件")
        ac3.metric("予測合計", f"{pred_partial:,}件")
        ac4.metric("差分", f"{diff:+,}件")

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(x=compare_df["date"], y=compare_df["predicted"], name="予測", marker_color="lightblue"))
    fig_cmp.add_trace(go.Bar(x=compare_df["date"], y=compare_df["actual"], name="実績", marker_color="darkblue"))
    fig_cmp.update_layout(barmode="group", xaxis_title="日付", yaxis_title="コール数", height=460)
    st.plotly_chart(fig_cmp, use_container_width=True)

    metrics, running = compute_metrics(compare_df)
    if metrics:
        with st.expander("📉 精度指標（MAPE / RMSE / Bias）", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("MAPE", f"{metrics['MAPE']:.2f}%", help="平均誤差率")
            c2.metric("RMSE", f"{metrics['RMSE']:.2f}", help="平均的なズレ量（件数）")
            c3.metric("Bias", f"{metrics['Bias']:.2f}", help="マイナス=過大予測、プラス=過小予測")

            if running is not None and not running.empty:
                fig_run = go.Figure()
                fig_run.add_trace(go.Scatter(x=running["date"], y=running["cumulative_mape"], mode="lines+markers", name="累積MAPE(%)", yaxis="y1"))
                fig_run.add_trace(go.Scatter(x=running["date"], y=running["cumulative_rmse"], mode="lines+markers", name="累積RMSE", yaxis="y1"))
                fig_run.add_trace(go.Scatter(x=running["date"], y=running["cumulative_bias"], mode="lines+markers", name="累積Bias", yaxis="y2"))
                fig_run.update_layout(
                    height=380,
                    xaxis_title="日付",
                    yaxis=dict(title="MAPE / RMSE"),
                    yaxis2=dict(title="Bias", overlaying="y", side="right"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_run, use_container_width=True)


st.markdown("---")
if IS_LOCAL:
    st.caption(f"© ドコモでんきコール予測アプリ Ver1 (Local) | 保存先: {SAVE_DIR}")
else:
    st.caption("© ドコモでんきコール予測アプリ Ver1 (Cloud) | Streamlit Community Cloud")