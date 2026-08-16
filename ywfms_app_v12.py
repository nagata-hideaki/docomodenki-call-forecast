# -*- coding: utf-8 -*-
"""
YOKOHAMA WORK FORCE MANAGEMENT SYSTEM (YWFMS) v11
============================================================
v11の変更点(v10からの改修):
  ・STEP0に「日別予測CSVアップロード」を追加
     - DLP/秘密度ラベルでローカル保存が暗号化(.pfile)される問題を回避
     - file_uploaderはメモリ上で直接読むため暗号化の影響を受けない
     - app3.pyの日次予測CSV(列: 日付/date, 予測件数/predicted 等)をそのまま取込
     - アップロードした予測は FORECAST_LOG 相当として STEP3(MAPE)で使用
  ・ローカル latest_forecast.csv も従来どおりフォールバックで読む(暗号化時は自動スキップ)

継承(v10)：余剰/不足の反転表示・意味づけ、状態列
継承(v9)：余力ランキング/シンプル示唆(研修計画表は別管理)
継承(v8)：UI(縦型ステッパー/ヒーローバナー/ドット式プログレス/KPIカード/行動提案カード)
継承(v7)：ローカル完結(CALL_HISTORY・内部ベースライン予測)、NaN修正
営業時間：09:00-20:00 / 配分方式：X案 / SL目標：80%20秒 / 時間帯配分キュー：でんき
============================================================
"""
import math
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ==========================================================
# 定数
# ==========================================================
APP_NAME = "YOKOHAMA WORK FORCE MANAGEMENT SYSTEM"
APP_SHORT = "YWFMS"
TARGET_QUEUE = "でんき"
DEFAULT_HIST_PATH = r"C:\Users\1991000\wfm_data\interval_history_all.tsv"
DEFAULT_MASTER_PATH = r"C:\Users\1991000\docomo_call_forecast\latest_master.tsv"
DEFAULT_CALLHIST_PATH = r"C:\Users\1991000\docomo_call_forecast\latest_history.tsv"
DEFAULT_FORECAST_CSV_PATH = r"C:\Users\1991000\docomo_call_forecast\latest_forecast.csv"
WEEKDAY_JP = ["月", "火", "水", "木", "金", "土", "日"]

BIZ_START = "09:00"
BIZ_END = "20:00"

TBL_CALL_HISTORY = "CALL_HISTORY"
TBL_SUBSCRIBER = "SUBSCRIBER_MASTER"
TBL_INTRADAY_LOG = "INTRADAY_LOG"
TBL_FORECAST_LOG = "FORECAST_LOG"


def detect_environment():
    local_path = Path(r"C:\Users\1991000")
    if local_path.exists() and os.name == "nt":
        return "local"
    return "cloud"


ENV = detect_environment()
IS_LOCAL = ENV == "local"
IS_CLOUD = ENV == "cloud"

BLOCKS = {
    "午前(9:00-12:00)": ("09:00", "12:00"),
    "午後(12:00-17:00)": ("12:00", "17:00"),
    "夕方(17:00-20:00)": ("17:00", "20:00"),
}

STEPS = [
    "STEP0 データ読込",
    "STEP1 予測対象月",
    "STEP2 日別・時間帯別予測",
    "STEP3 精度確認(任意)",
    "STEP4 必要人員数算定",
    "STEP5 予実管理",
    "STEP6 再予測",
    "STEP7 稼働調整示唆",
]

# ==========================================================
# ページ設定
# ==========================================================
st.set_page_config(page_title=APP_SHORT, layout="wide", page_icon="📞")


def inject_css():
    st.markdown(
        """
        <style>
        :root {
            --ywfms-navy:#0f2a52; --ywfms-blue:#1f6feb; --ywfms-accent:#00b3a4;
            --ywfms-bg:#f5f7fb; --ywfms-green:#12b886; --ywfms-amber:#f59f00;
            --ywfms-red:#e03131; --ywfms-gray:#adb5bd;
        }
        .stApp { background: var(--ywfms-bg); }
        .ywfms-hero {
            background: linear-gradient(120deg,#0f2a52 0%,#1f6feb 60%,#00b3a4 130%);
            border-radius:18px; padding:22px 28px; color:#fff; margin-bottom:8px;
            box-shadow:0 8px 24px rgba(15,42,82,.18);
        }
        .ywfms-hero h1 { font-size:26px; font-weight:800; margin:0; letter-spacing:.5px; color:#fff; }
        .ywfms-hero p { margin:6px 0 0 0; font-size:13.5px; opacity:.92; }
        .ywfms-hero .chips { margin-top:12px; }
        .ywfms-hero .chip {
            display:inline-block; background:rgba(255,255,255,.16);
            border:1px solid rgba(255,255,255,.28); padding:4px 12px;
            border-radius:999px; font-size:12px; margin-right:8px; backdrop-filter:blur(4px);
        }
        .ywfms-dots { display:flex; align-items:center; gap:6px; margin:14px 0 4px 0; }
        .ywfms-dot { height:8px; border-radius:999px; flex:1; background:#dfe4ee; transition:all .3s; }
        .ywfms-dot.done { background:var(--ywfms-green); }
        .ywfms-dot.current { background:var(--ywfms-blue); box-shadow:0 0 0 3px rgba(31,111,235,.18); }
        .ywfms-step-label { font-size:13px; color:#5b6577; font-weight:600; margin-bottom:10px; }
        section[data-testid="stSidebar"] { background:#fff; }
        .ywfms-brand { display:flex; align-items:center; gap:10px; margin-bottom:2px; }
        .ywfms-brand .logo {
            width:38px;height:38px;border-radius:10px;
            background:linear-gradient(135deg,#0f2a52,#1f6feb);
            display:flex;align-items:center;justify-content:center;color:#fff;font-size:20px;
        }
        .ywfms-brand .bt { font-size:20px; font-weight:800; color:var(--ywfms-navy); line-height:1; }
        .ywfms-brand .bs { font-size:10.5px; color:#8a94a6; }
        .ywfms-step {
            display:flex; align-items:center; gap:11px; padding:10px 12px;
            border-radius:12px; margin:6px 0; border:1px solid transparent; transition:all .2s;
        }
        .ywfms-step .badge {
            width:26px;height:26px;border-radius:50%;
            display:flex;align-items:center;justify-content:center;
            font-size:12.5px;font-weight:700;flex-shrink:0;
        }
        .ywfms-step .txt { font-size:13px; line-height:1.25; }
        .ywfms-step.done { background:#f0fbf6; }
        .ywfms-step.done .badge { background:var(--ywfms-green); color:#fff; }
        .ywfms-step.done .txt { color:#3f7a63; }
        .ywfms-step.current { background:#eef4ff; border-color:#c7dbff; }
        .ywfms-step.current .badge { background:var(--ywfms-blue); color:#fff; box-shadow:0 0 0 4px rgba(31,111,235,.15); }
        .ywfms-step.current .txt { color:var(--ywfms-navy); font-weight:800; }
        .ywfms-step.todo .badge { background:#eef1f6; color:#9aa4b5; border:1px solid #e2e7f0; }
        .ywfms-step.todo .txt { color:#9aa4b5; }
        div[data-testid="stMetric"] {
            background:#fff; border:1px solid #e6eaf2; border-radius:14px;
            padding:14px 16px; box-shadow:0 2px 8px rgba(15,42,82,.05);
        }
        div[data-testid="stMetricLabel"] p { font-size:12.5px !important; color:#6b7688 !important; }
        div[data-testid="stMetricValue"] { font-size:26px !important; color:var(--ywfms-navy) !important; font-weight:800; }
        h2, h3 { color:var(--ywfms-navy); }
        .stButton>button { border-radius:10px; font-weight:700; border:1px solid #d7deea; }
        .stButton>button[kind="primary"] { background:linear-gradient(135deg,#1f6feb,#00b3a4); border:none; }
        .ywfms-advice {
            border-radius:16px; padding:20px 24px; margin:6px 0 16px 0;
            font-size:15px; line-height:1.9; box-shadow:0 6px 18px rgba(15,42,82,.08);
        }
        .ywfms-advice.ok { background:#e9fbf3; border:1px solid #a8e6cf; color:#1b7a55; }
        .ywfms-advice.warn { background:#fff8e8; border:1px solid #ffe0a3; color:#8a5a00; }
        .ywfms-advice.crit { background:#fdecec; border:1px solid #ffc2c2; color:#a61b1b; }
        .ywfms-advice .headline { font-size:17px; font-weight:800; margin-bottom:8px; display:block; }
        .ywfms-note {
            background:#eef4ff; border:1px solid #c7dbff; border-radius:10px;
            padding:8px 14px; font-size:12.5px; color:#33507f; margin:4px 0 10px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# セッション
# ==========================================================
_DEFAULTS = {
    "wizard_idx": 0, "data_loaded": False, "call_history_df": None,
    "subscriber_df": None, "interval_hist_df": None, "forecast_log_df": None,
    "target_ym": None, "selected_categories": None, "forecast_source": None,
    "daily_forecast": None, "interval_ratio": None, "interval_forecast": None,
    "required_df": None, "intraday_df": None, "uploaded_forecast_df": None,
    "aht_sec": 1628, "sl_target": 0.80, "target_sec": 20, "shrink": 0.30,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==========================================================
# Snowflake接続
# ==========================================================
@st.cache_resource(show_spinner=False)
def get_snowpark_session():
    try:
        from snowflake.snowpark import Session
    except ImportError:
        raise RuntimeError("snowflake-snowpark-python が未導入です。")
    cfg = {}
    if "snowflake" in st.secrets:
        cfg.update(dict(st.secrets["snowflake"]))
    for k in ("account", "user", "warehouse", "database", "schema", "role"):
        env_v = os.getenv(f"SNOWFLAKE_{k.upper()}")
        if env_v:
            cfg[k] = env_v
    cfg.setdefault("authenticator", "externalbrowser")
    if not cfg.get("account") or not cfg.get("user"):
        raise RuntimeError("Snowflake接続情報が未設定です。")
    return Session.builder.configs(cfg).create()


def snowflake_available() -> bool:
    try:
        import snowflake.connector  # noqa
        return True
    except Exception:
        return False


_SF_CONN = None


def get_snowflake_conn():
    global _SF_CONN
    if _SF_CONN is not None:
        try:
            _SF_CONN.cursor().execute("SELECT 1")
            return _SF_CONN
        except Exception:
            _SF_CONN = None
    import snowflake.connector
    if IS_LOCAL:
        _SF_CONN = snowflake.connector.connect(
            user="nagatah@nttdocomo.com",
            account="nttdocomo_is_dl_pro.ap-northeast-1.aws",
            authenticator="externalbrowser",
            warehouse="USER_COSN_SWH_P01_LAK_01",
            database="USPIDB_N_P01_LAK",
            schema="USER_COSN_01",
        )
    else:
        _SF_CONN = snowflake.connector.connect(
            user=st.secrets["SNOWFLAKE_USER"],
            password=st.secrets["SNOWFLAKE_PASSWORD"],
            account=st.secrets["SNOWFLAKE_ACCOUNT"],
            warehouse=st.secrets["SNOWFLAKE_WAREHOUSE"],
            database="USPIDB_N_P01_LAK",
            schema="USER_COSN_01",
            login_timeout=120,
            network_timeout=120,
        )
    return _SF_CONN


def sf_run_query(sql: str) -> pd.DataFrame:
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)
    finally:
        cur.close()


def sf_write_df(df: pd.DataFrame, table: str, mode: str = "append") -> int:
    session = get_snowpark_session()
    session.create_dataframe(df).write.mode(mode).save_as_table(table)
    return len(df)


# ==========================================================
# 共通ロジック
# ==========================================================
def _in_biz_hours(ts: str) -> bool:
    return BIZ_START <= ts < BIZ_END


def _parse_flex_date(s):
    if pd.isna(s):
        return pd.NaT
    txt = str(s).strip()
    if not txt:
        return pd.NaT
    digits = txt.replace("/", "").replace("-", "")
    if digits.isdigit() and len(digits) == 6:
        try:
            return pd.to_datetime(digits + "01", format="%Y%m%d", errors="raise")
        except Exception:
            pass
    if ("年" in txt) and ("月" in txt):
        try:
            y = txt.split("年")[0]
            m = txt.split("年")[1].split("月")[0]
            return pd.to_datetime(f"{int(y):04d}-{int(m):02d}-01", errors="raise")
        except Exception:
            pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y%m%d", "%Y-%m", "%Y/%m"):
        try:
            return pd.to_datetime(txt, format=fmt, errors="raise")
        except Exception:
            continue
    return pd.to_datetime(txt, errors="coerce")


def load_local_tsv(path_str: str):
    p = Path(path_str)
    if path_str and p.exists():
        try:
            df = pd.read_csv(p, sep="\t", encoding="utf-8-sig")
            # DLP暗号化(.pfile)を検知したらNone扱い
            if df.shape[1] == 1 and str(df.columns[0]).startswith(".pfile"):
                return None
            return df
        except Exception:
            return None
    return None


def parse_forecast_csv(file_or_path):
    """
    日別予測CSV/TSVを読み込み、date/queue/predicted_calls/created_at へ正規化。
    - .tsv→タブ, それ以外→カンマを自動判定（失敗時は区切り自動推定）
    - 列名の揺らぎ(日付/date/ds、予測件数/predicted/予測/yhat等)を吸収
    - 暗号化(.pfile)ファイルは None を返す
    """
    # --- 読み込み（TSV/CSV自動判定） ---
    _name = getattr(file_or_path, "name", None) or str(file_or_path)
    _sep = "\t" if _name.lower().endswith(".tsv") else ","
    try:
        if hasattr(file_or_path, "read"):  # uploaded file
            try:
                raw = pd.read_csv(file_or_path, sep=_sep, encoding="utf-8-sig")
            except Exception:
                file_or_path.seek(0)
                raw = pd.read_csv(file_or_path, sep=None, engine="python",
                                  encoding="utf-8-sig")
        else:
            p = Path(file_or_path)
            if not p.exists():
                return None
            try:
                raw = pd.read_csv(p, sep=_sep, encoding="utf-8-sig")
            except Exception:
                raw = pd.read_csv(p, sep=None, engine="python",
                                  encoding="utf-8-sig")
    except Exception:
        return None

    # --- .pfile（暗号化）検出 ---
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if len(raw.columns) == 1 and raw.columns[0].startswith(".pfile"):
        return None

    # --- 列名判定（揺らぎ吸収） ---
    dcol = next((c for c in raw.columns
                 if c in ("date", "日付", "ds", "対象日", "予測日")), None)
    pcol = next((c for c in raw.columns
                 if ("predict" in c) or ("予測" in c) or ("forecast" in c)
                 or c in ("yhat", "call_count", "予測件数")), None)
    if dcol is None or pcol is None:
        return None

    # --- 正規化（出口は従来どおり4列） ---
    out = pd.DataFrame()
    out["date"] = raw[dcol].apply(_parse_flex_date).dt.date
    out["predicted_calls"] = pd.to_numeric(raw[pcol], errors="coerce")
    out = out.dropna(subset=["date", "predicted_calls"]).sort_values("date").reset_index(drop=True)
    if out.empty:
        return None
    out["queue"] = TARGET_QUEUE
    out["created_at"] = pd.NaT
    return out[["date", "queue", "predicted_calls", "created_at"]]

def compute_interval_ratio(df_hist: pd.DataFrame) -> pd.DataFrame:
    df = df_hist[df_hist["queue"] == TARGET_QUEUE].copy()
    if df.empty:
        return pd.DataFrame(columns=["weekday", "time_slot", "call_count", "ratio"])
    df = df[df["time_slot"].astype(str).apply(_in_biz_hours)]
    if df.empty:
        return pd.DataFrame(columns=["weekday", "time_slot", "call_count", "ratio"])
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.weekday
    agg = df.groupby(["weekday", "time_slot"], as_index=False)["call_count"].sum()
    agg["ratio"] = agg.groupby("weekday")["call_count"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0.0
    )
    return agg


def normalize_call_history(cdf: pd.DataFrame) -> pd.DataFrame:
    if cdf is None or cdf.empty:
        return pd.DataFrame(columns=["date", "category", "call_count"])
    df = cdf.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename = {}
    for c in df.columns:
        if c in ("date", "日付", "c1"):
            rename[c] = "date"
        elif c in ("category", "カテゴリ", "queue", "キュー", "c2"):
            rename[c] = "category"
        elif c in ("call_count", "コール数", "着信数", "c3"):
            rename[c] = "call_count"
        elif c in ("answered_count", "応答数", "answered", "c4"):
            rename[c] = "answered_count"
    df = df.rename(columns=rename)
    if "date" not in df.columns or "call_count" not in df.columns:
        return pd.DataFrame(columns=["date", "category", "call_count"])
    df["date"] = df["date"].apply(_parse_flex_date)
    df["call_count"] = pd.to_numeric(df["call_count"], errors="coerce").fillna(0)
    if "category" not in df.columns:
        df["category"] = "ALL"
    df = df.dropna(subset=["date"])
    return df[["date", "category", "call_count"]]


def baseline_daily_forecast(cdf, ym, categories=None, lookback_weeks=8):
    hist = normalize_call_history(cdf)
    if hist.empty:
        return pd.DataFrame(columns=["date", "predicted_calls"])
    if categories:
        hist = hist[hist["category"].isin(categories)]
    daily_hist = hist.groupby("date", as_index=False)["call_count"].sum()
    daily_hist["date"] = pd.to_datetime(daily_hist["date"])
    daily_hist["weekday"] = daily_hist["date"].dt.weekday
    y, m = map(int, ym.split("-"))
    month_start = pd.Timestamp(y, m, 1)
    n_days = (month_start + pd.offsets.MonthEnd(0)).day
    target_dates = [pd.Timestamp(y, m, d) for d in range(1, n_days + 1)]
    overall_mean = daily_hist["call_count"].mean() if not daily_hist.empty else 0.0
    rows = []
    for td in target_dates:
        wd = td.weekday()
        past = daily_hist[(daily_hist["weekday"] == wd) & (daily_hist["date"] < td)]
        past = past.sort_values("date").tail(lookback_weeks)
        if not past.empty:
            pred = float(past["call_count"].mean())
        else:
            any_wd = daily_hist[daily_hist["weekday"] == wd]
            pred = float(any_wd["call_count"].mean()) if not any_wd.empty else float(overall_mean)
        rows.append({"date": td.date(), "predicted_calls": round(pred, 1)})
    return pd.DataFrame(rows)


# ----- アーランC -----
def _erlang_b(agents, traffic):
    b = 1.0
    for k in range(1, agents + 1):
        b = (traffic * b) / (k + traffic * b)
    return b


def erlang_c(agents, traffic):
    if agents <= 0:
        return 1.0
    if traffic <= 0:
        return 0.0
    if agents <= traffic:
        return 1.0
    b = _erlang_b(agents, traffic)
    denom = agents - traffic * (1 - b)
    if denom <= 0:
        return 1.0
    return (agents * b) / denom


def service_level(agents, calls, aht, target_sec=20, interval_sec=1800):
    if calls <= 0 or aht <= 0:
        return 1.0
    lam = calls / interval_sec
    traffic = lam * aht
    if agents <= traffic:
        return 0.0
    pw = erlang_c(agents, traffic)
    sl = 1 - pw * math.exp(-(agents - traffic) * target_sec / aht)
    return max(0.0, min(1.0, sl))


def required_agents(calls, aht, sl_target=0.80, target_sec=20, interval_sec=1800, max_agents=500):
    if calls <= 0:
        return 0
    lam = calls / interval_sec
    traffic = lam * aht
    start = max(1, int(math.ceil(traffic)))
    for n in range(start, max_agents + 1):
        if service_level(n, calls, aht, target_sec, interval_sec) >= sl_target:
            return n
    return max_agents


def in_block(ts, bkey):
    s, e = BLOCKS[bkey]
    return s <= ts < e


# ----- 表示ヘルパー(余剰=プラス反転) -----
def surplus_label(shortage_val: int) -> str:
    v = int(shortage_val)
    if v >= 1:
        return f"🔴 ▲{v}名（他時間帯へ調整）"
    if v <= -1:
        return f"🟢 ＋{abs(v)}名（研修・休憩等に充当可）"
    return "⚪ ±0（適正）"


def surplus_number(shortage_val: int) -> int:
    return -int(shortage_val)


# ----- 精度指標 -----
def calc_mape(a, p):
    a = np.asarray(a, float); p = np.asarray(p, float)
    m = (~np.isnan(a)) & (~np.isnan(p)) & (a != 0)
    return float(np.mean(np.abs((a[m] - p[m]) / a[m])) * 100) if m.sum() else np.nan


def calc_rmse(a, p):
    a = np.asarray(a, float); p = np.asarray(p, float)
    m = (~np.isnan(a)) & (~np.isnan(p))
    return float(np.sqrt(np.mean((a[m] - p[m]) ** 2))) if m.sum() else np.nan


def calc_bias(a, p):
    a = np.asarray(a, float); p = np.asarray(p, float)
    m = (~np.isnan(a)) & (~np.isnan(p))
    return float(np.mean(p[m] - a[m])) if m.sum() else np.nan


def summarize_metrics(df, ac, pc, gc=None):
    if gc is None:
        return pd.DataFrame([{
            "MAPE(%)": round(calc_mape(df[ac], df[pc]), 2),
            "RMSE": round(calc_rmse(df[ac], df[pc]), 2),
            "Bias": round(calc_bias(df[ac], df[pc]), 2),
            "件数": int(df[[ac, pc]].dropna().shape[0]),
        }])
    rows = []
    for g, sub in df.groupby(gc):
        rows.append({
            gc: g,
            "MAPE(%)": round(calc_mape(sub[ac], sub[pc]), 2),
            "RMSE": round(calc_rmse(sub[ac], sub[pc]), 2),
            "Bias": round(calc_bias(sub[ac], sub[pc]), 2),
            "件数": int(sub[[ac, pc]].dropna().shape[0]),
        })
    return pd.DataFrame(rows)


def _fmt_slot_ranges(slots):
    if not slots:
        return []
    slots = sorted(slots)
    ranges = []
    start = prev = slots[0]

    def _add30(t):
        h, m = map(int, t.split(":"))
        dt = datetime(2000, 1, 1, h, m) + timedelta(minutes=30)
        return dt.strftime("%H:%M")

    for s in slots[1:]:
        if s == _add30(prev):
            prev = s
        else:
            ranges.append((start, _add30(prev)))
            start = prev = s
    ranges.append((start, _add30(prev)))
    return [f"{a}〜{b}" for a, b in ranges]


def _bold_all(s):
    out, toggle = [], True
    for part in s.split("**"):
        out.append(part)
        out.append("<b>" if toggle else "</b>")
        toggle = not toggle
    return "".join(out[:-1])


# ==========================================================
# サイドバー
# ==========================================================
def render_sidebar():
    idx = st.session_state.wizard_idx
    st.sidebar.markdown(
        f"""
        <div class="ywfms-brand">
            <div class="logo">🏢</div>
            <div>
                <div class="bt">{APP_SHORT}</div>
                <div class="bs">Yokohama Work Force Mgmt · v11</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div style="font-size:12px;color:#8a94a6;margin:10px 0 4px 2px;font-weight:700;'
        'letter-spacing:.4px;">WFM 標準フロー</div>',
        unsafe_allow_html=True,
    )
    short_labels = [
        "データ読込", "予測対象月", "日別・時間帯別予測", "精度確認(任意)",
        "必要人員数算定", "予実管理", "再予測", "稼働調整示唆",
    ]
    html = []
    for i, name in enumerate(short_labels):
        if i < idx:
            state, badge = "done", "✓"
        elif i == idx:
            state, badge = "current", str(i)
        else:
            state, badge = "todo", str(i)
        html.append(
            f'<div class="ywfms-step {state}">'
            f'<div class="badge">{badge}</div>'
            f'<div class="txt">STEP{i}<br>{name}</div></div>'
        )
    st.sidebar.markdown("".join(html), unsafe_allow_html=True)
    st.sidebar.markdown("---")
    fsrc = st.session_state.forecast_source or "(STEP0で判定)"
    st.sidebar.markdown(
        f"""
        <div style="background:#f5f7fb;border:1px solid #e6eaf2;border-radius:12px;
                    padding:12px 14px;font-size:12.5px;color:#4a5568;line-height:1.8;">
            <b style="color:#0f2a52;">運用スコープ</b><br>
            🎧 キュー：{TARGET_QUEUE}<br>
            🕘 営業時間：{BIZ_START}–{BIZ_END}<br>
            🎯 SL目標：80% / 20秒<br>
            🔮 予測：{fsrc}
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar.expander("❄ Snowflake 接続"):
        if snowflake_available():
            st.success("接続OK(app3.py同一設定)")
        else:
            st.warning("未接続(ローカルファイルで稼働)")


def nav_buttons(prev_ok=True, next_ok=True, next_label="次へ ▶"):
    c1, c2, c3 = st.columns([1, 6, 1])
    with c1:
        if st.session_state.wizard_idx > 0:
            if st.button("◀ 戻る", use_container_width=True):
                st.session_state.wizard_idx -= 1
                st.rerun()
    with c3:
        if next_ok and st.session_state.wizard_idx < len(STEPS) - 1:
            if st.button(next_label, use_container_width=True, type="primary"):
                st.session_state.wizard_idx += 1
                st.rerun()


# ==========================================================
# STEP0
# ==========================================================
def step0_load():
    st.header("STEP0 データ読み込み")
    st.caption("過去コール実績・契約者データ・30分履歴を読み込みます。")

    auto = st.session_state.data_loaded
    _, colB = st.columns([2, 1])
    with colB:
        do_reload = st.button("🔄 再読込", use_container_width=True)

    if (not auto) or do_reload:
        with st.status("過去データ・契約者データ 読み込み中...", expanded=True) as status:
            sf_ok = snowflake_available()

            st.write("▶ コール実績(CALL_HISTORY)...")
            call_df = None
            if sf_ok:
                try:
                    call_df = sf_run_query(
                        f"SELECT c1 AS date, c2 AS category, c3 AS call_count, c4 AS answered_count "
                        f"FROM {TBL_CALL_HISTORY}")
                    call_df.columns = [c.lower() for c in call_df.columns]
                    st.write(f"　└ Snowflakeから {len(call_df):,} 行")
                except Exception as e:
                    st.write(f"　└ Snowflake失敗({e})。ローカルへフォールバック")
            if call_df is None:
                raw = load_local_tsv(DEFAULT_CALLHIST_PATH)
                if raw is not None:
                    call_df = raw
                    st.write(f"　└ ローカル latest_history.tsv {len(call_df):,} 行")
                else:
                    st.write("　└ ⚠ latest_history.tsv 未検出")
            st.session_state.call_history_df = call_df

            st.write("▶ 契約者マスタ(SUBSCRIBER_MASTER)...")
            sub_df = None
            if sf_ok:
                try:
                    sub_df = sf_run_query(f"SELECT * FROM {TBL_SUBSCRIBER}")
                    sub_df.columns = [c.lower() for c in sub_df.columns]
                    st.write(f"　└ Snowflakeから {len(sub_df):,} 行")
                except Exception:
                    pass
            if sub_df is None:
                raw = load_local_tsv(DEFAULT_MASTER_PATH)
                if raw is not None:
                    raw.columns = [c.lower() for c in raw.columns]
                    sub_df = raw
                    st.write(f"　└ ローカルmaster {len(sub_df):,} 行")
            st.session_state.subscriber_df = sub_df

            st.write("▶ 30分履歴(interval_history)...")
            ih = load_local_tsv(DEFAULT_HIST_PATH)
            if ih is not None:
                st.write(f"　└ ローカルTSV {len(ih):,} 行 / 期間 {ih['date'].min()}〜{ih['date'].max()}")
            else:
                st.write("　└ ⚠ interval_history_all.tsv 未検出(STEP2で指定可)")
            st.session_state.interval_hist_df = ih

            st.write("▶ 日別予測(FORECAST_LOG / app3.py本番モデル)...")
            fdf = None
            src = None
            if sf_ok:
                try:
                    fdf = sf_run_query(
                        f"SELECT TARGET_DATE AS date, QUEUE AS queue, PREDICTED_CALLS AS predicted_calls, "
                        f"CREATED_AT AS created_at, GRANULARITY AS granularity "
                        f"FROM {TBL_FORECAST_LOG} WHERE UPPER(GRANULARITY)='DAILY' "
                        f"QUALIFY ROW_NUMBER() OVER (PARTITION BY TARGET_DATE, QUEUE "
                        f"ORDER BY FORECAST_RUN_TIME DESC, CREATED_AT DESC) = 1")
                    fdf.columns = [c.lower() for c in fdf.columns]
                    fdf["date"] = pd.to_datetime(fdf["date"]).dt.date
                    src = "Snowflake FORECAST_LOG"
                    st.write(f"　└ Snowflakeから {len(fdf):,} 行")
                except Exception as e:
                    st.write(f"　└ Snowflake FORECAST_LOG取得失敗({e})")
            if fdf is None:
                # アップロード済み予測を最優先で使う
                if st.session_state.uploaded_forecast_df is not None:
                    fdf = st.session_state.uploaded_forecast_df
                    src = "アップロード予測CSV"
                    st.write(f"　└ アップロード予測CSV {len(fdf):,} 行")
            if fdf is None:
                parsed = parse_forecast_csv(DEFAULT_FORECAST_CSV_PATH)
                if parsed is not None:
                    fdf = parsed
                    src = "ローカル予測CSV"
                    st.write(f"　└ ローカル予測CSV {len(fdf):,} 行")
                else:
                    p = Path(DEFAULT_FORECAST_CSV_PATH)
                    if p.exists():
                        st.write("　└ ⚠ ローカル予測CSVは暗号化(.pfile)等で読めません")
            if fdf is None:
                src = "内部ベースライン(STEP2で生成)"
                st.write("　└ ⚠ 正式予測なし → STEP2で内部ベースライン予測を自動生成します")
            st.session_state.forecast_log_df = fdf
            st.session_state.forecast_source = src

            st.session_state.data_loaded = True
            status.update(label="読み込み完了 ✅", state="complete", expanded=False)

    # ---- 日別予測CSVアップロード(v11新機能・DLP回避) ----
    with st.expander("📤 日別予測CSVをアップロード（推奨・DLP回避）",
                     expanded=(st.session_state.forecast_log_df is None)):
        st.caption(
            "app3.pyの「日次予測（合計）CSV」をここに直接アップロードしてください。"
            "ファイルはメモリ上で読むため、秘密度ラベル(.pfile)の影響を受けません。"
            "列は『日付/date』『予測件数/predicted』を自動判別します。"
        )
        up = st.file_uploader("app3.py 日次予測CSV", type=["csv", "tsv"], key="forecast_upload")
        if up is not None:
            parsed = parse_forecast_csv(up)
            if parsed is None:
                st.error("読み込めませんでした。列に『日付』『予測件数』相当があるCSVかご確認ください。")
            else:
                st.session_state.uploaded_forecast_df = parsed
                st.session_state.forecast_log_df = parsed
                st.session_state.forecast_source = "アップロード予測CSV"
                months = sorted(pd.to_datetime(parsed["date"]).dt.strftime("%Y-%m").unique())
                st.success(f"取込成功：{len(parsed)}日分（対象月：{'、'.join(months)}）")
                st.dataframe(
                    parsed[["date", "predicted_calls"]].rename(
                        columns={"date": "日付", "predicted_calls": "予測件数"}),
                    use_container_width=True, height=240, hide_index=True)

    st.subheader("読込サマリ")

    def _rows(df):
        return f"{len(df):,} 行" if df is not None else "未取得"

    fsrc = st.session_state.forecast_source or "-"
    summary = pd.DataFrame([
        {"データ": "コール実績 CALL_HISTORY", "状態": _rows(st.session_state.call_history_df)},
        {"データ": "契約者マスタ SUBSCRIBER_MASTER", "状態": _rows(st.session_state.subscriber_df)},
        {"データ": "30分履歴 interval_history", "状態": _rows(st.session_state.interval_hist_df)},
        {"データ": "日別予測", "状態": f"{_rows(st.session_state.forecast_log_df)}（{fsrc}）"},
    ])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    if st.session_state.forecast_log_df is None:
        st.info(
            "正式な日別予測がありません。上の「📤 日別予測CSVをアップロード」から取込むか、"
            "STEP2で内部ベースライン予測を自動生成します。"
        )
    else:
        st.success(f"日別予測ソース：{fsrc}")

    ready = (st.session_state.interval_hist_df is not None) and \
            (st.session_state.call_history_df is not None or st.session_state.forecast_log_df is not None)
    if not ready:
        st.warning("30分履歴、またはコール実績/予測のいずれかが未取得です。")
    nav_buttons(next_ok=True)


# ==========================================================
# STEP1
# ==========================================================
def step1_month():
    st.header("STEP1 予測対象月の選択")
    fdf = st.session_state.forecast_log_df
    months = []
    if fdf is not None and not fdf.empty:
        ym = pd.to_datetime(fdf["date"]).dt.strftime("%Y-%m")
        months = sorted(ym.unique())
    if months:
        st.success(f"日別予測({st.session_state.forecast_source}) に {len(months)} ヶ月分あります。")
        sel = st.selectbox("予測対象月", months, index=len(months) - 1)
    else:
        st.warning("正式な日別予測がないため、内部ベースライン予測で対象月を作成します。対象月を指定してください。")
        today = date.today()
        c1, c2 = st.columns(2)
        with c1:
            y = st.number_input("年", 2024, 2030, today.year)
        with c2:
            m = st.number_input("月", 1, 12, today.month)
        sel = f"{int(y):04d}-{int(m):02d}"
    st.session_state.target_ym = sel
    st.info(f"選択中の予測対象月：**{sel}**")

    cdf = st.session_state.call_history_df
    hist = normalize_call_history(cdf)
    if not hist.empty:
        cats = sorted(hist["category"].dropna().unique().tolist())
        with st.expander(f"予測対象カテゴリの選択(コール実績 {len(cats)}分類)", expanded=(fdf is None)):
            st.caption("内部ベースライン予測は、選択カテゴリの合算を日次総コールとして扱います。既定は全カテゴリです。")
            sel_cats = st.multiselect("対象カテゴリ", cats, default=cats)
            st.session_state.selected_categories = sel_cats
            vol = hist.groupby("category")["call_count"].sum().sort_values(ascending=False)
            st.dataframe(
                vol.reset_index().rename(columns={"category": "カテゴリ", "call_count": "累計コール"}),
                use_container_width=True, height=220, hide_index=True)
    else:
        st.session_state.selected_categories = None
    nav_buttons()


# ==========================================================
# STEP2
# ==========================================================
def step2_forecast():
    st.header("STEP2 日別・時間帯別予測")
    ym = st.session_state.target_ym
    if not ym:
        st.warning("先にSTEP1で予測対象月を選択してください。")
        nav_buttons(next_ok=False)
        return
    fsrc = st.session_state.forecast_source or "内部ベースライン"
    st.caption(f"対象月：{ym} / 予測ソース：{fsrc}")

    fdf = st.session_state.forecast_log_df
    daily = None
    used_source = None
    if fdf is not None and not fdf.empty:
        f = fdf.copy()
        f = f[f["queue"] == TARGET_QUEUE] if "queue" in f.columns else f
        f["ym"] = pd.to_datetime(f["date"]).dt.strftime("%Y-%m")
        f = f[f["ym"] == ym]
        if not f.empty:
            if "created_at" in f.columns and f["created_at"].notna().any():
                f = f.sort_values("created_at").groupby("date", as_index=False).last()
            daily = f[["date", "predicted_calls"]].sort_values("date").reset_index(drop=True)
            used_source = fsrc

    if daily is None or daily.empty:
        cdf = st.session_state.call_history_df
        cats = st.session_state.selected_categories
        daily = baseline_daily_forecast(cdf, ym, categories=cats)
        used_source = "内部ベースライン予測(同一曜日直近平均)"
        if daily is None or daily.empty:
            st.error("コール実績が無いため内部ベースライン予測も生成できません。STEP0でCALL_HISTORYをご確認ください。")
            nav_buttons(next_ok=False)
            return
        cat_note = "全カテゴリ合算" if (not cats) else f"{len(cats)}カテゴリ選択"
        st.info(f"正式予測が無いため、内部ベースライン予測を生成しました（{cat_note}）。")

    st.session_state.daily_forecast = daily
    st.caption(f"採用予測：{used_source}")

    st.subheader("日別予測")
    disp = daily.rename(columns={"date": "日付", "predicted_calls": "予測コール数"})
    st.dataframe(disp, use_container_width=True, height=260)
    c1, c2, c3 = st.columns(3)
    c1.metric("対象日数", f"{len(daily)}日")
    c2.metric("月間合計コール", f"{int(daily['predicted_calls'].sum()):,}件")
    c3.metric("日平均", f"{int(daily['predicted_calls'].mean()):,}件")

    st.subheader(f"時間帯配分(X案 / {BIZ_START}〜{BIZ_END})")
    ih = st.session_state.interval_hist_df
    if ih is None:
        path = st.text_input("履歴TSVパス", value=DEFAULT_HIST_PATH)
        ih = load_local_tsv(path)
        st.session_state.interval_hist_df = ih
    if ih is None:
        st.error("30分履歴が読み込めません。")
        nav_buttons(next_ok=False)
        return

    ratio_long = compute_interval_ratio(ih)
    if ratio_long.empty:
        st.error(f"queue=='{TARGET_QUEUE}' の営業時間内データがありません。")
        nav_buttons(next_ok=False)
        return
    st.session_state.interval_ratio = ratio_long

    pivot = ratio_long.pivot(index="weekday", columns="time_slot", values="ratio").fillna(0.0)
    pivot.index = [WEEKDAY_JP[i] for i in pivot.index]
    pivot.index.name = "曜日"
    with st.expander("曜日×時間帯 配分率(%)を表示"):
        st.dataframe((pivot * 100).round(2), use_container_width=True)

    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["weekday"] = d["date"].dt.weekday
    rows = []
    for _, dr in d.iterrows():
        wd = int(dr["weekday"])
        wr = ratio_long[ratio_long["weekday"] == wd]
        for _, rr in wr.iterrows():
            rows.append({
                "date": dr["date"].date(), "weekday": WEEKDAY_JP[wd],
                "time_slot": rr["time_slot"], "ratio": rr["ratio"],
                "predicted_calls": float(dr["predicted_calls"]) * float(rr["ratio"]),
            })
    di = pd.DataFrame(rows)
    di["predicted_calls"] = di["predicted_calls"].round(2)
    st.session_state.interval_forecast = di

    st.subheader("30分単位予測")
    st.dataframe(
        di.rename(columns={"date": "日付", "weekday": "曜日", "time_slot": "時間帯",
                           "ratio": "配分率", "predicted_calls": "予測コール数"}),
        use_container_width=True, height=340)
    csv = di.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("30分単位予測CSV", csv, f"interval_forecast_{ym}.csv", "text/csv")
    nav_buttons()


# ==========================================================
# STEP3
# ==========================================================
def step3_accuracy():
    st.header("STEP3 精度確認(任意)")
    st.caption("日別予測と実績(CALL_HISTORY)を突合し、MAPE/RMSE/Biasを確認します。")
    fdf = st.session_state.forecast_log_df
    cdf = st.session_state.call_history_df
    if fdf is None or fdf.empty or cdf is None or cdf.empty:
        st.info("日別予測 または CALL_HISTORY が未取得のため、この画面はスキップできます。"
                "MAPEを見るには、STEP0で予測CSVをアップロードしてください。")
        nav_buttons()
        return

    a = normalize_call_history(cdf)
    a["date"] = pd.to_datetime(a["date"]).dt.date
    a = a.groupby("date", as_index=False)["call_count"].sum().rename(columns={"call_count": "actual_calls"})

    f = fdf.copy()
    if "created_at" in f.columns and f["created_at"].notna().any():
        _keys = ["date", "queue"] if "queue" in f.columns else ["date"]
        f = f.sort_values("created_at").groupby(_keys, as_index=False).last()
    f = f.groupby("date", as_index=False)["predicted_calls"].sum()

    joined = f.merge(a, on="date", how="inner")
    if joined.empty:
        st.warning("予測と実績の突合対象がありません（日付が一致する実績がありません）。")
        nav_buttons()
        return
    joined["weekday"] = joined["date"].apply(lambda x: WEEKDAY_JP[pd.Timestamp(x).weekday()])

    st.subheader("📌 全体サマリ")
    ov = summarize_metrics(joined, "actual_calls", "predicted_calls")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAPE", f"{ov['MAPE(%)'].iloc[0]:.2f}%")
    c2.metric("RMSE", f"{ov['RMSE'].iloc[0]:.2f}")
    c3.metric("Bias", f"{ov['Bias'].iloc[0]:.2f}")
    c4.metric("突合日数", f"{int(ov['件数'].iloc[0]):,}")

    _mape = float(ov['MAPE(%)'].iloc[0])
    if _mape >= 10.0:
        st.error("MAPE\u304c10%\u4ee5\u4e0a\u3067\u3059\u3002\u30ab\u30c6\u30b4\u30ea\u30fc\u7684\u306b\u306f\u6599\u91d1\u3084\u65b0\u898f\u95a2\u9023\u306e\u554f\u3044\u5408\u308f\u305b\u306b\u304a\u3051\u308b\u8aa4\u5dee\u7387\u304c\u9ad8\u3044\u305f\u3081\u3001\u4e88\u6e2c\u30ed\u30b8\u30c3\u30af\u306e\u898b\u76f4\u3057\u304c\u5fc5\u8981\u3067\u3059\u3002\uff08MAPE={:.2f}%\uff09".format(_mape))
    elif _mape >= 5.0:
        st.warning("MAPE\u304c5%\u4ee5\u4e0a\u3067\u3059\u3002\u30ab\u30c6\u30b4\u30ea\u30fc\u7684\u306b\u306f\u6599\u91d1\u306b\u95a2\u3059\u308b\u8aa4\u5dee\u7387\u304c\u9ad8\u3044\u3067\u3059\u3002\uff08MAPE={:.2f}%\uff09".format(_mape))
    else:
        st.success("MAPE\u306f5%\u672a\u6e80\u3067\u3059\u3002\uff08MAPE={:.2f}%\uff09".format(_mape))

    cuts = st.multiselect("切り口", ["日別", "曜日別"], default=["曜日別"])
    if "日別" in cuts:
        st.markdown("**日別**")
        st.dataframe(summarize_metrics(joined, "actual_calls", "predicted_calls", "date"), use_container_width=True)
    if "曜日別" in cuts:
        st.markdown("**曜日別**")
        st.dataframe(summarize_metrics(joined, "actual_calls", "predicted_calls", "weekday"), use_container_width=True)

    st.subheader("📈 累積MAPE推移")
    dd = joined.sort_values("date").copy()
    dd["abs_pct_err"] = np.abs((dd["actual_calls"] - dd["predicted_calls"]) /
                               dd["actual_calls"].replace(0, np.nan)) * 100
    dd["cum_mape"] = dd["abs_pct_err"].expanding().mean()
    st.line_chart(dd.set_index("date")[["cum_mape"]])
    nav_buttons()


# ==========================================================
# STEP4
# ==========================================================
def step4_staffing():
    st.header("STEP4 時間帯別 必要人員数算定")
    di = st.session_state.interval_forecast
    if di is None or di.empty:
        st.warning("先にSTEP2で時間帯別予測を作成してください。")
        nav_buttons(next_ok=False)
        return
    st.markdown("**パラメータ**")
    c1, c2, c3, c4 = st.columns(4)
    with c1: sl = st.slider("SL目標(%)", 50, 99, int(st.session_state.sl_target * 100)) / 100
    with c2: tsec = st.number_input("目標応答(秒)", 5, 120, st.session_state.target_sec, step=5)
    with c3: aht = st.number_input("AHT(秒)", 60, 3600, st.session_state.aht_sec, step=10)
    with c4: shrink = st.slider("シュリンケージ(%)", 0, 50, int(st.session_state.shrink * 100)) / 100
    st.session_state.update(sl_target=sl, target_sec=tsec, aht_sec=aht, shrink=shrink)

    df = di.copy()
    with st.spinner("アーランC計算中..."):
        df["required_raw"] = df["predicted_calls"].apply(lambda x: required_agents(x, aht, sl, tsec))
        df["required_with_shrink"] = np.ceil(df["required_raw"] / max(1e-6, 1 - shrink)).astype(int)
        df["projected_sl"] = df.apply(
            lambda r: service_level(int(r["required_raw"]), r["predicted_calls"], aht, tsec), axis=1).round(4)
    st.session_state.required_df = df

    st.markdown(
        '<div class="ywfms-note">💡 <b>必要人員(シュリンケージ込)</b>＝受電に必要な人員を、'
        '休憩・研修・欠勤等の非稼働率で割り戻した<b>シフト確保すべき人数</b>です。'
        'この人数を集めれば、余剰時間は研修・休憩・後処理に充当できます。</div>',
        unsafe_allow_html=True,
    )

    disp = df.rename(columns={
        "date": "日付", "weekday": "曜日", "time_slot": "時間帯", "predicted_calls": "予測コール数",
        "required_raw": "受電必要人員", "required_with_shrink": "必要人員(シュリンケージ込)",
        "projected_sl": "予測SL"})
    disp["予測SL"] = (disp["予測SL"] * 100).round(1).astype(str) + "%"
    st.dataframe(disp, use_container_width=True, height=340)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最大必要人員(ｼｭﾘﾝｹｰｼﾞ込)", f"{int(df['required_with_shrink'].max())}人")
    c2.metric("平均必要人員", f"{df['required_with_shrink'].mean():.1f}人")
    c3.metric("平均予測SL", f"{df['projected_sl'].mean()*100:.1f}%")
    c4.metric("月間必要工数", f"{int(df['required_with_shrink'].sum()*0.5):,}人時")
    nav_buttons()


# ==========================================================
# STEP5
# ==========================================================
def step5_intraday():
    st.header("STEP5 予実管理(当日)")
    req = st.session_state.required_df
    if req is None or req.empty:
        st.warning("先にSTEP4で必要人員を算定してください。")
        nav_buttons(next_ok=False)
        return
    r = req.copy()
    r["date"] = pd.to_datetime(r["date"]).dt.date
    days = sorted(r["date"].unique())
    target = st.selectbox("対象日", days, index=len(days) - 1)
    day = r[r["date"] == target].sort_values("time_slot").reset_index(drop=True)
    day = day[day["time_slot"].astype(str).apply(_in_biz_hours)].reset_index(drop=True)

    st.subheader("🧑‍💼 本日の稼働人員(欠勤反映)")
    st.markdown(
        '<div class="ywfms-note">💡 初期値は各ブロックの<b>必要人員(シュリンケージ込)</b>のピーク値です。'
        'これはWFM上「シフトで確保すべき人数」に相当します。当日欠勤等があれば実数へ調整してください。</div>',
        unsafe_allow_html=True,
    )
    defo = {}
    for b in BLOCKS:
        mask = day["time_slot"].apply(lambda t: in_block(t, b))
        _mx = day.loc[mask, "required_with_shrink"].max()
        defo[b] = int(_mx) if pd.notna(_mx) else 0
    c1, c2, c3 = st.columns(3)
    sb = {}
    with c1: sb["午前(9:00-12:00)"] = st.number_input("午前(9:00-12:00)", 0, value=defo["午前(9:00-12:00)"])
    with c2: sb["午後(12:00-17:00)"] = st.number_input("午後(12:00-17:00)", 0, value=defo["午後(12:00-17:00)"])
    with c3: sb["夕方(17:00-20:00)"] = st.number_input("夕方(17:00-20:00)", 0, value=defo["夕方(17:00-20:00)"])

    def staffed_of(ts):
        for b in BLOCKS:
            if in_block(ts, b):
                return sb[b]
        return 0
    day["staffed"] = day["time_slot"].apply(staffed_of)

    st.subheader("📞 実績コール数入力")
    base = day[["time_slot", "predicted_calls"]].copy()
    base["actual_calls"] = np.nan
    ed = st.data_editor(base.rename(columns={"time_slot": "時間帯", "predicted_calls": "予測コール数",
                                             "actual_calls": "実績コール数"}),
                        use_container_width=True, num_rows="fixed", height=340, key=f"ed_{target}")
    day["actual_calls"] = pd.to_numeric(ed["実績コール数"], errors="coerce").values

    aht = st.session_state.aht_sec
    sl = st.session_state.sl_target
    tsec = st.session_state.target_sec
    day["effective_calls"] = day["actual_calls"].fillna(day["predicted_calls"])
    day["required_now"] = day["effective_calls"].apply(lambda x: required_agents(x, aht, sl, tsec))
    day["shortage"] = day["required_now"] - day["staffed"]
    day["actual_sl"] = day.apply(
        lambda r_: service_level(int(r_["staffed"]), r_["effective_calls"], aht, tsec), axis=1).round(4)
    st.session_state.intraday_df = day

    st.subheader("📊 予測 vs 実績")
    entered = day["actual_calls"].notna()
    tp = day.loc[entered, "predicted_calls"].sum()
    ta = day.loc[entered, "actual_calls"].sum()
    err = ((ta - tp) / tp * 100) if tp > 0 else 0
    _peak = day.loc[day["shortage"] > 0, "shortage"].max()
    peak_short = int(_peak) if pd.notna(_peak) else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("予測誤差", f"{err:+.1f}%")
    c2.metric("ピーク不足人員", f"▲{peak_short}名" if peak_short > 0 else "なし")
    c3.metric("平均SL見込", f"{day['actual_sl'].mean()*100:.1f}%")
    c4.metric("不足コマ数", f"{int((day['shortage']>0).sum())}コマ")

    show = day[["time_slot", "predicted_calls", "actual_calls", "required_now", "staffed"]].copy()
    show["余剰/不足"] = day["shortage"].apply(surplus_number)
    show["状態"] = day["shortage"].apply(surplus_label)
    show = show.rename(columns={"time_slot": "時間帯", "predicted_calls": "予測", "actual_calls": "実績",
                                "required_now": "受電必要", "staffed": "稼働"})

    def _hl(v):
        if v < 0: return "background-color:#fdecec"
        if v > 0: return "background-color:#e9fbf3"
        return ""
    try:
        st.dataframe(show.style.map(_hl, subset=["余剰/不足"]), use_container_width=True, height=340)
    except Exception:
        st.dataframe(show, use_container_width=True, height=340)

    if st.button("❄ INTRADAY_LOG に保存"):
        try:
            s = day[["date", "time_slot", "predicted_calls", "actual_calls", "staffed",
                     "required_now", "shortage", "actual_sl"]].copy()
            s["queue"] = TARGET_QUEUE
            s["updated_at"] = datetime.now()
            n = sf_write_df(s, TBL_INTRADAY_LOG, mode="append")
            st.success(f"INTRADAY_LOG に {n} 行 append")
        except Exception as e:
            st.error(f"保存失敗：{e}")
    nav_buttons()


# ==========================================================
# STEP6
# ==========================================================
def step6_reforecast():
    st.header("STEP6 再予測(当日実績で補正)")
    day = st.session_state.intraday_df
    if day is None or day.empty:
        st.warning("先にSTEP5で実績を入力してください。")
        nav_buttons(next_ok=False)
        return
    entered = day["actual_calls"].notna()
    tp = day.loc[entered, "predicted_calls"].sum()
    ta = day.loc[entered, "actual_calls"].sum()
    if entered.sum() < 2 or tp <= 0:
        st.info("再予測には実績が2コマ以上必要です。STEP5で実績を入力してください。")
        nav_buttons()
        return
    corr = ta / tp
    day = day.copy()
    day["revised_forecast"] = np.where(entered, day["actual_calls"], day["predicted_calls"] * corr).round(1)
    aht = st.session_state.aht_sec
    sl = st.session_state.sl_target
    tsec = st.session_state.target_sec
    day["revised_required"] = day["revised_forecast"].apply(lambda x: required_agents(x, aht, sl, tsec))
    day["revised_shortage"] = day["revised_required"] - day["staffed"]
    st.session_state.intraday_df = day

    st.metric("補正係数(実績/予測)", f"{corr:.3f}", help="1.0超=想定より多い。残りコマの予測に係数を乗算。")

    show = day[["time_slot", "predicted_calls", "actual_calls", "revised_forecast",
                "revised_required", "staffed"]].copy()
    show["余剰/不足"] = day["revised_shortage"].apply(surplus_number)
    show["状態"] = day["revised_shortage"].apply(surplus_label)
    show = show.rename(columns={"time_slot": "時間帯", "predicted_calls": "元予測", "actual_calls": "実績",
                                "revised_forecast": "補正後予測", "revised_required": "補正後受電必要",
                                "staffed": "稼働"})

    def _hl(v):
        if v < 0: return "background-color:#fdecec"
        if v > 0: return "background-color:#e9fbf3"
        return ""
    try:
        st.dataframe(show.style.map(_hl, subset=["余剰/不足"]), use_container_width=True, height=380)
    except Exception:
        st.dataframe(show, use_container_width=True, height=380)

    st.line_chart(day.set_index("time_slot")[["predicted_calls", "revised_forecast"]].rename(
        columns={"predicted_calls": "元予測", "revised_forecast": "補正後予測"}))
    nav_buttons(next_label="稼働調整示唆へ ▶")


# ==========================================================
# STEP7
# ==========================================================
def step7_suggest():
    st.header("STEP7 稼働調整示唆")
    day = st.session_state.intraday_df
    if day is None or day.empty:
        st.warning("先にSTEP5/6を実施してください。")
        nav_buttons(next_ok=False)
        return

    short_col = "revised_shortage" if "revised_shortage" in day.columns else "shortage"
    d = day.copy()
    d["_short"] = d[short_col]

    _peak = d.loc[d["_short"] > 0, "_short"].max()
    total_short_headcount = int(_peak) if pd.notna(_peak) else 0
    total_surplus = int((-d["_short"].clip(upper=0)).sum())

    short_slots = d.loc[d["_short"] >= 1, "time_slot"].tolist()
    heavy_slots = d.loc[d["_short"] >= 3, "time_slot"].tolist()
    short_ranges = _fmt_slot_ranges(short_slots)
    heavy_ranges = _fmt_slot_ranges(heavy_slots)

    surplus_df = d[d["_short"] <= -1].copy()
    surplus_df["余剰"] = -surplus_df["_short"]
    surplus_top = surplus_df.sort_values("余剰", ascending=False).head(5)

    st.markdown(
        '<div class="ywfms-note">📌 <b>見方</b>：'
        '<b style="color:#12b886;">＋N名（余剰）</b>＝受電必要を上回る人員で、研修・休憩・後処理に充当できる計画枠。 '
        '<b style="color:#e03131;">▲N名（不足）</b>＝受電必要が稼働を上回る枠で、研修・休憩を他時間帯へ調整。</div>',
        unsafe_allow_html=True,
    )

    st.subheader("🗣 本日の運営アドバイス")

    if total_short_headcount <= 0 and not short_slots:
        lines = ["コール予測に対し、全時間帯で必要人員を確保できています。"
                 "余剰枠は計画どおり研修・休憩・後処理に充当できます。"]
        if not surplus_top.empty:
            top_ranges = _fmt_slot_ranges(surplus_top["time_slot"].tolist())
            lines.append(
                f"特に余剰が大きい **{'、'.join(top_ranges)}** に、研修・面談・後処理業務を寄せると効率的です。"
            )
        st.markdown(
            f'<div class="ywfms-advice ok">'
            f'<span class="headline">✅ 本日は全時間帯で必要人員を確保できています</span>'
            f'{"<br>".join(_bold_all(x) for x in lines)}</div>',
            unsafe_allow_html=True,
        )
    else:
        lines = []
        lines.append(
            f"本日はコール予測に対し、**{'、'.join(short_ranges)}** の時間帯で"
            f"最大 **▲{total_short_headcount}名** の稼働人員不足が見込まれます。"
        )
        lines.append(
            f"不足時間帯に研修・休憩が組まれている場合は、"
            f"**『研修計画表』側で他の時間帯へ変更するよう調整** してください。"
        )
        if heavy_ranges:
            lines.append(
                f"特に **{'、'.join(heavy_ranges)}** は不足幅が大きいため、優先的に受電体制を厚くしてください。"
            )
        if not surplus_top.empty:
            top_ranges = _fmt_slot_ranges(surplus_top["time_slot"].tolist())
            lines.append(
                f"一方、**{'、'.join(top_ranges)}** は余剰があるため、"
                f"研修・休憩・後処理業務はこれらの時間帯へ寄せると効率的です。"
            )
        css_cls = "crit" if total_short_headcount >= 3 else "warn"
        head = ("🚨 一部時間帯で稼働人員が不足しています" if total_short_headcount >= 3
                else "⚠️ 一部時間帯で人員調整が必要です")
        st.markdown(
            f'<div class="ywfms-advice {css_cls}">'
            f'<span class="headline">{head}</span>'
            f'{"<br>".join(_bold_all(x) for x in lines)}</div>',
            unsafe_allow_html=True,
        )

    if not surplus_top.empty:
        st.subheader("🟢 研修・休憩の推奨枠(余剰の大きい時間帯)")
        rk = surplus_top[["time_slot", "余剰"]].rename(columns={"time_slot": "時間帯", "余剰": "余剰人数"})
        rk["余剰人数"] = "＋" + rk["余剰人数"].astype(int).astype(str) + "名"
        st.dataframe(rk, use_container_width=True, hide_index=True)

    st.subheader("📋 根拠(30分単位)")
    show = d[["time_slot", "predicted_calls", "actual_calls"]].copy()
    if "revised_forecast" in d.columns:
        show["補正後予測"] = d["revised_forecast"]
    show["受電必要"] = d["required_now"]
    show["稼働"] = d["staffed"]
    show["余剰/不足"] = d["_short"].apply(surplus_number)
    show["状態"] = d["_short"].apply(surplus_label)
    show = show.rename(columns={"time_slot": "時間帯", "predicted_calls": "予測", "actual_calls": "実績"})

    def _hl(v):
        if v < 0: return "background-color:#fdecec"
        if v > 0: return "background-color:#e9fbf3"
        return ""
    try:
        st.dataframe(show.style.map(_hl, subset=["余剰/不足"]), use_container_width=True, height=380)
    except Exception:
        st.dataframe(show, use_container_width=True, height=380)

    c1, c2, c3 = st.columns(3)
    c1.metric("ピーク不足人員", f"▲{total_short_headcount}名" if total_short_headcount > 0 else "なし")
    c2.metric("不足時間帯数", f"{len(short_slots)}コマ")
    c3.metric("総余剰(コマ×人)", f"＋{total_surplus}")

    st.caption("※ ＋=余剰（研修・休憩等に充当可）／▲=不足（他時間帯へ調整）。STEP6実施時は補正後で判定。")

    csv = d.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("本日の運営データCSV", csv, "intraday_advice.csv", "text/csv")

    st.markdown("---")
    if st.button("🔄 新しい対象月で最初から", type="primary"):
        st.session_state.wizard_idx = 1
        st.rerun()
    nav_buttons(next_ok=False)


# ==========================================================
# ルーター
# ==========================================================
inject_css()
render_sidebar()

idx = st.session_state.wizard_idx
ym = st.session_state.target_ym or "—"

st.markdown(
    f"""
    <div class="ywfms-hero">
        <h1>🏢 {APP_NAME}</h1>
        <p>WFM標準フロー ： 予測 → シフティング → リアルタイムマネジメント</p>
        <div class="chips">
            <span class="chip">📍 横浜コンタクトセンター</span>
            <span class="chip">🕘 {BIZ_START}–{BIZ_END}</span>
            <span class="chip">🎯 SL 80/20</span>
            <span class="chip">📅 対象月：{ym}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

dots = []
for i in range(len(STEPS)):
    cls = "done" if i < idx else ("current" if i == idx else "")
    dots.append(f'<div class="ywfms-dot {cls}"></div>')
st.markdown(
    f'<div class="ywfms-dots">{"".join(dots)}</div>'
    f'<div class="ywfms-step-label">STEP {idx} / {len(STEPS)-1} ・ {STEPS[idx]}</div>',
    unsafe_allow_html=True,
)

if idx == 0:
    step0_load()
elif idx == 1:
    step1_month()
elif idx == 2:
    step2_forecast()
elif idx == 3:
    step3_accuracy()
elif idx == 4:
    step4_staffing()
elif idx == 5:
    step5_intraday()
elif idx == 6:
    step6_reforecast()
elif idx == 7:
    step7_suggest()
