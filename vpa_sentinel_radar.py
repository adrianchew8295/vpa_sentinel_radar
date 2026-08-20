from datetime import datetime, timedelta
import json
import os
import numpy as np
import pandas as pd
import requests

# ==============================================================================
# 0. 核心配置与 12 标的监控矩阵 (CONFIG & 12-TICKER BASKET)
# ==============================================================================
TIINGO_API_KEY = "bcffe3a5cf7eeef085e405cfa4a3e5691b976217"

# 专属 Google Sheets Web App 接口
GOOGLE_SHEET_WEB_APP_URL = "https://script.google.com/macros/s/AKfycbwu9_Asvct80WwHUMkM5iNKt9SrVujvoRKum8KQfwbOXF0PJ0-ouU5G78H2ZLTouEQP/exec"

TARGET_INDEX = "QQQ"
MAG_7 = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]
MEMORY_BASKET = ["MU", "SNDK", "WDC", "STX"]  # 存储与闪存 4 神经独立审计
ALL_TICKERS = [TARGET_INDEX] + MAG_7 + MEMORY_BASKET

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Token {TIINGO_API_KEY}",
}


# ==============================================================================
# 1. 物理量价法医引擎与态势定级 (VPA FORENSIC ENGINE & RADAR)
# ==============================================================================
def classify_vpa_radar(open_p, high_p, low_p, close_p, volume, avg_volume):
    """解剖单根 K 线做功，并输出 VPA 信号、RVol 与 WATCH/ALERT 态势定级"""
    spread = max(high_p - low_p, 0.0001)
    body = abs(close_p - open_p)
    upper_wick = high_p - max(open_p, close_p)
    lower_wick = min(open_p, close_p) - low_p

    upper_wick_ratio = upper_wick / spread
    lower_wick_ratio = lower_wick / spread
    close_pos = (close_p - low_p) / spread
    rvol = volume / (avg_volume if avg_volume > 0 else 1.0)

    # 1. 机构托底承接 -> WATCH
    if rvol >= 1.8 and lower_wick_ratio >= 0.40 and close_pos >= 0.33:
        return (
            "STOPPING_VOLUME",
            rvol,
            close_pos,
            "WATCH",
            "巨量长下影机构托底承接",
        )

    # 2. 冲高受阻出货 -> ALERT
    if rvol >= 1.8 and upper_wick_ratio >= 0.45 and close_pos <= 0.35:
        return (
            "UPTHRUST_TOPPING",
            rvol,
            close_pos,
            "ALERT",
            "放量长上影诱多倒货",
        )

    # 3. 抛盘枯竭测试 -> WATCH
    if rvol <= 0.65 and close_pos >= 0.50 and (body / spread) <= 0.40:
        return (
            "LOW_VOL_TEST",
            rvol,
            close_pos,
            "WATCH",
            "缩量窄实体极度地量测试",
        )

    # 4. 高效放量推进 -> WATCH
    if rvol >= 1.6 and (body / spread) >= 0.60 and close_p > open_p:
        return "VALID_DRIVE", rvol, close_pos, "WATCH", "放量大阳线高速真实推进"

    # 5. 缩量冲高背离 -> ALERT
    if rvol <= 0.75 and close_p > open_p and upper_wick_ratio >= 0.30:
        return (
            "VOLUME_DIVERGENCE",
            rvol,
            close_pos,
            "ALERT",
            "量能萎缩虚拉动能衰竭",
        )

    # 6. 巨量原地打滑 -> ALERT
    if rvol >= 2.0 and (body / spread) <= 0.25:
        return (
            "ABSORPTION_STALL",
            rvol,
            close_pos,
            "ALERT",
            "爆巨量但实体狭窄推进受阻",
        )

    return "NORMAL", rvol, close_pos, "NEUTRAL", "常态流动性震荡"


def calculate_volume_profile(
    df_intraday, price_col="close", vol_col="volume", bins=30
):
    """计算 Volume Profile: 精确 POC 与 70% 价值区 (VAH / VAL)"""
    if df_intraday.empty or df_intraday[vol_col].sum() == 0:
        return None, None, None

    min_p = df_intraday["low"].min()
    max_p = df_intraday["high"].max()
    if min_p == max_p:
        return round(min_p, 2), round(max_p, 2), round(min_p, 2)

    price_bins = np.linspace(min_p, max_p, bins)
    vol_profile = np.zeros(bins - 1)

    for _, row in df_intraday.iterrows():
        p = row[price_col]
        v = row[vol_col]
        idx = np.digitize(p, price_bins) - 1
        if 0 <= idx < len(vol_profile):
            vol_profile[idx] += v

    poc_idx = np.argmax(vol_profile)
    poc = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2.0

    total_vol = vol_profile.sum()
    target_va_vol = total_vol * 0.70
    accumulated_vol = vol_profile[poc_idx]

    low_idx, high_idx = poc_idx, poc_idx
    while accumulated_vol < target_va_vol and (
        low_idx > 0 or high_idx < len(vol_profile) - 1
    ):
        next_low_vol = vol_profile[low_idx - 1] if low_idx > 0 else 0
        next_high_vol = (
            vol_profile[high_idx + 1] if high_idx < len(vol_profile) - 1 else 0
        )

        if next_low_vol >= next_high_vol and low_idx > 0:
            low_idx -= 1
            accumulated_vol += next_low_vol
        elif high_idx < len(vol_profile) - 1:
            high_idx += 1
            accumulated_vol += next_high_vol
        else:
            break

    val = price_bins[low_idx]
    vah = price_bins[high_idx + 1]
    return round(poc, 2), round(vah, 2), round(val, 2)


def calculate_atr(df, period=14):
    """计算 5M / 1H 真实波幅 ATR"""
    if len(df) < period + 1:
        return 1.50
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean().iloc[-1]
    return round(float(atr), 2) if not np.isnan(atr) else 1.50


def extract_macro_vpa_zones(df_intraday, live_price, atr_val):
    """
    【自适应近端区域提取引擎】
    锁定现价 1.0 ~ 2.8 ATR 内的 1H 筹码密集带，彻底杜绝 Y 轴拉扯与盲区问题
    """
    if df_intraday.empty or len(df_intraday) < 20:
        res_top = round(live_price + atr_val * 2.0, 2)
        res_bot = round(live_price + atr_val * 1.0, 2)
        sup_top = round(live_price - atr_val * 1.0, 2)
        sup_bot = round(live_price - atr_val * 2.0, 2)
        return res_top, res_bot, sup_top, sup_bot

    df = df_intraday.copy()
    df.set_index("date", inplace=True)
    df_1h = (
        df.resample("1h")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )

    # 1. 寻找上方近端阻力密集区 (锁定在现价上方 1.0 ~ 2.8 ATR 内)
    upper_candidates = df_1h[
        (df_1h["high"] > live_price)
        & (df_1h["high"] <= live_price + atr_val * 3.0)
    ]
    if not upper_candidates.empty:
        peak_bar = upper_candidates.loc[upper_candidates["high"].idxmax()]
        res_top = round(peak_bar["high"], 2)
        body_top = max(peak_bar["open"], peak_bar["close"])
        res_bot = round(max(body_top, res_top - atr_val * 0.6), 2)
        if res_top == res_bot:
            res_bot = round(res_top - atr_val * 0.5, 2)
    else:
        # 若近端无历史密集峰，基于 ATR 动态生成近端天花板
        res_top = round(live_price + atr_val * 2.0, 2)
        res_bot = round(live_price + atr_val * 1.2, 2)

    # 2. 寻找下方近端承接密集区 (锁定在现价下方 1.0 ~ 2.8 ATR 内)
    lower_candidates = df_1h[
        (df_1h["low"] < live_price)
        & (df_1h["low"] >= live_price - atr_val * 3.0)
    ]
    if not lower_candidates.empty:
        trough_bar = lower_candidates.loc[lower_candidates["low"].idxmin()]
        sup_bot = round(trough_bar["low"], 2)
        body_bot = min(trough_bar["open"], trough_bar["close"])
        sup_top = round(min(body_bot, sup_bot + atr_val * 0.6), 2)
        if sup_top == sup_bot:
            sup_top = round(sup_bot + atr_val * 0.5, 2)
    else:
        # 若近端无历史密集谷，基于 ATR 动态生成近端地板
        sup_top = round(live_price - atr_val * 1.2, 2)
        sup_bot = round(live_price - atr_val * 2.0, 2)

    # 安全断言：确保上界绝对大于下界
    if res_top <= res_bot:
        res_top = round(res_bot + atr_val * 0.5, 2)
    if sup_top <= sup_bot:
        sup_top = round(sup_bot + atr_val * 0.5, 2)

    return res_top, res_bot, sup_top, sup_bot


# ==============================================================================
# 2. Tiingo API 数据提取管道 (REAL-TIME & MULTI-DAY INTRADAY)
# ==============================================================================
def fetch_tiingo_daily(ticker, days=252):
    start_date = (datetime.now() - timedelta(days=int(days * 1.5))).strftime(
        "%Y-%m-%d"
    )
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&token={TIINGO_API_KEY}"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            df = pd.DataFrame(res.json())
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                return df
    except Exception as e:
        print(f"[-] Daily fetch notice for {ticker}: {e}")
    return pd.DataFrame()


def fetch_tiingo_realtime_quote(ticker):
    url = f"https://api.tiingo.com/iex/{ticker}?token={TIINGO_API_KEY}"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]
    except Exception as e:
        print(f"[-] Realtime quote notice for {ticker}: {e}")
    return {}


def fetch_tiingo_intraday_multi_days(ticker, days=5):
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = f"https://api.tiingo.com/iex/{ticker}/prices?startDate={start_date}&resampleFreq=5min&columns=date,open,high,low,close,volume&token={TIINGO_API_KEY}"
    try:
        res = requests.get(url, headers=HEADERS, timeout=12)
        if res.status_code == 200:
            df = pd.DataFrame(res.json())
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        print(f"[-] Multi-day Intraday fetch notice for {ticker}: {e}")
    return pd.DataFrame()


# ==============================================================================
# 3. Google Sheets 自动化推送模块 (AUTO CLOUD SYNC)
# ==============================================================================
def push_to_google_sheets(
    df_qqq, df_breadth, df_dip, bias, mag7_ratio, mem_pulse
):
    if not GOOGLE_SHEET_WEB_APP_URL:
        print("[-] 未配置 Google Sheet Web App URL，跳过云端同步。")
        return

    print("\n⏳ 正在自动同步数据至 Google Sheets...")
    today_str = datetime.now().strftime("%Y-%m-%d")
    qqq_row = df_qqq.iloc[0].to_dict()

    # 1. Today_War_Room 载荷
    war_room_headers = [
        "Date",
        "Symbol",
        "HTF_Bias",
        "AMT_Open_Status",
        "RES_TOP",
        "RES_BOT",
        "PMH",
        "Premarket_POC",
        "PML",
        "Yesterday_POC",
        "Yesterday_VAH",
        "Yesterday_VAL",
        "SUP_TOP",
        "SUP_BOT",
        "Hard_SL",
        "ATR_SL",
        "Live_Price",
    ]
    war_room_row = [
        today_str,
        "QQQ",
        bias,
        qqq_row.get("AMT_Open_Status", "NORMAL"),
        qqq_row["RES_TOP"],
        qqq_row["RES_BOT"],
        qqq_row["PMH"],
        qqq_row["Premarket_POC"],
        qqq_row["PML"],
        qqq_row["Yesterday_POC"],
        qqq_row["Yesterday_VAH"],
        qqq_row["Yesterday_VAL"],
        qqq_row["SUP_TOP"],
        qqq_row["SUP_BOT"],
        qqq_row["Hard_SL"],
        qqq_row.get("ATR_SL", qqq_row["Hard_SL"]),
        qqq_row["Last_Premarket_Price"],
    ]

    # 2. Daily_History 追加行载荷
    history_row = [
        today_str,
        "QQQ",
        bias,
        qqq_row["Premarket_POC"],
        qqq_row["PMH"],
        qqq_row["PML"],
        qqq_row["Yesterday_POC"],
        qqq_row["Yesterday_VAH"],
        qqq_row["Yesterday_VAL"],
        mag7_ratio,
        mem_pulse,
    ]

    # 3. 市场宽度与折扣池载荷
    breadth_headers = list(df_breadth.columns)
    breadth_rows = df_breadth.values.tolist()
    dip_headers = list(df_dip.columns)
    dip_rows = df_dip.values.tolist()

    payload = {
        "war_room": {"headers": war_room_headers, "row": war_room_row},
        "history_row": history_row,
        "breadth": {"headers": breadth_headers, "rows": breadth_rows},
        "dip_pool": {"headers": dip_headers, "rows": dip_rows},
    }

    try:
        res = requests.post(GOOGLE_SHEET_WEB_APP_URL, json=payload, timeout=15)
        if res.status_code == 200 and "SUCCESS" in res.text:
            print(" Google Sheets 4 大工作表已全部全自动同步更新！")
        else:
            print(f"[-] Google Sheet 同步响应: {res.text}")
    except Exception as e:
        print(f"[-] Google Sheet 同步请求失败: {e}")


# ==============================================================================
# 4. 主执行流与数据聚合 (PIPELINE AGGREGATION)
# ==============================================================================
def generate_sentinel_report():
    print(
        "[*] 正在启动 VPA SENTINEL RADAR (QQQ + 7 巨头 + 4 闪存 态势感知雷达)..."
    )

    # 1. 抓取 QQQ 多周期与实时数据
    qqq_daily = fetch_tiingo_daily(TARGET_INDEX, days=252)
    qqq_intra_all = fetch_tiingo_intraday_multi_days(TARGET_INDEX, days=5)
    qqq_quote = fetch_tiingo_realtime_quote(TARGET_INDEX)

    yesterday_close = (
        round(qqq_daily["close"].iloc[-1], 2)
        if not qqq_daily.empty
        else 717.50
    )
    live_price = (
        qqq_quote.get("tngoLast") or qqq_quote.get("last") or yesterday_close
    )
    live_price = round(float(live_price), 2)

    # 2. 计算 ATR 与日内多空锚点
    atr_val = 1.50
    y_poc, y_vah, y_val = (
        yesterday_close,
        round(yesterday_close * 1.004, 2),
        round(yesterday_close * 0.996, 2),
    )
    pmh, pml, poc, vah, val = (
        round(live_price * 1.003, 2),
        round(live_price * 0.997, 2),
        live_price,
        live_price,
        live_price,
    )

    if not qqq_intra_all.empty:
        qqq_intra_all["date_only"] = qqq_intra_all["date"].dt.date
        unique_dates = sorted(qqq_intra_all["date_only"].unique())

        # 今日盘前
        today_df = qqq_intra_all[qqq_intra_all["date_only"] == unique_dates[-1]]
        if len(today_df) >= 2:
            pmh = round(today_df["high"].max(), 2)
            pml = round(today_df["low"].min(), 2)
            poc, vah, val = calculate_volume_profile(today_df)

        # 昨日 RTH 价值区
        if len(unique_dates) >= 2:
            yesterday_df = qqq_intra_all[
                qqq_intra_all["date_only"] == unique_dates[-2]
            ]
            if not yesterday_df.empty:
                y_poc, y_vah, y_val = calculate_volume_profile(yesterday_df)

        atr_val = calculate_atr(qqq_intra_all, period=14)

    # 3. 提取 4H/1H 自适应近端抛压与承接带
    res_top, res_bot, sup_top, sup_bot = extract_macro_vpa_zones(
        qqq_intra_all, live_price, atr_val
    )

    # 拍卖市场理论 (AMT) 状态判定
    if live_price > y_vah:
        amt_status = "ABOVE_Y_VAH (BULLISH_IMBALANCE)"
    elif live_price < y_val:
        amt_status = "BELOW_Y_VAL (BEARISH_IMBALANCE)"
    else:
        amt_status = "INSIDE_Y_VALUE (RANGE_BALANCE_CHOP_RISK)"

    hard_sl = round(pml * 0.997, 2)
    atr_sl = round(pml - (0.25 * atr_val), 2)
    no_trade_zone = f"{round(pml + (poc - pml)*0.3, 2)} - {round(poc + (pmh - poc)*0.3, 2)}"

    df_qqq_summary = pd.DataFrame(
        [
            {
                "Ticker": "QQQ",
                "RES_TOP": res_top,
                "RES_BOT": res_bot,
                "SUP_TOP": sup_top,
                "SUP_BOT": sup_bot,
                "PMH": pmh,
                "PML": pml,
                "Premarket_POC": poc,
                "VAH": vah,
                "VAL": val,
                "Yesterday_POC": y_poc,
                "Yesterday_VAH": y_vah,
                "Yesterday_VAL": y_val,
                "AMT_Open_Status": amt_status,
                "Hard_SL": hard_sl,
                "ATR_5m": atr_val,
                "ATR_SL": atr_sl,
                "No_Trade_Zone": no_trade_zone,
                "Yesterday_Close": yesterday_close,
                "Last_Premarket_Price": live_price,
                "Status": "ABOVE_POC" if live_price >= poc else "BELOW_POC",
            }
        ]
    )

    # 4. 11 权重标的 VPA 态势感知雷达与 -20% 折扣监控
    breadth_rows, dip_rows = [], []
    mag7_bullish_count, memory_bullish_count = 0, 0

    for ticker in ALL_TICKERS:
        daily_df = fetch_tiingo_daily(ticker, days=252)
        intra_df = fetch_tiingo_intraday_multi_days(ticker, days=2)
        quote = fetch_tiingo_realtime_quote(ticker)

        peak_high = (
            round(daily_df["high"].max(), 2) if not daily_df.empty else 100.0
        )
        last_close = (
            round(daily_df["close"].iloc[-1], 2)
            if not daily_df.empty
            else 90.0
        )
        curr_price = float(
            quote.get("tngoLast") or quote.get("last") or last_close
        )
        curr_price = round(curr_price, 2)
        drawdown_pct = round(((curr_price - peak_high) / peak_high) * 100, 2)

        if ticker == "QQQ":
            group = "INDEX"
        elif ticker in MAG_7:
            group = "MAG_7"
            if curr_price >= last_close:
                mag7_bullish_count += 1
        else:
            group = "MEMORY_FLASH"
            if curr_price >= last_close:
                memory_bullish_count += 1

        vpa_sig, rvol, close_pos, radar_status, reason = (
            "NORMAL",
            1.0,
            0.5,
            "NEUTRAL",
            "常态波动",
        )
        if not intra_df.empty and len(intra_df) > 0:
            last_row = intra_df.iloc[-1]
            avg_vol = intra_df["volume"].mean()
            vpa_sig, rvol, close_pos, radar_status, reason = classify_vpa_radar(
                last_row["open"],
                last_row["high"],
                last_row["low"],
                last_row["close"],
                last_row["volume"],
                avg_vol,
            )

        breadth_rows.append(
            {
                "Ticker": ticker,
                "Group": group,
                "Current_Price": curr_price,
                "Premarket_RVol": round(rvol, 2),
                "Close_Pos": round(close_pos, 2),
                "VPA_Signal": vpa_sig,
                "Radar_Status": radar_status,
                "Forensic_Note": reason,
            }
        )

        if ticker != "QQQ":
            in_buy_zone = drawdown_pct <= -20.0
            dip_rows.append(
                {
                    "Ticker": ticker,
                    "Group": group,
                    "52W_Peak": peak_high,
                    "Current_Price": curr_price,
                    "Drawdown_%": f"{drawdown_pct}%",
                    "Status": "IN_BUY_ZONE" if in_buy_zone else "MONITORING",
                    "VPA_Setup": vpa_sig,
                    "Buy_Trigger": curr_price if in_buy_zone else "-",
                    "Hard_SL": round(curr_price * 0.96, 2)
                    if in_buy_zone
                    else "-",
                    "TP1 (+15%)": round(curr_price * 1.15, 2)
                    if in_buy_zone
                    else "-",
                    "TP2 (+25%)": round(curr_price * 1.25, 2)
                    if in_buy_zone
                    else "-",
                }
            )

    df_breadth = pd.DataFrame(breadth_rows)
    df_dip = pd.DataFrame(dip_rows)

    htf_bias = (
        "BULLISH"
        if live_price >= poc and mag7_bullish_count >= 4
        else "RANGE_BALANCE"
    )
    if live_price < poc and mag7_bullish_count <= 2:
        htf_bias = "BEARISH"

    mag7_ratio_str = f"{mag7_bullish_count}/7"
    mem_pulse_str = "STRONG" if memory_bullish_count >= 2 else "WEAK_DRIFT"

    # 5. 生成本地 Excel
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
    excel_filename = f"VPA_Sentinel_{timestamp_str}.xlsx"
    with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
        df_qqq_summary.to_excel(
            writer, sheet_name="QQQ_Coordinates", index=False
        )
        df_breadth.to_excel(
            writer, sheet_name="Market_Breadth_12", index=False
        )
        df_dip.to_excel(writer, sheet_name="Dip_To_Swing_20Pct", index=False)

    print(f"\n[+] 战役 Excel 生成成功: {excel_filename}")

    # 6. 推送至 Google Sheets
    push_to_google_sheets(
        df_qqq_summary,
        df_breadth,
        df_dip,
        htf_bias,
        mag7_ratio_str,
        mem_pulse_str,
    )

    # 7. 富途指标参数注入代码 (纯净 7 行，直接全选复制覆盖)
    print("\n" + "=" * 50)
    print(f"RES_TOP := {res_top:.2f};")
    print(f"RES_BOT := {res_bot:.2f};")
    print(f"SUP_TOP := {sup_top:.2f};")
    print(f"SUP_BOT := {sup_bot:.2f};")
    print(f"PMH_VAL := {pmh:.2f};")
    print(f"POC_VAL := {poc:.2f};")
    print(f"PML_VAL := {pml:.2f};")
    print("=" * 50)

    # 8. 打印结构化 JSON Payload
    print("\n" + "=" * 80)
    print("      发送给 Gemini 的结构化 RAW DATA PAYLOAD")
    print("=" * 80)
    payload_text = {
        "QQQ_Coordinates": df_qqq_summary.to_dict(orient="records")[0],
        "HTF_Bias": htf_bias,
        "AMT_Open_Status": amt_status,
        "Breadth_Summary": {
            "Mag7_Bullish_Count": mag7_ratio_str,
            "Memory_Pulse": mem_pulse_str,
            "Details": df_breadth.to_dict(orient="records"),
        },
        "Dip_Buys_Triggered": df_dip[
            df_dip["Status"] == "IN_BUY_ZONE"
        ].to_dict(orient="records"),
    }
    print(json.dumps(payload_text, indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    generate_sentinel_report()
