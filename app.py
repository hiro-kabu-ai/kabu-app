import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 設定 ---
st.set_page_config(page_title="複合シグナル検証ツール", page_icon="📊", layout="wide")

# 日本株コード整形
def format_ticker(ticker):
    ticker = str(ticker)
    ticker = ticker.translate(str.maketrans({chr(0xFF10 + i): chr(0x30 + i) for i in range(10)}))
    if not ticker.endswith(".T") and ticker.isdigit():
        return ticker + ".T"
    return ticker

# --- データ取得 ---
@st.cache_data
def get_stock_data(ticker, period="5y"): 
    try:
        df = yf.download(ticker, period=period, progress=False)
        
        company_name = ticker
        try:
            ticker_info = yf.Ticker(ticker)
            info = ticker_info.info
            company_name = info.get('longName', info.get('shortName', ticker))
        except:
            pass
            
        if len(df) == 0: return None, None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df, company_name
    except:
        return None, None

# --- 指標計算 ---
def add_indicators(df, params):
    df = df.copy()
    
    # RSI
    rsi_n = params.get('rsi_n', 14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_n).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # SMA
    if params['use_ma_entry']:
        df['SMA'] = df['Close'].rolling(window=params['ma_n']).mean()
        
    # BB
    if params['use_bb_entry']:
        sma_bb = df['Close'].rolling(window=params['bb_n']).mean()
        std = df['Close'].rolling(window=params['bb_n']).std()
        df['BB_Upper'] = sma_bb + (std * params['bb_sigma'])
        df['BB_Lower'] = sma_bb - (std * params['bb_sigma'])
        
    return df

# --- バックテスト実行 ---
def backtest_strategy(df, params, lot_size):
    position = 0
    entry_price = 0
    trade_log = []
    
    buy_signals = []
    sell_signals = []
    
    # パラメータ展開
    use_rsi = params['use_rsi_entry']
    use_ma = params['use_ma_entry']
    use_bb = params['use_bb_entry']
    
    take_profit_pct = params['take_profit_pct'] / 100
    stop_loss_pct = params['stop_loss_pct'] / 100
    
    for i in range(len(df)):
        price = df['Close'].iloc[i]
        date = df.index[i]
        
        # 指標取得（計算していない指標は参照しないよう注意）
        rsi = df['RSI'].iloc[i] if 'RSI' in df.columns else np.nan
        sma = df['SMA'].iloc[i] if 'SMA' in df.columns else np.nan
        bb_lower = df['BB_Lower'].iloc[i] if 'BB_Lower' in df.columns else np.nan
        
        # NaNチェック
        if pd.isna(rsi): # RSIは必須計算にしているのでチェック
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)
            continue

        # ==========================
        # 🟢 買い判定 (エントリー)
        # ==========================
        buy_condition = True
        
        if use_rsi and not (rsi <= params['rsi_buy_thresh']): buy_condition = False
        if use_ma and not (price > sma): buy_condition = False
        if use_bb and not (price <= bb_lower): buy_condition = False
        if not (use_rsi or use_ma or use_bb): buy_condition = False # 何も選んでなければ買わない

        # ==========================
        # 🔴 売り判定 (エグジット)
        # ==========================
        sell_condition = False
        sell_reason = ""
        
        if position == 1:
            # 1. 利確・損切り判定 (優先)
            pnl_pct = (price - entry_price) / entry_price
            
            if pnl_pct >= take_profit_pct:
                sell_condition = True
                sell_reason = "利確"
            elif pnl_pct <= -stop_loss_pct:
                sell_condition = True
                sell_reason = "損切"
            
            # 2. テクニカル売り (オプション)
            # もし「RSI高値売り」も併用したい場合はここで判定
            # if rsi >= 70: sell_condition = True

        # ==========================
        # 注文執行
        # ==========================
        if position == 0 and buy_condition:
            position = 1
            entry_price = price
            buy_signals.append(price)
            sell_signals.append(np.nan)
            trade_log.append({
                '日付': date, '売買': '買い', 
                '単価': price, '株数': lot_size, '損益': 0, '理由': 'Entry'
            })
            
        elif position == 1 and sell_condition:
            position = 0
            profit_per_share = price - entry_price
            total_profit = profit_per_share * lot_size
            buy_signals.append(np.nan)
            sell_signals.append(price)
            trade_log.append({
                '日付': date, '売買': '売り', 
                '単価': price, '株数': lot_size, '損益': total_profit, '理由': sell_reason
            })
            
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)

    df['Buy_Signal'] = buy_signals
    df['Sell_Signal'] = sell_signals
    
    return df, pd.DataFrame(trade_log)

# ==========================================
# UI設計
# ==========================================
st.sidebar.header("🔧 設定パネル")

# 1. 銘柄と資金
input_ticker = st.sidebar.text_input("銘柄コード", "7203")
ticker = format_ticker(input_ticker)

with st.sidebar.expander("💰 資金・ロット設定", expanded=False):
    initial_capital = st.number_input("元手 (円)", value=1000000, step=100000)
    lot_size = st.number_input("取引株数 (株)", value=100, step=100)

st.sidebar.markdown("---")

# ==========================
# 🟢 買い条件 (エントリー)
# ==========================
st.sidebar.subheader("🟢 買い条件 (エントリー)")

# RSI設定
use_rsi_entry = st.sidebar.checkbox("RSI (逆張り)", value=True, key="rsi_in")
rsi_buy_thresh = 30
rsi_n = 14
if use_rsi_entry:
    rsi_n = st.sidebar.slider("RSI期間", 5, 30, 14, key="rsi_n_slider")
    rsi_buy_thresh = st.sidebar.slider("買い基準 (RSI以下)", 10, 50, 30, key="rsi_buy_slider")

# 移動平均設定
use_ma_entry = st.sidebar.checkbox("移動平均 (トレンド)", value=False, key="ma_in")
ma_n = 25
if use_ma_entry:
    ma_n = st.sidebar.slider("MA期間 (価格 > MA)", 5, 200, 25, key="ma_n_slider")

# ボリンジャー設定
use_bb_entry = st.sidebar.checkbox("ボリンジャー (逆張り)", value=False, key="bb_in")
bb_n = 20; bb_sigma = 2.0
if use_bb_entry:
    bb_n = st.sidebar.slider("BB期間", 10, 50, 20, key="bb_n_slider")
    bb_sigma = st.sidebar.slider("σ (シグマ)", 1.0, 3.0, 2.0, key="bb_s_slider")

st.sidebar.markdown("---")

# ==========================
# 🔴 売り条件 (エグジット)
# ==========================
st.sidebar.subheader("🔴 売り条件 (エグジット)")

# 利確・損切り設定
col_p, col_l = st.sidebar.columns(2)
with col_p:
    take_profit_pct = st.number_input("利確 (%)", value=5.0, step=0.5)
with col_l:
    stop_loss_pct = st.number_input("損切 (%)", value=3.0, step=0.5)

st.sidebar.caption(f"💡 買い値から +{take_profit_pct}% で利益確定、 -{stop_loss_pct}% で損切りします")


# パラメータまとめ
params = {
    'use_rsi_entry': use_rsi_entry, 'rsi_n': rsi_n, 'rsi_buy_thresh': rsi_buy_thresh,
    'use_ma_entry': use_ma_entry, 'ma_n': ma_n,
    'use_bb_entry': use_bb_entry, 'bb_n': bb_n, 'bb_sigma': bb_sigma,
    'take_profit_pct': take_profit_pct, 'stop_loss_pct': stop_loss_pct
}

# ==========================================
# メイン画面処理
# ==========================================

with st.spinner('データと社名を取得中...'):
    df_origin, company_name = get_stock_data(ticker)

if df_origin is not None:
    st.title(f"📊 {company_name} ({ticker})")
    
    # 計算実行
    df_calc = add_indicators(df_origin, params)
    df_result, trade_log_df = backtest_strategy(df_calc, params, lot_size)
    
    # --- 結果表示 ---
    if not trade_log_df.empty:
        total_profit = trade_log_df['損益'].sum()
        final_capital = initial_capital + total_profit
        roi = (total_profit / initial_capital) * 100
        
        wins = trade_log_df[(trade_log_df['売買']=='売り') & (trade_log_df['損益'] > 0)]
        loses = trade_log_df[(trade_log_df['売買']=='売り') & (trade_log_df['損益'] <= 0)]
        
        win_rate = len(wins) / (len(wins) + len(loses)) * 100 if (len(wins) + len(loses)) > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("最終資金", f"{final_capital:,.0f}円", f"{total_profit:,.0f}円 ({roi:+.1f}%)")
        col2.metric("勝率", f"{win_rate:.1f}%", f"{len(wins)}勝 {len(loses)}敗")
        col3.metric("設定", f"利確 {take_profit_pct}% / 損切 {stop_loss_pct}%", f"{lot_size}株")
    else:
        st.warning("条件に合う取引がありませんでした。")

    # チャート
    st.subheader("📉 資産推移と売買ポイント")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])

    fig.add_trace(go.Scatter(x=df_result.index, y=df_result['Close'], mode='lines', name='株価', line=dict(color='gray')), row=1, col=1)
    
    # エントリーに使った指標だけ表示
    if use_ma_entry:
        fig.add_trace(go.Scatter(x=df_result.index, y=df_result['SMA'], name='MA', line=dict(color='orange')), row=1, col=1)
    if use_bb_entry:
        fig.add_trace(go.Scatter(x=df_result.index, y=df_result['BB_Upper'], name='+2σ', line=dict(color='blue', width=0.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_result.index, y=df_result['BB_Lower'], name='-2σ', line=dict(color='blue', width=0.5)), row=1, col=1)

    buy_pts = df_result[df_result['Buy_Signal'].notna()]
    sell_pts = df_result[df_result['Sell_Signal'].notna()]
    fig.add_trace(go.Scatter(x=buy_pts.index, y=buy_pts['Buy_Signal'], mode='markers', name='買い', marker=dict(symbol='triangle-up', size=12, color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_pts.index, y=sell_pts['Sell_Signal'], mode='markers', name='売り', marker=dict(symbol='triangle-down', size=12, color='blue')), row=1, col=1)

    # RSIは常に表示(サブチャート)
    fig.add_trace(go.Scatter(x=df_result.index, y=df_result['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=rsi_buy_thresh, line_dash="dash", line_color="red", annotation_text="買い", row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_layout(height=600, margin=dict(t=20, b=20, l=20, r=20), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    if not trade_log_df.empty:
        st.subheader("📝 取引詳細")
        trade_log_df['日付'] = trade_log_df['日付'].dt.strftime('%Y-%m-%d')
        # 理由カラムを追加して表示
        st.dataframe(trade_log_df[['日付', '売買', '単価', '株数', '損益', '理由']], use_container_width=True)

else:
    st.error("データ取得エラー: 銘柄コードを確認してください")