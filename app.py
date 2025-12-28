import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 設定 ---
st.set_page_config(page_title="Pro株分析AI Ver.2", page_icon="📊", layout="wide")

# 人気銘柄リスト
POPULAR_STOCKS = {
    "7203.T": "トヨタ自動車",
    "9984.T": "ソフトバンクG",
    "8306.T": "三菱UFJ",
    "6920.T": "レーザーテック",
    "6758.T": "ソニーG",
    "9983.T": "ファーストリテイリング",
    "8035.T": "東京エレクトロン",
    "4502.T": "武田薬品",
    "9432.T": "NTT",
    "7974.T": "任天堂",
    "6861.T": "キーエンス",
    "6098.T": "リクルート",
    "4063.T": "信越化学",
    "6301.T": "コマツ",
    "8058.T": "三菱商事",
    "1570.T": "日経レバETF"
}

# --- データ取得 ---
@st.cache_data
def get_stock_data(ticker, period="1y"): 
    try:
        df = yf.download(ticker, period=period, progress=False)
        if len(df) == 0: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return None

# --- 指標計算 ---
def add_indicators(df, params):
    df = df.copy()
    
    # 1. RSI
    rsi_n = params.get('rsi_n', 14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_n).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. VWAP (20日)
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['Typical_Price'] * df['Volume']
    vwap_window = 20
    df['VWAP'] = df['VP'].rolling(window=vwap_window).sum() / df['Volume'].rolling(window=vwap_window).sum()
    
    # 3. SMA / BB
    if params.get('use_ma_entry'):
        df['SMA'] = df['Close'].rolling(window=params['ma_n']).mean()
    if params.get('use_bb_entry'):
        sma_bb = df['Close'].rolling(window=params['bb_n']).mean()
        std = df['Close'].rolling(window=params['bb_n']).std()
        df['BB_Upper'] = sma_bb + (std * params['bb_sigma'])
        df['BB_Lower'] = sma_bb - (std * params['bb_sigma'])

    # 4. MACD
    # EMA(12) - EMA(26)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # 5. ADX (DMI)
    # +DM, -DM, TR計算
    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()
    df['+DM'] = np.where((high_diff > 0) & (high_diff > -low_diff), high_diff, 0)
    df['-DM'] = np.where((low_diff < 0) & (-low_diff > high_diff), -low_diff, 0) # low_diffは負の値なので-をつける
    
    df['TR'] = pd.concat([
        df['High'] - df['Low'], 
        (df['High'] - df['Close'].shift(1)).abs(), 
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    
    adx_n = 14
    tr_smooth = df['TR'].rolling(window=adx_n).sum()
    plus_dm_smooth = df['+DM'].rolling(window=adx_n).sum()
    minus_dm_smooth = df['-DM'].rolling(window=adx_n).sum()
    
    df['+DI'] = 100 * (plus_dm_smooth / tr_smooth)
    df['-DI'] = 100 * (minus_dm_smooth / tr_smooth)
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    df['ADX'] = df['DX'].rolling(window=adx_n).mean()

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
    use_vwap = params['use_vwap_entry']
    use_ma = params['use_ma_entry']
    use_bb = params['use_bb_entry']
    use_macd = params['use_macd_entry'] # New
    use_adx = params['use_adx_filter']  # New
    
    use_rsi_exit = params['use_rsi_exit']
    
    take_profit_pct = params['take_profit_pct'] / 100
    stop_loss_pct = params['stop_loss_pct'] / 100
    
    for i in range(len(df)):
        price = df['Close'].iloc[i]
        date = df.index[i]
        
        rsi = df['RSI'].iloc[i] if 'RSI' in df.columns else np.nan
        vwap = df['VWAP'].iloc[i] if 'VWAP' in df.columns else np.nan
        sma = df['SMA'].iloc[i] if 'SMA' in df.columns else np.nan
        bb_lower = df['BB_Lower'].iloc[i] if 'BB_Lower' in df.columns else np.nan
        
        macd = df['MACD'].iloc[i] if 'MACD' in df.columns else np.nan
        macd_sig = df['MACD_Signal'].iloc[i] if 'MACD_Signal' in df.columns else np.nan
        adx = df['ADX'].iloc[i] if 'ADX' in df.columns else np.nan

        # 計算用データ不足時スキップ
        if pd.isna(rsi) or (use_macd and pd.isna(macd)) or (use_adx and pd.isna(adx)):
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)
            continue

        # ==========================
        # 🟢 買い判定 (AND条件)
        # ==========================
        buy_condition = True
        
        # 1. RSI (逆張り)
        if use_rsi and not (rsi <= params['rsi_buy_thresh']): 
            buy_condition = False
            
        # 2. VWAP (価格帯)
        if use_vwap:
            lower_limit = vwap * (1 - params['vwap_low_pct'] / 100)
            upper_limit = vwap * (1 + params['vwap_high_pct'] / 100)
            if not (lower_limit <= price <= upper_limit):
                buy_condition = False
        
        # 3. MACD (トレンドフォロー)
        # MACD > Signal (ゴールデンクロス中) なら買い
        if use_macd and not (macd > macd_sig):
            buy_condition = False
            
        # 4. ADX (トレンド強度フィルター)
        # ADXが指定値以上(トレンド発生中)でなければ買わない
        if use_adx and not (adx >= params['adx_thresh']):
            buy_condition = False

        # その他 (MA, BB)
        if use_ma and not (price > sma): buy_condition = False
        if use_bb and not (price <= bb_lower): buy_condition = False
        
        # 何も選んでなければ買わない
        if not any([use_rsi, use_vwap, use_ma, use_bb, use_macd]):
            buy_condition = False

        # ==========================
        # 🔴 売り判定 (OR条件)
        # ==========================
        sell_condition = False
        sell_reason = ""
        
        if position == 1:
            # 1. 損益決済
            pnl_pct = (price - entry_price) / entry_price
            if pnl_pct >= take_profit_pct:
                sell_condition = True; sell_reason = "利確"
            elif pnl_pct <= -stop_loss_pct:
                sell_condition = True; sell_reason = "損切"
            
            # 2. テクニカル決済
            if not sell_condition:
                # RSI売り
                if use_rsi_exit and rsi >= params['rsi_sell_thresh']:
                    sell_condition = True; sell_reason = f"RSI高({int(rsi)})"
                
                # MACD売り（デッドクロス）※オプション
                # if use_macd and (macd < macd_sig):
                #    sell_condition = True; sell_reason = "MACDクロス"

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
st.title("⚡ Pro株分析AI Ver.2.0")
st.caption("MACD, ADXを含む高度な複合シグナル分析ツール")

# --- サイドバー設定 ---
st.sidebar.header("🔍 設定パネル")

# 1. 銘柄複数選択
default_tickers = ["7203.T", "6920.T", "1570.T"]
selected_tickers = st.sidebar.multiselect(
    "分析対象 (複数選択可)",
    options=list(POPULAR_STOCKS.keys()),
    default=default_tickers,
    format_func=lambda x: f"{POPULAR_STOCKS[x]} ({x})"
)

st.sidebar.markdown("---")

# 2. 条件設定
with st.sidebar.expander("⚙️ 戦略・パラメータ設定", expanded=True):
    st.subheader("🟢 買いエントリー条件")
    st.caption("チェックした全条件を満たす時にエントリー")
    
    # MACD (New!)
    use_macd_entry = st.sidebar.checkbox("MACD (上昇トレンド)", value=False)
    
    # ADX (New!)
    use_adx_filter = st.sidebar.checkbox("ADX (トレンド発生中のみ)", value=False)
    adx_thresh = 25
    if use_adx_filter:
        adx_thresh = st.sidebar.slider("ADX値 以上", 10, 50, 25, help="25以上でトレンド発生とみなすのが一般的")

    # RSI
    use_rsi_entry = st.sidebar.checkbox("RSI (売られすぎ)", value=True)
    rsi_buy_thresh = 30
    if use_rsi_entry:
        rsi_buy_thresh = st.sidebar.slider("RSI値 以下", 10, 50, 30)

    # VWAP
    use_vwap_entry = st.sidebar.checkbox("VWAP (価格帯)", value=False)
    vwap_high_pct = 1.0; vwap_low_pct = 3.0
    if use_vwap_entry:
        col_v1, col_v2 = st.columns(2)
        with col_v1: vwap_high_pct = st.number_input("上 (+%)", value=1.0)
        with col_v2: vwap_low_pct = st.number_input("下 (-%)", value=3.0)
            
    # その他
    use_ma_entry = st.sidebar.checkbox("MA (価格 > 移動平均)", value=False)
    use_bb_entry = st.sidebar.checkbox("BB (-2σ割れ)", value=False)

    st.markdown("---")
    
    # --- 売り設定 ---
    st.subheader("🔴 売りエグジット条件")
    
    # 損益
    col_p, col_l = st.columns(2)
    with col_p: take_profit_pct = st.number_input("利確 (%)", value=5.0, step=0.5)
    with col_l: stop_loss_pct = st.number_input("損切 (%)", value=3.0, step=0.5)
        
    # RSI売り
    use_rsi_exit = st.sidebar.checkbox("RSI (買われすぎ)", value=False)
    rsi_sell_thresh = 70
    if use_rsi_exit:
        rsi_sell_thresh = st.sidebar.slider("RSI値 以上", 50, 95, 75)

    st.markdown("---")
    lot_size = st.number_input("1回の株数", value=100)

# パラメータまとめ
params = {
    'use_rsi_entry': use_rsi_entry, 'rsi_n': 14, 'rsi_buy_thresh': rsi_buy_thresh,
    'use_vwap_entry': use_vwap_entry, 'vwap_high_pct': vwap_high_pct, 'vwap_low_pct': vwap_low_pct,
    'use_ma_entry': use_ma_entry, 'ma_n': 25, 
    'use_bb_entry': use_bb_entry, 'bb_n': 20, 'bb_sigma': 2.0,
    'use_macd_entry': use_macd_entry, # New
    'use_adx_filter': use_adx_filter, 'adx_thresh': adx_thresh, # New
    'take_profit_pct': take_profit_pct, 'stop_loss_pct': stop_loss_pct,
    'use_rsi_exit': use_rsi_exit, 'rsi_sell_thresh': rsi_sell_thresh
}

# ==========================================
# メイン処理
# ==========================================

if st.button("🚀 分析スタート"):
    
    results = []
    detail_data = {}
    progress_bar = st.progress(0)
    
    for i, ticker in enumerate(selected_tickers):
        name = POPULAR_STOCKS.get(ticker, ticker)
        df, _ = get_stock_data(ticker)
        
        if df is not None:
            df_calc = add_indicators(df, params)
            df_res, log = backtest_strategy(df_calc, params, lot_size)
            
            if not log.empty:
                total_profit = log['損益'].sum()
                wins = len(log[log['損益'] > 0])
                win_rate = (wins / (len(log)/2)) * 100
                results.append({
                    "銘柄名": name, "コード": ticker,
                    "利益": total_profit, "勝率": f"{win_rate:.1f}%", "回数": len(log)//2
                })
            else:
                results.append({"銘柄名": name, "コード": ticker, "利益": 0, "勝率": "-", "回数": 0})
            
            detail_data[ticker] = (df_res, log, name)
        
        progress_bar.progress((i + 1) / len(selected_tickers))
    
    # 結果表示
    st.markdown("### 📊 分析結果ランキング")
    
    tab1, tab2, tab3 = st.tabs(["🏆 収益一覧", "📈 詳細チャート", "📝 取引ログ"])
    
    with tab1:
        if results:
            df_summary = pd.DataFrame(results).sort_values("利益", ascending=False)
            st.dataframe(df_summary.style.format({"利益": "{:,.0f}円"}), use_container_width=True, hide_index=True)
        else:
            st.error("データなし")

    with tab2:
        target = st.selectbox("チャートを表示", df_summary['コード'].tolist(), format_func=lambda x: f"{POPULAR_STOCKS.get(x,x)}")
        if target in detail_data:
            df_res, _, name = detail_data[target]
            
            # 3段構成のチャート (株価 / MACD / ADX & RSI)
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True, 
                row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.05,
                subplot_titles=("株価 & 売買サイン", "MACD", "RSI & ADX")
            )
            
            # 1. 株価
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Close'], name='株価', line=dict(color='gray')), row=1, col=1)
            # VWAP
            if params['use_vwap_entry']:
                fig.add_trace(go.Scatter(x=df_res.index, y=df_res['VWAP'], name='VWAP', line=dict(color='orange', dash='dot')), row=1, col=1)
            # サイン
            buy_pts = df_res[df_res['Buy_Signal'].notna()]
            sell_pts = df_res[df_res['Sell_Signal'].notna()]
            fig.add_trace(go.Scatter(x=buy_pts.index, y=buy_pts['Buy_Signal'], mode='markers', name='買い', marker=dict(symbol='triangle-up', size=12, color='red')), row=1, col=1)
            fig.add_trace(go.Scatter(x=sell_pts.index, y=sell_pts['Sell_Signal'], mode='markers', name='売り', marker=dict(symbol='triangle-down', size=12, color='blue')), row=1, col=1)
            
            # 2. MACD
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['MACD'], name='MACD', line=dict(color='cyan')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['MACD_Signal'], name='Signal', line=dict(color='orange')), row=2, col=1)
            fig.add_bar(x=df_res.index, y=df_res['MACD_Hist'], name='Hist', marker_color='gray', row=2, col=1)
            
            # 3. RSI & ADX
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['ADX'], name='ADX', line=dict(color='green', width=1)), row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="red", row=3, col=1) # RSI底
            fig.add_hline(y=25, line_dash="dash", line_color="green", row=3, col=1) # ADX目安
            
            fig.update_layout(height=800, margin=dict(t=20, b=20, l=10, r=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        target_log = st.selectbox("ログを表示", df_summary['コード'].tolist(), key="log_sel", format_func=lambda x: f"{POPULAR_STOCKS.get(x,x)}")
        if target_log in detail_data:
            _, log, _ = detail_data[target_log]
            if not log.empty:
                log['日付'] = log['日付'].dt.strftime('%Y-%m-%d')
                st.dataframe(log[['日付', '売買', '単価', '損益', '理由']], use_container_width=True)
            else:
                st.info("取引なし")