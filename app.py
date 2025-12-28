import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# --- 設定 ---
st.set_page_config(page_title="Pro株分析AI Ver.2.6", page_icon="📊", layout="wide")

# 人気銘柄辞書
NAME_MAP = {
    "7203.T": "トヨタ自動車", "9984.T": "ソフトバンクG", "8306.T": "三菱UFJ",
    "6920.T": "レーザーテック", "6758.T": "ソニーG", "9983.T": "ファーストリテイリング",
    "8035.T": "東京エレクトロン", "4502.T": "武田薬品", "9432.T": "NTT",
    "7974.T": "任天堂", "6861.T": "キーエンス", "6098.T": "リクルート",
    "4063.T": "信越化学", "6301.T": "コマツ", "8058.T": "三菱商事",
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
    rsi_n = 14
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
    
    if params.get('use_bb_entry') or params.get('use_bb_exit'):
        sma_bb = df['Close'].rolling(window=params['bb_n']).mean()
        std = df['Close'].rolling(window=params['bb_n']).std()
        df['BB_Upper'] = sma_bb + (std * params['bb_sigma'])
        df['BB_Lower'] = sma_bb - (std * params['bb_sigma'])

    # 4. MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # 5. ADX (DMI)
    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()
    df['+DM'] = np.where((high_diff > 0) & (high_diff > -low_diff), high_diff, 0)
    df['-DM'] = np.where((low_diff < 0) & (-low_diff > high_diff), -low_diff, 0)
    
    df['TR'] = pd.concat([
        df['High'] - df['Low'], 
        (df['High'] - df['Close'].shift(1)).abs(), 
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    
    adx_n = 14
    tr_smooth = df['TR'].rolling(window=adx_n).sum()
    plus_dm_smooth = df['+DM'].rolling(window=adx_n).sum()
    minus_dm_smooth = df['-DM'].rolling(window=adx_n).sum()
    
    tr_smooth = tr_smooth.replace(0, np.nan)
    df['+DI'] = 100 * (plus_dm_smooth / tr_smooth)
    df['-DI'] = 100 * (minus_dm_smooth / tr_smooth)
    
    di_sum = df['+DI'] + df['-DI']
    di_sum = di_sum.replace(0, np.nan)
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / di_sum)
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
    rsi_mode = params.get('rsi_mode', '逆張り')
    
    use_vwap = params['use_vwap_entry']
    use_ma = params['use_ma_entry']
    
    use_bb = params['use_bb_entry']
    bb_mode = params.get('bb_mode', '逆張り')
    
    use_macd = params['use_macd_entry']
    use_adx = params['use_adx_filter']
    
    use_rsi_exit = params['use_rsi_exit']
    use_bb_exit = params['use_bb_exit']
    
    take_profit_pct = params['take_profit_pct'] / 100
    stop_loss_pct = params['stop_loss_pct'] / 100
    
    for i in range(len(df)):
        price = df['Close'].iloc[i]
        date = df.index[i]
        
        rsi = df['RSI'].iloc[i] if 'RSI' in df.columns else np.nan
        vwap = df['VWAP'].iloc[i] if 'VWAP' in df.columns else np.nan
        sma = df['SMA'].iloc[i] if 'SMA' in df.columns else np.nan
        bb_lower = df['BB_Lower'].iloc[i] if 'BB_Lower' in df.columns else np.nan
        bb_upper = df['BB_Upper'].iloc[i] if 'BB_Upper' in df.columns else np.nan
        
        macd = df['MACD'].iloc[i] if 'MACD' in df.columns else np.nan
        macd_sig = df['MACD_Signal'].iloc[i] if 'MACD_Signal' in df.columns else np.nan
        adx = df['ADX'].iloc[i] if 'ADX' in df.columns else np.nan

        if pd.isna(rsi) or (use_macd and pd.isna(macd)) or (use_adx and pd.isna(adx)):
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)
            continue

        # --- 買い判定 (AND条件) ---
        buy_condition = True
        
        # 1. RSI (順張り/逆張り 切り替え)
        if use_rsi:
            if '逆張り' in rsi_mode: # 以下なら買い
                if not (rsi <= params['rsi_buy_thresh']): buy_condition = False
            else: # 順張り: 以上なら買い
                if not (rsi >= params['rsi_buy_thresh']): buy_condition = False
        
        # 2. VWAP
        if use_vwap:
            lower_limit = vwap * (1 - params['vwap_low_pct'] / 100)
            upper_limit = vwap * (1 + params['vwap_high_pct'] / 100)
            if not (lower_limit <= price <= upper_limit): buy_condition = False
        
        # 3. MACD
        if use_macd and not (macd > macd_sig): buy_condition = False
            
        # 4. ADX
        if use_adx and not (adx >= params['adx_thresh']): buy_condition = False

        # 5. MA
        if use_ma and not (price > sma): buy_condition = False

        # 6. BB (順張り/逆張り 切り替え)
        if use_bb:
            if '逆張り' in bb_mode: # -2σ割れで買い
                if not (price <= bb_lower): buy_condition = False
            else: # 順張り: +2σ越えで買い
                if not (price >= bb_upper): buy_condition = False
        
        # 条件未選択なら買わない
        if not any([use_rsi, use_vwap, use_ma, use_bb, use_macd]): buy_condition = False

        # --- 売り判定 (OR条件) ---
        sell_condition = False
        sell_reason = ""
        
        if position == 1:
            # 1. 損益
            pnl_pct = (price - entry_price) / entry_price
            if pnl_pct >= take_profit_pct:
                sell_condition = True; sell_reason = "利確"
            elif pnl_pct <= -stop_loss_pct:
                sell_condition = True; sell_reason = "損切"
            
            # 2. テクニカル
            if not sell_condition:
                if use_rsi_exit and rsi >= params['rsi_sell_thresh']:
                    sell_condition = True; sell_reason = f"RSI高({int(rsi)})"
                
                if use_bb_exit and price >= bb_upper:
                    sell_condition = True; sell_reason = "BB+2σ"

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
st.title("⚡ Pro株分析AI Ver.2.6")
st.caption("順張り・逆張りの両方に対応した高度シグナル分析ツール")

# --- サイドバー ---
st.sidebar.header("🔍 分析対象の設定")

st.sidebar.caption("銘柄コードを入力してください (複数可)")
tickers_input = st.sidebar.text_area(
    "コード入力 (改行 または カンマ区切り)",
    value="",
    placeholder="例:\n7203\n8306\n9984",
    height=100
)

selected_tickers = []
if tickers_input:
    raw_tickers = re.split(r'[,\n\s]+', tickers_input)
    for t in raw_tickers:
        t = t.strip()
        if t:
            if t.isdigit(): t = t + ".T"
            selected_tickers.append(t)

st.sidebar.markdown("[🔎 銘柄コードを検索する (Yahoo!ファイナンス)](https://finance.yahoo.co.jp/)")
st.sidebar.markdown("---")

# 条件設定
with st.sidebar.expander("⚙️ 条件設定", expanded=True):
    
    # === 買い条件 ===
    st.subheader("🟢 買い条件")
    st.caption("※チェックした全条件を満たす時に買います")
    
    # MACD
    use_macd_entry = st.checkbox("MACD (上昇トレンド有無)", value=False)
    
    # ADX
    use_adx_filter = st.checkbox("ADX (トレンド発生度合)", value=False)
    adx_thresh = 25
    if use_adx_filter:
        adx_thresh = st.slider("ADX値 以上", 10, 50, 25)

    # RSI (ここを修正：ラジオボタンで見やすく)
    use_rsi_entry = st.checkbox("RSI (売られすぎ度合)", value=True)
    rsi_mode = '逆張り'
    rsi_buy_thresh = 30
    if use_rsi_entry:
        # ドロップダウンからラジオボタンに変更し、横並び配置の制限を解除
        rsi_mode = st.radio("判定モード", ["逆張り (〇〇以下で買い)", "順張り (〇〇以上で買い)"], horizontal=False)
        
        if "逆張り" in rsi_mode:
            rsi_buy_thresh = st.number_input("RSI値 以下なら買い", value=30, step=1)
        else:
            rsi_buy_thresh = st.number_input("RSI値 以上なら買い", value=50, step=1)

    # VWAP
    use_vwap_entry = st.checkbox("VWAP (平均取引価格)", value=False)
    vwap_high_pct = 1.0; vwap_low_pct = 3.0
    if use_vwap_entry:
        col_v1, col_v2 = st.columns(2)
        with col_v1: vwap_high_pct = st.number_input("上 (+%)", value=1.0)
        with col_v2: vwap_low_pct = st.number_input("下 (-%)", value=3.0)
            
    # MA
    use_ma_entry = st.checkbox("移動平均線 (ゴールデンクロス)", value=False)
    
    # BB
    use_bb_entry = st.checkbox("ボリンジャーバンド (反発/ブレイク)", value=False)
    bb_mode = '逆張り'
    if use_bb_entry:
        bb_mode = st.radio("BB判定", ["逆張り (-2σ割れで買い)", "順張り (+2σブレイクで買い)"], horizontal=False)

    st.markdown("---")
    
    # === 売り条件 ===
    st.subheader("🔴 売り条件")
    
    # 損益
    col_p, col_l = st.columns(2)
    with col_p: take_profit_pct = st.number_input("利確 (%)", value=5.0, step=0.5)
    with col_l: stop_loss_pct = st.number_input("損切 (%)", value=3.0, step=0.5)
        
    # 指標売り
    use_rsi_exit = st.checkbox("RSI (買われすぎ度合)", value=False)
    rsi_sell_thresh = 70
    if use_rsi_exit:
        rsi_sell_thresh = st.slider("売りRSI値 以上", 50, 95, 75)
        
    use_bb_exit = st.checkbox("ボリンジャーバンド (+2σ越え)", value=False)

    st.markdown("---")
    lot_size = st.number_input("1回の株数", value=100)

# パラメータ
params = {
    'use_rsi_entry': use_rsi_entry, 'rsi_mode': rsi_mode, 'rsi_buy_thresh': rsi_buy_thresh,
    'use_vwap_entry': use_vwap_entry, 'vwap_high_pct': vwap_high_pct, 'vwap_low_pct': vwap_low_pct,
    'use_ma_entry': use_ma_entry, 'ma_n': 25, 
    'use_bb_entry': use_bb_entry, 'bb_mode': bb_mode, 'bb_n': 20, 'bb_sigma': 2.0,
    'use_macd_entry': use_macd_entry,
    'use_adx_filter': use_adx_filter, 'adx_thresh': adx_thresh,
    'take_profit_pct': take_profit_pct, 'stop_loss_pct': stop_loss_pct,
    'use_rsi_exit': use_rsi_exit, 'rsi_sell_thresh': rsi_sell_thresh,
    'use_bb_exit': use_bb_exit
}

# ==========================================
# メイン処理
# ==========================================

if st.button("🚀 分析スタート"):
    
    results = []
    detail_data = {}
    progress_bar = st.progress(0)
    
    if not selected_tickers:
        st.error("銘柄コードを入力してください")
    else:
        for i, ticker in enumerate(selected_tickers):
            name = NAME_MAP.get(ticker, ticker)
            df = get_stock_data(ticker)
            
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
                st.error("データなし（コードが正しいか確認してください）")

        with tab2:
            if results:
                target_options = df_summary['コード'].tolist()
                target = st.selectbox("チャートを表示", target_options, format_func=lambda x: f"{NAME_MAP.get(x,x)}")
                
                if target in detail_data:
                    df_res, _, name = detail_data[target]
                    
                    fig = make_subplots(
                        rows=3, cols=1, shared_xaxes=True, 
                        row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.05,
                        subplot_titles=("株価 & 売買サイン", "MACD", "RSI & ADX")
                    )
                    
                    # 1. 株価
                    fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Close'], name='株価', line=dict(color='gray')), row=1, col=1)
                    if params['use_vwap_entry']:
                        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['VWAP'], name='VWAP', line=dict(color='orange', dash='dot')), row=1, col=1)
                    if params['use_bb_entry'] or params['use_bb_exit']:
                         fig.add_trace(go.Scatter(x=df_res.index, y=df_res['BB_Upper'], name='+2σ', line=dict(color='green', width=1, dash='dot')), row=1, col=1)
                         fig.add_trace(go.Scatter(x=df_res.index, y=df_res['BB_Lower'], name='-2σ', line=dict(color='red', width=1, dash='dot')), row=1, col=1)

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
                    fig.add_hline(y=30, line_dash="dash", line_color="red", row=3, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="blue", row=3, col=1)
                    fig.add_hline(y=25, line_dash="dash", line_color="green", row=3, col=1)
                    
                    fig.update_layout(height=800, margin=dict(t=20, b=20, l=10, r=10), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

        with tab3:
            if results:
                target_log = st.selectbox("ログを表示", df_summary['コード'].tolist(), key="log_sel", format_func=lambda x: f"{NAME_MAP.get(x,x)}")
                if target_log in detail_data:
                    _, log, _ = detail_data[target_log]
                    if not log.empty:
                        log['日付'] = log['日付'].dt.strftime('%Y-%m-%d')
                        st.dataframe(log[['日付', '売買', '単価', '損益', '理由']], use_container_width=True)
                    else:
                        st.info("取引なし")