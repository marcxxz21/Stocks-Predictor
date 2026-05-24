import gradio as gr
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import joblib
import json
import base64
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# ========================================================
# 1. PYTORCH MODEL
# =========================================================
class StockSageModel(nn.Module):
    def __init__(self, n_features):
        super(StockSageModel, self).__init__()
        self.cnn = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.cnn(x))
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        x = self.relu(self.fc1(out[:, -1, :]))
        return self.sigmoid(self.fc2(x))

# =========================================================
# 2. LOAD MODEL ASSETS
# =========================================================
def load_assets():
    try:
        with open('model_artifacts/metadata.json', 'r') as f:
            meta = json.load(f)
        scaler = joblib.load('model_artifacts/scaler.pkl')
        model = StockSageModel(len(meta['feature_cols']))
        model.load_state_dict(
            torch.load('model_artifacts/best_model.pth', map_location=torch.device('cpu'))
        )
        model.eval()
        return model, scaler, meta
    except Exception as e:
        return None, None, None

# =========================================================
# 3. FEATURE ENGINEERING
# =========================================================
def get_features(df):
    df = df.copy()
    df['Returns'] = df['Close'].pct_change()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    df['Vol_Rel'] = df['Volume'] / df['Volume'].rolling(20).mean()
    return df.replace([np.inf, -np.inf], np.nan).dropna()


HORIZON_DAYS = {
    "1 Week": 5,
    "1 Month": 21,
    "3 Months": 63,
    "6 Months": 126,
}


def forecast_future_price(df_feat, timeframe):
    """Train a ticker-local forward-return model and produce a price forecast."""
    horizon_days = HORIZON_DAYS.get(timeframe, 21)
    feature_cols = ['Close', 'Returns', 'MA20', 'MA50', 'RSI', 'Vol_Rel']

    model_df = df_feat.copy()
    model_df['Forward_Log_Return'] = np.log(model_df['Close'].shift(-horizon_days) / model_df['Close'])
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(model_df) < max(90, horizon_days * 3):
        raise ValueError("Not enough historical data to produce a price forecast for this horizon.")

    X = model_df[feature_cols]
    y = model_df['Forward_Log_Return']
    split_idx = max(int(len(model_df) * 0.8), len(model_df) - max(30, horizon_days))

    X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]

    regressor = Pipeline([
        ('scaler', RobustScaler()),
        ('model', RidgeCV(alphas=np.logspace(-4, 4, 25)))
    ])
    regressor.fit(X_train, y_train)

    valid_pred = regressor.predict(X_valid)
    valid_actual_prices = model_df['Close'].iloc[split_idx:].to_numpy() * np.exp(y_valid.to_numpy())
    valid_pred_prices = model_df['Close'].iloc[split_idx:].to_numpy() * np.exp(valid_pred)
    mape = float(mean_absolute_percentage_error(valid_actual_prices, valid_pred_prices))
    residual_std = float(np.std(y_valid.to_numpy() - valid_pred))

    latest_features = df_feat[feature_cols].tail(1)
    predicted_log_return = float(regressor.predict(latest_features)[0])
    current_price = float(df_feat['Close'].iloc[-1])
    forecast_price = current_price * float(np.exp(predicted_log_return))

    # 80% empirical interval from validation residuals in log-return space.
    interval_width = max(1.28 * residual_std, 0.01)
    lower_price = current_price * float(np.exp(predicted_log_return - interval_width))
    upper_price = current_price * float(np.exp(predicted_log_return + interval_width))

    forecast_dates = pd.bdate_range(df_feat.index[-1], periods=horizon_days + 1)[1:]
    forecast_path = np.geomspace(current_price, forecast_price, num=horizon_days + 1)[1:]
    lower_path = np.geomspace(current_price, lower_price, num=horizon_days + 1)[1:]
    upper_path = np.geomspace(current_price, upper_price, num=horizon_days + 1)[1:]

    return {
        "horizon_days": horizon_days,
        "forecast_price": forecast_price,
        "lower_price": lower_price,
        "upper_price": upper_price,
        "expected_return_pct": (forecast_price / current_price - 1) * 100,
        "validation_mape": mape * 100,
        "forecast_dates": forecast_dates,
        "forecast_path": forecast_path,
        "lower_path": lower_path,
        "upper_path": upper_path,
    }

# =========================================================
# 4. LOAD LOGO
# =========================================================
try:
    with open('sage.png', 'rb') as f:
        LOGO_B64 = base64.b64encode(f.read()).decode()
except Exception:
    LOGO_B64 = ""

# =========================================================
# 5. MAIN ANALYSIS FUNCTION
# =========================================================
def run_full_analysis(ticker, amount, target_pct, loss_pct, timeframe, progress=gr.Progress()):

    progress(0.1, desc="Loading AI Quant Engine...")
    model, scaler, meta = load_assets()

    if model is None:
        empty = "<div class='panel-empty'>Model loading failed. Ensure 'model_artifacts' exists.</div>"
        return (empty, empty, go.Figure(), empty, empty, empty, empty, empty)

    progress(0.25, desc=f"Retrieving Market Data: {ticker}")
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(period="5y")

    if df.empty:
        empty = "<div class='panel-empty'>Invalid ticker or no data.</div>"
        return (empty, empty, go.Figure(), empty, empty, empty, empty, empty)

    df_feat = get_features(df.copy())
    try:
        price_forecast = forecast_future_price(df_feat, timeframe)
        forecast_error_html = ""
    except Exception as e:
        price_forecast = None
        forecast_error_html = f"<div class='forecast-note'>Price forecast unavailable: {str(e)}</div>"

    # ── Model prediction ──────────────────────────────────
    progress(0.45, desc="Running Deep Learning Forecast...")
    last_data = df_feat[meta['feature_cols']].tail(meta['seq_len'])
    last_scaled = scaler.transform(last_data)
    input_tensor = torch.FloatTensor(last_scaled).unsqueeze(0)

    with torch.no_grad():
        prob = model(input_tensor).item()

    # ── Compute metrics ───────────────────────────────────
    current_price = float(df['Close'].iloc[-1])
    prev_close = float(df['Close'].iloc[-2]) if len(df) >= 2 else current_price
    price_change = current_price - prev_close
    price_change_pct = (price_change / prev_close) * 100
    price_up = price_change >= 0
    price_arrow = "+" if price_up else ""

    take_profit_price = current_price * (1 + target_pct / 100)
    stop_loss_price   = current_price * (1 - loss_pct / 100)
    potential_profit   = amount * (target_pct / 100)
    potential_loss     = amount * (loss_pct / 100)
    risk_reward = potential_profit / potential_loss if potential_loss > 0 else 0

    trend       = "BULLISH" if prob > 0.5 else "BEARISH"
    trend_color = "#00ff9d" if prob > 0.5 else "#ff4d6d"
    confidence  = prob if prob > 0.5 else (1 - prob)

    current_rsi  = df_feat['RSI'].iloc[-1]
    current_ma50 = df_feat['MA50'].iloc[-1]
    current_vol  = df_feat['Volume'].iloc[-1]
    avg_vol      = df_feat['Volume'].rolling(20).mean().iloc[-1]
    vol_alert    = "VOLUME SPIKE" if current_vol > avg_vol * 1.5 else "NORMAL"
    ann_volatility = df_feat['Returns'].tail(20).std() * np.sqrt(252) * 100

    momentum_state = "OVERBOUGHT" if current_rsi > 70 else "OVERSOLD" if current_rsi < 30 else "NEUTRAL"
    trend_alignment = "Bullish" if current_price > current_ma50 else "Bearish"

    if risk_reward >= 2 and prob > 0.6:
        action_plan = "STRONG BUY"; action_color = "#00ff9d"
    elif risk_reward >= 1.5 and prob > 0.5:
        action_plan = "HOLD / CAUTION"; action_color = "#fbbf24"
    else:
        action_plan = "AVOID / SELL"; action_color = "#ff4d6d"

    # ── Helper colors ─────────────────────────────────────
    rsi_color = "#ff4d6d" if current_rsi > 70 else "#00ff9d" if current_rsi < 30 else "#fbbf24"
    vol_color = "#ff4d6d" if ann_volatility > 40 else "#fbbf24" if ann_volatility > 20 else "#00ff9d"
    rr_color  = "#00ff9d" if risk_reward >= 2 else "#fbbf24" if risk_reward >= 1 else "#ff4d6d"
    conf_color = "#00ff9d" if confidence > 0.7 else "#fbbf24" if confidence > 0.55 else "#ff4d6d"
    ta_color   = "#00ff9d" if trend_alignment == "Bullish" else "#ff4d6d"

    rsi_pct    = min(max(current_rsi, 0), 100)
    conf_pct   = round(confidence * 100)
    loss_bar   = round(1 / (1 + risk_reward if risk_reward > 0 else 2) * 100)
    gain_bar   = 100 - loss_bar
    if price_forecast:
        predicted_price = price_forecast["forecast_price"]
        predicted_move = price_forecast["expected_return_pct"]
        predicted_color = "#00ff9d" if predicted_move >= 0 else "#ff4d6d"
        forecast_accuracy = max(0, 100 - price_forecast["validation_mape"])
        forecast_horizon_label = f"{timeframe} ({price_forecast['horizon_days']} trading days)"
    else:
        predicted_price = current_price
        predicted_move = 0
        predicted_color = "#94a3b8"
        forecast_accuracy = 0
        forecast_horizon_label = timeframe

    # =====================================================
    # TERMINAL HEADER
    # =====================================================
    price_chg_color = "#00ff9d" if price_up else "#ff4d6d"
    terminal_header_html = f"""
    <div class='price-banner'>
        <div class='pb-price'>${current_price:.2f}</div>
        <div class='pb-change' style='color:{price_chg_color}; background:{price_chg_color}1A;'>
            {price_arrow}{price_change_pct:.2f}%
        </div>
        <div class='pb-signal' style='color:{trend_color}'>
            <span style='font-size: 11px; color:var(--text-muted); font-weight:500; display:block; line-height:1;'>SIGNAL</span>
            {trend}
        </div>
    </div>
    """

    # =====================================================
    # METRIC CARDS
    # =====================================================
    metric_cards = f"""
    <div class='m-grid'>
        <div class='m-card glow-pulse'>
            <div class='m-label'>AI Confidence</div>
            <div class='m-val mono' style='color:{conf_color}'>{confidence:.1%}</div>
        </div>
        <div class='m-card'>
            <div class='m-label'>RSI (14)</div>
            <div class='m-val mono' style='color:{rsi_color}'>{current_rsi:.1f}</div>
        </div>
        <div class='m-card'>
            <div class='m-label'>Ann. Volatility</div>
            <div class='m-val mono' style='color:{vol_color}'>{ann_volatility:.1f}%</div>
        </div>
        <div class='m-card'>
            <div class='m-label'>Risk/Reward</div>
            <div class='m-val mono' style='color:{rr_color}'>1:{risk_reward:.2f}</div>
        </div>
        <div class='m-card forecast-card'>
            <div class='m-label'>Forecast Price</div>
            <div class='m-val mono' style='color:{predicted_color}'>${predicted_price:.2f}</div>
        </div>
        <div class='m-card forecast-card'>
            <div class='m-label'>Expected Move</div>
            <div class='m-val mono' style='color:{predicted_color}'>{predicted_move:+.2f}%</div>
        </div>
    </div>
    """

    # =====================================================
    # CHART
    # =====================================================
    progress(0.65, desc="Building Chart...")
    df_plot = df_feat.tail(90)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.60, 0.20, 0.20])

    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'],
        name="Price", increasing_line_color='#00ff9d', decreasing_line_color='#ff4d6d',
        increasing_fillcolor='rgba(0,255,157,0.6)', decreasing_fillcolor='rgba(255,77,109,0.6)'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], mode='lines', line=dict(color='#a78bfa', width=1.5, dash='dot'), name="MA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'], mode='lines', line=dict(color='#38bdf8', width=2), name="MA50"), row=1, col=1)
    if price_forecast:
        forecast_x = [df_feat.index[-1], *list(price_forecast["forecast_dates"])]
        forecast_y = [current_price, *list(price_forecast["forecast_path"])]
        upper_y = [current_price, *list(price_forecast["upper_path"])]
        lower_y = [current_price, *list(price_forecast["lower_path"])]
        fig.add_trace(go.Scatter(
            x=forecast_x, y=forecast_y, mode='lines', name=f"{timeframe} Forecast",
            line=dict(color=predicted_color, width=2.5, dash='dash')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=forecast_x, y=upper_y, mode='lines', name="Forecast Upper",
            line=dict(color='rgba(148,163,184,0)', width=0), showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=forecast_x, y=lower_y, mode='lines', name="Forecast Range",
            line=dict(color='rgba(148,163,184,0)', width=0), fill='tonexty',
            fillcolor='rgba(148,163,184,0.16)', showlegend=True
        ), row=1, col=1)

    fig.add_hline(y=take_profit_price, line_dash="dash", line_color="rgba(0,255,157,0.6)", annotation_text=f"TP ${take_profit_price:.2f}", annotation_font_color="#00ff9d", row=1, col=1)
    fig.add_hline(y=stop_loss_price, line_dash="dash", line_color="rgba(255,77,109,0.6)", annotation_text=f"SL ${stop_loss_price:.2f}", annotation_font_color="#ff4d6d", row=1, col=1)

    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], mode='lines', line=dict(color='#fbbf24', width=1.5), name="RSI", fill='tozeroy', fillcolor='rgba(251,191,36,0.04)'), row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor='rgba(255,77,109,0.06)', line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor='rgba(0,255,157,0.06)', line_width=0, row=2, col=1)
    fig.add_hline(y=70, line_dash='dot', line_color='rgba(255,77,109,0.3)', row=2, col=1)
    fig.add_hline(y=30, line_dash='dot', line_color='rgba(0,255,157,0.3)', row=2, col=1)

    vol_colors = ['#00ff9d' if r['Close'] >= r['Open'] else '#ff4d6d' for _, r in df_plot.iterrows()]
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], marker_color=vol_colors, name="Volume", opacity=0.7), row=3, col=1)

    fig.update_layout(
        template="plotly_dark", height=550, hovermode="x unified", dragmode="pan",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="'IBM Plex Mono', monospace", color="#94a3b8", size=11),
        margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(color="#64748b", size=9))
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False, tickfont=dict(color="#64748b", size=9))
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    # =====================================================
    # STRATEGY & DEEP ANALYSIS PANELS (Side by Side)
    # =====================================================
    strategy_html = f"""
    <div class='signal-panel flex-col'>
        <div class='qp-section-title'>Execution Strategy</div>
        <div class='sp-hero'>
            <div class='sp-action' style='color:{action_color};border-color:{action_color};box-shadow:0 0 16px {action_color}22'>{action_plan}</div>
            <div class='sp-conf'>Model Confidence: <span class='mono' style='color:{conf_color}'>{confidence:.1%}</span></div>
            <div class='conf-bar'><div class='conf-fill' style='width:{conf_pct}%;background:{conf_color}'></div></div>
        </div>
        <div class='sp-grid'>
            <div class='sp-item'><div class='sp-lbl'>Momentum</div><div class='sp-val' style='color:{rsi_color}'>{momentum_state}</div></div>
            <div class='sp-item'><div class='sp-lbl'>Volume</div><div class='sp-val'>{"⚠️ " if vol_alert == "VOLUME SPIKE" else ""}{vol_alert}</div></div>
            <div class='sp-item'><div class='sp-lbl'>Trend (MA50)</div><div class='sp-val' style='color:{ta_color}'>{trend_alignment.upper()}</div></div>
            <div class='sp-item'><div class='sp-lbl'>Forecast Price</div><div class='sp-val mono' style='color:{predicted_color}'>${predicted_price:.2f}</div></div>
        </div>
    </div>
    """

    rsi_bg = "rgba(255,77,109,0.12)" if current_rsi > 70 else "rgba(0,255,157,0.12)" if current_rsi < 30 else "rgba(251,191,36,0.12)"
    forecast_range_rows = ""
    if price_forecast:
        forecast_range_rows = f"""
            <div class='pl-row'><span style='color:{predicted_color}'>{forecast_horizon_label} Forecast</span><span class='mono' style='color:{predicted_color}'>${predicted_price:.2f} <small>{predicted_move:+.2f}%</small></span></div>
            <div class='pl-row'><span>80% Forecast Range</span><span class='mono'>${price_forecast["lower_price"]:.2f} - ${price_forecast["upper_price"]:.2f}</span></div><div class='pl-sep'></div>
        """
    basic_analysis_html = f"""
    <div class='quant-panel flex-col'>
        <div class='qp-section-title'>Quantitative Trade Metrics</div>
        <div class='ind-item'>
            <div class='ind-hdr'>
                <span class='ind-name'>RSI (14) Indicator</span>
                <span class='ind-badge' style='color:{rsi_color};background:{rsi_bg}'>{momentum_state}</span>
                <span class='ind-val mono' style='color:{rsi_color}'>{rsi_pct:.1f}</span>
            </div>
            <div class='rsi-track'>
                <div class='rsi-zone os'></div><div class='rsi-zone mid'></div><div class='rsi-zone ob'></div>
                <div class='rsi-needle' style='left:calc({rsi_pct:.1f}% - 5px)'><div class='rsi-dot' style='background:{rsi_color};box-shadow:0 0 6px {rsi_color}'></div></div>
            </div>
            <div class='rsi-labels mono'><span>0</span><span>30</span><span>70</span><span>100</span></div>
        </div>
        <div class='price-levels'>
            <div class='pl-row'><span>Current Entry</span><span class='mono'>${current_price:.2f}</span></div><div class='pl-sep'></div>
            {forecast_range_rows}
            <div class='pl-row'><span style='color:#00ff9d'>Take Profit Goal</span><span class='mono' style='color:#00ff9d'>${take_profit_price:.2f} <small>+{target_pct:.0f}%</small></span></div>
            <div class='pl-row'><span style='color:#ff4d6d'>Stop Loss Limit</span><span class='mono' style='color:#ff4d6d'>${stop_loss_price:.2f} <small>-{loss_pct:.0f}%</small></span></div>
        </div>
        {forecast_error_html}
        <div class='rr-wrap' style='margin-top:auto;'>
            <div class='rr-hdr'><span>Capital At Risk vs Potential Gain</span><span class='mono' style='color:{rr_color}'>1 : {risk_reward:.2f}</span></div>
            <div class='rr-bar'>
                <div class='rr-risk' style='width:{loss_bar}%'><small>-${potential_loss:,.0f}</small></div>
                <div class='rr-gain' style='width:{gain_bar}%'><small>+${potential_profit:,.0f}</small></div>
            </div>
        </div>
    </div>
    """

    # =====================================================
    # IN-DEPTH QUANT ANALYSIS
    # =====================================================
    trend_str = "🟢 BULLISH EXPECTED" if prob > 0.5 else "🔴 BEARISH EXPECTED"
    if current_rsi > 70: rsi_str = "Overbought (High risk of pullback)"
    elif current_rsi < 30: rsi_str = "Oversold (Potential bounce up)"
    else: rsi_str = "Neutral Activity"

    if action_plan == "STRONG BUY":
        rec_title, rec_bg, rec_border = "🟢 STRONG BUY SIGNAL", "rgba(0, 255, 157, 0.1)", "#00ff9d"
        rec_text = "The AI predicts a Bullish trend, and your risk-to-reward ratio is optimal. Favorable conditions to consider entering this trade."
    elif action_plan == "HOLD / CAUTION":
        rec_title, rec_bg, rec_border = "🟡 CAUTION / MODERATE SIGNAL", "rgba(251, 191, 36, 0.1)", "#fbbf24"
        rec_text = "The AI is leaning Bullish, but the potential reward isn't quite high enough compared to the risk. Consider aiming for a higher target."
    else:
        rec_title, rec_bg, rec_border = "🔴 DANGER / DO NOT BUY", "rgba(255, 77, 109, 0.1)", "#ff4d6d"
        rec_text = "The AI predicts a Bearish trend, or the math is too risky based on your parameters. Capital preservation recommended."

    indepth_html_content = f"""
    <div class="insight-board">
        <div class="insight-header">
            <h3 style="margin:0; font-size:18px; color:var(--text-main);">🧠 AI Quant Deep-Dive Breakdown</h3>
            <span class="insight-badge" style="color:{trend_color}; border-color:{trend_color};">{trend_str}</span>
        </div>
        
        <div class="insight-grid">
            <div class="ic-card">
                <div class="ic-icon">🎯</div>
                <div class="ic-info"><span class="ic-label">AI Confidence</span><span class="ic-val" style="color:{conf_color}">{confidence:.1%}</span></div>
            </div>
            <div class="ic-card">
                <div class="ic-icon">⚡</div>
                <div class="ic-info"><span class="ic-label">Expected Volatility</span><span class="ic-val">{ann_volatility:.1f}%</span></div>
            </div>
            <div class="ic-card">
                <div class="ic-icon">📊</div>
                <div class="ic-info"><span class="ic-label">Momentum (RSI)</span><span class="ic-val" style="color:{rsi_color}">{rsi_str}</span></div>
            </div>
            <div class="ic-card">
                <div class="ic-icon">⚖️</div>
                <div class="ic-info"><span class="ic-label">Risk/Reward Setup</span><span class="ic-val" style="color:{rr_color}">1 : {risk_reward:.2f}</span></div>
            </div>
            <div class="ic-card">
                <div class="ic-icon">📈</div>
                <div class="ic-info"><span class="ic-label">Forecast Price</span><span class="ic-val" style="color:{predicted_color}">${predicted_price:.2f} ({predicted_move:+.2f}%)</span></div>
            </div>
            <div class="ic-card">
                <div class="ic-icon">🧪</div>
                <div class="ic-info"><span class="ic-label">Backtest Accuracy</span><span class="ic-val">{forecast_accuracy:.1f}% price fit</span></div>
            </div>
        </div>
        <div class="insight-footer" style="background:{rec_bg}; border: 1px solid {rec_border};">
            <h4 style="color:{rec_border}; margin:0 0 6px 0; font-size: 16px;">{rec_title}</h4>
            <p style="margin:0; font-size: 14px; color: var(--text-main); line-height: 1.6;">{rec_text}</p>
        </div>
    </div>
    """

# =====================================================
    # MACRO INTEL (Rich Media News Grid)
    # =====================================================
    progress(0.8, desc="Retrieving Macro & Corporate Data...")
    ticker_obj = yf.Ticker(ticker)
    news_items = ticker_obj.news[:12] if ticker_obj.news else []
    
    if news_items:
        news_html = "<div class='news-grid'>"
        for item in news_items:
            content = item.get('content', item)
            
            # --- FIXED: Bulletproof Link Extraction ---
            click_data = content.get('clickThroughUrl')
            if click_data and isinstance(click_data, dict):
                link = click_data.get('url', content.get('link', '#'))
            else:
                link = content.get('link', '#')
                
            # --- FIXED: Bulletproof Publisher Extraction ---
            provider_data = content.get('provider')
            if provider_data and isinstance(provider_data, dict):
                publisher = provider_data.get('displayName', content.get('publisher', 'Unknown'))
            else:
                publisher = content.get('publisher', 'Unknown')
                
            title = content.get('title', 'No Title')
            
            # --- FIXED: Bulletproof Image Extraction ---
            img_url = "https://images.unsplash.com/photo-1590283603385-18ff3827ec00?auto=format&fit=crop&w=600&q=80"
            try:
                thumb_data = content.get('thumbnail')
                if thumb_data and isinstance(thumb_data, dict):
                    res = thumb_data.get('resolutions', [])
                    if res and isinstance(res, list) and len(res) > 0:
                        img_url = res[0].get('url', img_url)
            except Exception:
                pass

            news_html += f"""
            <a href="{link}" target="_blank" class="news-card">
                <div class="news-img" style="background-image: url('{img_url}');"></div>
                <div class="news-content">
                    <div class="news-src">{publisher}</div>
                    <div class="news-title">{title}</div>
                    <div class="news-read">Read Article →</div>
                </div>
            </a>
            """
        news_html += "</div>"
    else:
        news_html = "<div class='panel-empty'>No recent market catalysts available.</div>"
    # =====================================================
    # COMPANY PROFILE
    # =====================================================
    info = ticker_obj.info
    mcap = info.get('marketCap', 0)
    mcap_str = f"${mcap/1e12:.2f}T" if mcap >= 1e12 else f"${mcap/1e9:.2f}B" if mcap >= 1e9 else f"${mcap/1e6:.2f}M"
    
    fundamentals_html = f"""
    <div class='prof-container'>
        <div class='prof-header'>
            <div class='prof-title'>
                <h1 style='margin:0; font-size: 32px; color: var(--text-main);'>{info.get('shortName', ticker)}</h1>
                <span class='ticker-pill'>{ticker}</span>
                <span class='sector-pill'>{info.get('sector', 'Unknown Sector')}</span>
            </div>
            <p class='prof-desc'>{info.get('longBusinessSummary', 'No corporate overview available.')}</p>
        </div>
        
        <div class='prof-metrics-grid'>
            <div class='pm-group'>
                <div class='pm-group-title'>Valuation & Core Stats</div>
                <div class='pm-item'><span>Market Cap</span><span class='mono'>{mcap_str}</span></div>
                <div class='pm-item'><span>Trailing P/E</span><span class='mono'>{info.get('trailingPE', 'N/A')}</span></div>
                <div class='pm-item'><span>Forward P/E</span><span class='mono'>{info.get('forwardPE', 'N/A')}</span></div>
                <div class='pm-item'><span>Price to Book</span><span class='mono'>{info.get('priceToBook', 'N/A')}</span></div>
            </div>
            <div class='pm-group'>
                <div class='pm-group-title'>Profitability & Margins</div>
                <div class='pm-item'><span>Profit Margin</span><span class='mono'>{info.get('profitMargins', 0)*100:.2f}%</span></div>
                <div class='pm-item'><span>Operating Margin</span><span class='mono'>{info.get('operatingMargins', 0)*100:.2f}%</span></div>
                <div class='pm-item'><span>Return on Equity</span><span class='mono'>{info.get('returnOnEquity', 0)*100:.2f}%</span></div>
                <div class='pm-item'><span>Dividend Yield</span><span class='mono'>{info.get('dividendYield', 0)*100:.2f}%</span></div>
            </div>
            <div class='pm-group'>
                <div class='pm-group-title'>Trading & Volatility</div>
                <div class='pm-item'><span>Beta (1Y)</span><span class='mono'>{info.get('beta', 'N/A')}</span></div>
                <div class='pm-item'><span>52 Week High</span><span class='mono'>${info.get('fiftyTwoWeekHigh', 0):.2f}</span></div>
                <div class='pm-item'><span>52 Week Low</span><span class='mono'>${info.get('fiftyTwoWeekLow', 0):.2f}</span></div>
                <div class='pm-item'><span>Avg Volume</span><span class='mono'>{info.get('averageVolume', 0):,}</span></div>
            </div>
        </div>
    </div>
    """

    progress(1.0, desc="Analysis Complete")
    return (terminal_header_html, metric_cards, fig, strategy_html, basic_analysis_html, indepth_html_content, news_html, fundamentals_html)


# =========================================================
# 6. PROFESSIONAL UI DESIGN THEME & CUSTOM CSS
# =========================================================
pro_theme = gr.themes.Base(
    primary_hue="emerald",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=[gr.themes.GoogleFont('IBM Plex Mono'), 'ui-monospace', 'Consolas', 'monospace'],
).set(
    body_background_fill='#0B0F19', body_text_color='#F8FAFC',
    background_fill_primary='#131A2A', background_fill_secondary='#0B0F19', 
    block_background_fill='#131A2A', block_border_width='1px',
    block_border_color='#2A3441', block_radius='12px',
    input_background_fill='#0B0F19', input_border_color='#2A3441',
    input_border_color_focus='#00ff9d', input_border_width='1px',
    input_radius='8px', input_padding='12px',
    button_primary_background_fill='#00ff9d', button_primary_background_fill_hover='#00cc7d',
    button_primary_text_color='#0B0F19', button_primary_border_color='#00ff9d',
    button_secondary_background_fill='#1E293B', button_secondary_text_color='#F8FAFC',
    border_color_primary='#2A3441', panel_border_color='#2A3441',
)

custom_css = """
:root {
    --text-main: #F8FAFC; --text-muted: #94A3B8; --border: #2A3441;
    --panel-bg: #131A2A; --green: #00ff9d; --red: #ff4d6d;
}
/* FORCE TRUE 100% WIDTH - KILL GRADIO CONSTRAINTS */
body, .gradio-container { max-width: 100% !important; width: 100% !important; padding: 0 !important; margin: 0 !important; overflow-x: hidden; }
footer { display: none !important; }
.mono { font-family: 'IBM Plex Mono', monospace !important; }
/* MAIN CONTENT WRAPPER */
#main-layout { padding: 0 40px 40px 40px; max-width: 2400px; margin: 0 auto; box-sizing: border-box; }
/* MODERN SAAS HEADER - EXACT SCREENSHOT MATCH */
.pro-header { display: flex; justify-content: space-between; align-items: center; padding: 20px 40px; background: #0B0F19; border-bottom: 1px solid var(--border); width: 100%; box-sizing: border-box; margin-bottom: 24px; }
/* Left Side Header */
.header-left { display: flex; align-items: center; gap: 20px; }
.header-logo-img { height: 60px; width: auto; object-fit: contain; filter: drop-shadow(0 0 10px rgba(0, 255, 157, 0.2)); }
.header-divider { width: 1px; height: 50px; background-color: rgba(255,255,255,0.15); }
.header-title-block { display: flex; flex-direction: column; justify-content: center; }
.header-title-top { font-size: 26px; font-weight: 800; color: #ffffff; line-height: 0.95; letter-spacing: 0.5px; }
.header-title-bottom { font-size: 26px; font-weight: 800; color: var(--green); line-height: 0.95; letter-spacing: 0.5px; }
.header-subtitle { font-size: 13px; color: var(--text-muted); margin-top: 4px; font-weight: 500; }
/* Right Side Header */
.header-right { display: flex; flex-direction: column; align-items: flex-end; gap: 6px; }
.header-nav { display: flex; gap: 24px; font-size: 16px; margin-bottom: 2px; }
.header-nav span.muted { color: var(--text-muted); font-weight: 500; }
.header-nav span.active { color: var(--text-main); font-weight: 600; }
.header-signal-text { font-size: 13px; color: var(--text-muted); }
.header-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 99px; border: 1px solid rgba(0, 255, 157, 0.4); color: var(--green); background: rgba(0, 255, 157, 0.05); font-size: 12px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
/* ------------------------------------- */
.sidebar-title { color: var(--text-main); font-size: 13px; font-weight: 800; margin-bottom: 12px; letter-spacing: 1px; text-transform: uppercase; }
.panel-empty { color: var(--text-muted); font-size: 14px; text-align: center; padding: 60px 20px; border: 1px dashed var(--border); border-radius: 12px; background: rgba(255,255,255,0.01); }
/* TERMINAL METRICS */
.price-banner { display: flex; align-items: center; gap: 16px; padding: 20px 24px; border-radius: 12px; background: rgba(0,0,0,0.2); margin-bottom: 16px; border: 1px solid var(--border); }
.pb-price { font-family: 'IBM Plex Mono', monospace; font-size: 40px; font-weight: 600; color: var(--text-main); line-height: 1; }
.pb-change { font-family: 'IBM Plex Mono', monospace; font-size: 14px; font-weight: 600; padding: 4px 10px; border-radius: 6px; }
.pb-signal { font-size: 24px; font-weight: 800; letter-spacing: 1px; margin-left: auto; text-align: right; }
.m-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-bottom: 16px; }
.m-card { background: rgba(0,0,0,0.2); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }
.forecast-card { border-color: rgba(56,189,248,0.25); background: rgba(56,189,248,0.04); }
.forecast-note { color: var(--text-muted); font-size: 12px; padding: 10px 14px; border: 1px dashed var(--border); border-radius: 8px; margin-bottom: 16px; }
.m-label { color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; font-weight: 600; }
.m-val { font-size: 24px; font-weight: 600; color: var(--text-main); }
.glow-pulse { animation: gPulse 2.5s infinite; border-color: rgba(0,255,157,0.3); background: rgba(0,255,157,0.03); }
@keyframes gPulse { 0% { box-shadow: 0 0 0 rgba(0,255,157,0); } 50% { box-shadow: 0 0 15px rgba(0,255,157,0.1); } 100% { box-shadow: 0 0 0 rgba(0,255,157,0); } }
/* SIDE-BY-SIDE PANELS */
.analysis-container { display: flex; gap: 16px; margin-top: 16px; align-items: stretch; }
.signal-panel, .quant-panel { background: rgba(0,0,0,0.15); padding: 24px; border-radius: 12px; border: 1px solid var(--border); width: 100%; box-sizing: border-box; }
.flex-col { display: flex; flex-direction: column; justify-content: flex-start; height: 100%; }
.qp-section-title { font-size: 12px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
/* STRATEGY */
.sp-action { text-align: center; padding: 16px; border: 1px solid; border-radius: 8px; font-size: 18px; font-weight: 800; letter-spacing: 2px; margin-bottom: 16px; background: rgba(0,0,0,0.3); }
.sp-conf { font-size: 13px; color: var(--text-muted); font-weight: 500; display: flex; justify-content: space-between; }
.conf-bar { height: 4px; background: rgba(255,255,255,0.05); border-radius: 99px; margin-top: 8px; overflow: hidden; margin-bottom: 20px;}
.conf-fill { height: 100%; border-radius: 99px; }
.sp-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: auto; }
.sp-item { background: rgba(0,0,0,0.2); padding: 16px; border-radius: 8px; border: 1px solid var(--border); }
.sp-lbl { color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; font-weight: 600; }
.sp-val { font-size: 15px; font-weight: 700; color: var(--text-main); }
/* QUANT PANEL */
.ind-item { margin-bottom: 28px; }
.ind-hdr { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
.ind-name { font-size: 14px; color: var(--text-muted); font-weight: 600; flex: 1; }
.ind-badge { font-size: 11px; font-weight: 700; border-radius: 4px; padding: 4px 8px; }
.ind-val { font-size: 18px; font-weight: 600; min-width: 44px; text-align: right; }
.rsi-track { position: relative; display: flex; height: 8px; border-radius: 99px; gap: 2px; }
.rsi-zone.os { width: 30%; background: rgba(255,77,109,0.15); border-radius: 99px 0 0 99px; }
.rsi-zone.mid { width: 40%; background: rgba(255,255,255,0.05); }
.rsi-zone.ob { width: 30%; background: rgba(0,255,157,0.15); border-radius: 0 99px 99px 0; }
.rsi-needle { position: absolute; top: -5px; width: 18px; height: 18px; display: flex; align-items: center; justify-content: center; }
.rsi-dot { width: 12px; height: 12px; border-radius: 50%; border: 2px solid var(--panel-bg); }
.rsi-labels { display: flex; justify-content: space-between; margin-top: 8px; font-size: 11px; color: var(--text-muted); }
.price-levels { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; margin-bottom: 24px; background: rgba(0,0,0,0.2); }
.pl-row { display: flex; justify-content: space-between; align-items: center; padding: 14px 18px; font-size: 14px; color: var(--text-muted); }
.pl-sep { height: 1px; background: var(--border); }
.rr-hdr { display: flex; justify-content: space-between; margin-bottom: 12px; font-size: 14px; color: var(--text-muted); }
.rr-bar { display: flex; height: 36px; border-radius: 8px; overflow: hidden; gap: 2px; }
.rr-risk { background: rgba(255,77,109,0.2); display: flex; align-items: center; justify-content: center; border-radius: 8px 0 0 8px; border-right: 1px solid rgba(0,0,0,0.5); }
.rr-gain { background: rgba(0,255,157,0.15); display: flex; align-items: center; justify-content: center; border-radius: 0 8px 8px 0; }
/* IN-DEPTH INSIGHT BOARD */
.insight-board { background: rgba(0,0,0,0.2); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-top: 16px; }
.insight-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }
.insight-badge { padding: 6px 12px; border-radius: 6px; font-weight: 800; font-size: 13px; border: 1px solid; background: rgba(255,255,255,0.03); }
.insight-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }
.ic-card { display: flex; align-items: center; gap: 16px; background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
.ic-icon { font-size: 24px; background: rgba(0,0,0,0.2); width: 48px; height: 48px; border-radius: 8px; display: flex; align-items: center; justify-content: center; }
.ic-info { display: flex; flex-direction: column; gap: 4px; }
.ic-label { font-size: 12px; color: var(--text-muted); font-weight: 600; text-transform: uppercase; }
.ic-val { font-size: 16px; font-weight: 700; color: var(--text-main); font-family: 'IBM Plex Mono', monospace; }
.insight-footer { border-radius: 10px; padding: 20px; text-align: left; }
/* RICH MEDIA NEWS GRID */
.news-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; padding: 10px 0; }
.news-card { display: flex; flex-direction: column; text-decoration: none; background: rgba(0,0,0,0.2); border-radius: 12px; border: 1px solid var(--border); overflow: hidden; transition: all 0.2s ease; height: 100%; box-sizing: border-box; }
.news-img { width: 100%; height: 170px; background-size: cover; background-position: center; border-bottom: 1px solid var(--border); transition: transform 0.3s ease, filter 0.3s ease; }
.news-content { padding: 20px; display: flex; flex-direction: column; flex-grow: 1; }
.news-src { color: var(--green); font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 10px; letter-spacing: 1px; }
.news-title { color: var(--text-main); font-size: 16px; font-weight: 600; margin-bottom: 20px; line-height: 1.5; flex-grow: 1; }
.news-read { color: var(--text-muted); font-size: 12px; font-weight: 600; margin-top: auto; display: flex; align-items: center; gap: 4px; transition: color 0.2s; }
.news-card:hover { border-color: rgba(0,255,157,0.4); transform: translateY(-4px); box-shadow: 0 12px 30px rgba(0,0,0,0.3); }
.news-card:hover .news-img { filter: brightness(1.1); }
.news-card:hover .news-read { color: var(--green); }
/* COMPANY PROFILE DASHBOARD */
.prof-container { padding: 10px 0; }
.prof-header { margin-bottom: 32px; border-bottom: 1px solid var(--border); padding-bottom: 24px; }
.prof-title { display: flex; align-items: center; gap: 16px; }
.ticker-pill { font-family: 'IBM Plex Mono', monospace; font-size: 14px; font-weight: 700; color: #0B0F19; background: var(--green); border-radius: 6px; padding: 4px 12px; }
.sector-pill { font-size: 13px; font-weight: 600; color: var(--text-muted); border: 1px solid var(--border); border-radius: 6px; padding: 4px 12px; background: rgba(255,255,255,0.02); }
.prof-desc { color: #CBD5E1; font-size: 15px; line-height: 1.8; margin: 24px 0 0 0; padding: 24px 28px; background: rgba(0,0,0,0.3); border: 1px solid var(--border); border-left: 4px solid var(--green); border-radius: 8px; box-shadow: inset 0 2px 10px rgba(0,0,0,0.2); }
.prof-metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; }
.pm-group { background: rgba(0,0,0,0.2); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
.pm-group-title { font-size: 14px; font-weight: 700; color: var(--text-main); border-bottom: 1px solid var(--border); padding-bottom: 12px; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 1px; }
.pm-item { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.03); }
.pm-item:last-child { border-bottom: none; padding-bottom: 0; }
.pm-item span:first-child { color: var(--text-muted); font-size: 13px; font-weight: 500; }
.pm-item span:last-child { color: var(--text-main); font-size: 14px; font-weight: 600; }
"""

# =========================================================
# 7. UI LAYOUT
# =========================================================
with gr.Blocks(
    title="StockSage AI Terminal",
) as app:

# ── Professional Top Header Navbar ──
    logo_src = f"data:image/png;base64,{LOGO_B64}" if LOGO_B64 else ""
    # Using inline CSS on the image guarantees Gradio won't blow it up
    img_tag = f'<img src="{logo_src}" style="height: 45px; width: auto; object-fit: contain; margin-right: 15px;" alt="StockSage Logo" />' if logo_src else ''
    
    gr.HTML(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 20px 30px; background: #0B0F19; border-bottom: 1px solid #2A3441; margin-bottom: 24px; border-radius: 8px;">
        
        <div style="display: flex; align-items: center;">
            {img_tag}
            <div style="font-size: 32px; font-weight: 800; letter-spacing: 1px; line-height: 1; font-family: 'Inter', sans-serif;">
                <span style="color: #ffffff;">STOCK</span><span style="color: #00ff9d;">SAGE</span>
            </div>
            <div style="margin-left: 15px; padding-left: 15px; border-left: 1px solid #2A3441; color: #94A3B8; font-size: 13px; font-weight: 500;">
                System standby
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 24px;">
            <div style="color: #94A3B8; font-size: 15px; font-weight: 500;">Market Intel</div>
            <div style="color: #F8FAFC; font-size: 15px; font-weight: 600;">Strategy Terminal</div>
            
            <div style="display: flex; flex-direction: column; align-items: flex-end; margin-left: 10px;">
                <span style="font-size: 12px; color: #94A3B8; margin-bottom: 4px;">Current signal: System Standby</span>
                <div style="padding: 4px 12px; border-radius: 99px; border: 1px solid rgba(0, 255, 157, 0.4); background: rgba(0, 255, 157, 0.05); color: #00ff9d; font-size: 12px; font-weight: 600; font-family: 'IBM Plex Mono', monospace;">
                    &#x21BA; System Standby
                </div>
            </div>
        </div>
    </div>
    """)

    # ── Main Full-Width Content Wrapper ──
    with gr.Row(elem_id="main-layout"):
        with gr.Column(scale=1, min_width=320):
            gr.HTML('<div class="sidebar-title">&#9881; TRADE PARAMETERS</div>')
            with gr.Group():
                ticker = gr.Dropdown(
                    choices=["NVDA","AAPL","MSFT","AMZN","GOOGL","META","TSLA","AMD","NFLX","SPY","QQQ","BTC-USD"],
                    label="Ticker Symbol", value="NVDA", allow_custom_value=True
                )
                amount = gr.Number(value=1000, label="Capital Deployment ($)")

            gr.HTML('<div class="sidebar-title" style="margin-top:20px;">&#128309; RISK CONTROLS</div>')
            with gr.Group():
                target = gr.Slider(minimum=1, maximum=100, value=15, step=1, label="Target Profit (%)")
                loss   = gr.Slider(minimum=1, maximum=50, value=5, step=1, label="Max Loss Tolerance (%)")
                time   = gr.Dropdown(["1 Week", "1 Month", "3 Months", "6 Months"], label="Time Horizon", value="1 Month")

            analyze_btn = gr.Button("EXECUTE ANALYSIS", variant="primary", size="lg")

        with gr.Column(scale=5): 
            with gr.Tabs():
                with gr.Tab("Strategy Terminal", id="tab-strategy"):
                    terminal_header = gr.HTML("<div class='panel-empty'>System Standby: Awaiting Input Execution</div>")
                    metrics_output = gr.HTML()
                    
                    with gr.Group(): 
                        plot_output = gr.Plot(show_label=False)

                    with gr.Row(elem_classes="analysis-container"):
                        with gr.Column(scale=1):
                            strategy_output = gr.HTML()
                        with gr.Column(scale=1):
                            basic_analysis_output = gr.HTML()
                            
                    indepth_output = gr.HTML()

                with gr.Tab("Macro Intel", id="tab-macro"):
                    news_output = gr.HTML("<div class='panel-empty'>Execute analysis to stream market catalysts</div>")

                with gr.Tab("Company Profile", id="tab-company"):
                    fundamentals_output = gr.HTML("<div class='panel-empty'>Execute analysis to load corporate fundamental data</div>")

    # ── Event ─────────────────────────────────────────────
    analyze_btn.click(
        fn=run_full_analysis,
        inputs=[ticker, amount, target, loss, time],
        outputs=[
            terminal_header,
            metrics_output,
            plot_output,
            strategy_output,
            basic_analysis_output,
            indepth_output,
            news_output,
            fundamentals_output
        ]
    )

if __name__ == "__main__":
    app.launch(
        theme=pro_theme,
        css=custom_css,
        head="""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        """
    )
