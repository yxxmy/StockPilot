"""Plotly只画工作流返回的真实数值，不计算或补齐行情。"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def market_chart(data):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045,
        row_heights=[0.48, 0.16, 0.19, 0.17], subplot_titles=["日K与均线 · 元", "成交量 · 手", "MACD 12/26/9", "RSI 14"])
    dates = data["date"]
    fig.add_trace(go.Candlestick(x=dates, open=data["open"], high=data["high"], low=data["low"], close=data["close"],
        name="日K", increasing_line_color="#dc2626", decreasing_line_color="#059669"), row=1, col=1)
    for name, color in [("ma5", "#d97706"), ("ma10", "#2563eb"), ("ma20", "#7c3aed")]:
        fig.add_trace(go.Scatter(x=dates, y=data[name], name=name.upper(), line={"color": color, "width": 1.5}), row=1, col=1)
    colors = ["#dc2626" if close >= opening else "#059669" for close, opening in zip(data["close"], data["open"])]
    fig.add_trace(go.Bar(x=dates, y=data["volume"], name="成交量", marker_color=colors), row=2, col=1)
    fig.add_trace(go.Bar(x=dates, y=data["macd_hist"], name="MACD柱"), row=3, col=1)
    for name in ["macd_dif", "macd_dea"]:
        fig.add_trace(go.Scatter(x=dates, y=data[name], name=name.upper()), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=data["rsi14"], name="RSI14", line_color="#7c3aed"), row=4, col=1)
    fig.add_hline(y=70, line_dash="dot", row=4, col=1)
    fig.add_hline(y=30, line_dash="dot", row=4, col=1)
    fig.update_yaxes(range=[0, 100], row=4, col=1)
    fig.update_layout(height=950, template="plotly_white", margin={"l": 20, "r": 20, "t": 45, "b": 20},
        legend={"orientation": "h", "y": 1.08}, xaxis_rangeslider_visible=False)
    return fig
