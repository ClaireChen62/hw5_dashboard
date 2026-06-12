import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from arch import arch_model
from itertools import combinations
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="FE HW5 - DCC-GARCH Dashboard", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    h1 { font-size: 1.6rem; font-weight: 700; }
    h2 { font-size: 1.2rem; font-weight: 600; }
    .stMetric label { font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

TICKERS     = ['SPY', 'GLD', 'TLT', 'EEM', 'XOM']
COLORS      = ['#2166ac', '#d6604d', '#4dac26', '#762a83', '#e08214']
COLOR_MAP   = dict(zip(TICKERS, COLORS))
START       = '2015-01-01'
END         = '2024-12-31'
COVID_START = '2020-02-01'
COVID_END   = '2020-05-31'

# ── Cached computations ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Downloading price data...")
def load_returns():
    raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True)['Close']
    raw = raw.dropna()
    ret = 100 * np.log(raw / raw.shift(1)).dropna()
    ret.columns = TICKERS
    return ret

@st.cache_data(show_spinner="Fitting GARCH models...")
def fit_garch(_returns):
    cond_vol  = pd.DataFrame(index=_returns.index, columns=TICKERS, dtype=float)
    std_resid = pd.DataFrame(index=_returns.index, columns=TICKERS, dtype=float)
    params = {}
    for col in TICKERS:
        am  = arch_model(_returns[col], vol='Garch', p=1, q=1, dist='normal', rescale=False)
        res = am.fit(disp='off')
        params[col]    = res.params
        cond_vol[col]  = res.conditional_volatility
        std_resid[col] = res.resid / res.conditional_volatility
    return cond_vol.dropna(), std_resid.dropna(), params

@st.cache_data(show_spinner="Estimating DCC parameters (~2 min)...")
def fit_dcc(_std_resid):
    e     = _std_resid.values
    T, N  = e.shape
    Q_bar = e.T @ e / T

    def neg_loglik(params):
        a, b = params
        if a <= 0 or b <= 0 or (a + b) >= 1:
            return 1e10
        Q  = Q_bar.copy()
        ll = 0.0
        for t in range(T):
            if t > 0:
                Q = (1 - a - b) * Q_bar + a * np.outer(e[t-1], e[t-1]) + b * Q
            d = np.sqrt(np.diag(Q))
            R = Q / np.outer(d, d)
            s, ld = np.linalg.slogdet(R)
            if s <= 0:
                return 1e10
            ll += ld + float(e[t] @ np.linalg.solve(R, e[t]))
        return 0.5 * ll

    opt = minimize(neg_loglik, x0=[0.05, 0.90], method='Nelder-Mead',
                   options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6})
    a_dcc, b_dcc = opt.x

    R_series = np.zeros((T, N, N))
    Q = Q_bar.copy()
    for t in range(T):
        if t > 0:
            Q = (1 - a_dcc - b_dcc) * Q_bar + a_dcc * np.outer(e[t-1], e[t-1]) + b_dcc * Q
        d = np.sqrt(np.diag(Q))
        R_series[t] = Q / np.outer(d, d)
    return a_dcc, b_dcc, R_series, Q_bar

def build_sigma(R_series, cond_vol):
    T, N, _ = R_series.shape
    Sigma = np.zeros((T, N, N))
    for t in range(T):
        D = np.diag(cond_vol.values[t])
        Sigma[t] = D @ R_series[t] @ D
    return Sigma

def compute_mvp(Sigma_series, index):
    N    = Sigma_series.shape[1]
    ones = np.ones(N)
    w    = np.zeros((len(index), N))
    for t in range(len(index)):
        try:
            Si   = np.linalg.inv(Sigma_series[t])
            w[t] = (Si @ ones) / (ones @ Si @ ones)
        except:
            w[t] = ones / N
    return pd.DataFrame(w, index=index, columns=TICKERS)

def make_subplot_fig(n_rows, height_per_row=180):
    return make_subplots(rows=n_rows, cols=1, shared_xaxes=False,
                         vertical_spacing=0.06)

def style_fig(fig, n_rows):
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                      margin=dict(t=20, b=20))
    for r in range(1, n_rows + 1):
        fig.update_xaxes(showgrid=True, gridcolor='#eeeeee',
                         tickformat='%Y', showticklabels=True, row=r, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='#eeeeee')
    return fig

def add_covid(fig, row, y0, y1):
    fig.add_shape(type="rect", xref="x", yref="y",
                  x0=COVID_START, x1=COVID_END, y0=y0, y1=y1,
                  fillcolor="rgba(255,0,0,0.10)", line_width=0,
                  row=row, col=1)
    return fig

# ── Load all data ─────────────────────────────────────────────────────────
returns               = load_returns()
cond_vol, std_resid, garch_params = fit_garch(returns)
a_dcc, b_dcc, R_series, Q_bar     = fit_dcc(std_resid)
Sigma_series          = build_sigma(R_series, cond_vol)
mvp_df                = compute_mvp(Sigma_series, std_resid.index)
ret_aligned           = returns.loc[std_resid.index]
mvp_returns           = pd.Series((mvp_df.values * ret_aligned.values).sum(axis=1),
                                   index=std_resid.index, name='MVP')
ew_returns            = ret_aligned.mean(axis=1)
ew_returns.name       = 'EW'
dcc_index             = std_resid.index
pair_indices          = list(combinations(range(len(TICKERS)), 2))
pair_labels           = [f"{TICKERS[i]} vs {TICKERS[j]}" for i, j in pair_indices]

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.title("FE HW5")
st.sidebar.markdown("**DCC-GARCH Dashboard**")
st.sidebar.markdown("Yingzhou Fang · Ke Chen · Satya Anirudh Pachipulusu")
st.sidebar.markdown("---")
page = st.sidebar.radio("Section", [
    "Returns & Descriptive Statistics",
    "DCC-GARCH Model",
    "Portfolio Analysis"
])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Sample:** {START} to {END}")
st.sidebar.markdown(f"**Observations:** {len(returns):,}")
st.sidebar.markdown(f"**DCC a:** {a_dcc:.4f} | **b:** {b_dcc:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1
# ══════════════════════════════════════════════════════════════════════════
if page == "Returns & Descriptive Statistics":
    st.title("Returns and Descriptive Statistics")

    # Q1 - time series
    st.subheader("Q1 - Daily Percentage Log Returns")
    asset_sel = st.multiselect("Select assets", TICKERS, default=TICKERS)
    if asset_sel:
        n = len(asset_sel)
        fig = make_subplot_fig(n)
        for i, ticker in enumerate(asset_sel, 1):
            fig.add_trace(go.Scatter(
                x=returns.index, y=returns[ticker],
                mode='lines', line=dict(width=0.6, color=COLOR_MAP[ticker]),
                name=ticker, showlegend=False
            ), row=i, col=1)
            add_covid(fig, i, returns[ticker].min(), returns[ticker].max())
            fig.update_yaxes(title_text=f"{ticker}  |  Return (%)",
                             title_font_size=10, row=i, col=1)
        fig.update_layout(height=180 * n)
        style_fig(fig, n)
        fig.update_yaxes(zeroline=True, zerolinecolor='black', zerolinewidth=0.5)
        st.plotly_chart(fig, use_container_width=True)

    # Q1 - descriptive stats
    st.subheader("Q1 - Descriptive Statistics")
    desc = pd.DataFrame({
        'Mean':         returns.mean(),
        'Std Dev':      returns.std(),
        'Min':          returns.min(),
        'Max':          returns.max(),
        'Skewness':     returns.skew(),
        'Ex. Kurtosis': returns.kurtosis(),
    }).round(4)
    st.dataframe(desc, use_container_width=True)

    # Q1 - unconditional correlation
    st.subheader("Q1 - Unconditional Correlation Matrix")
    corr = returns.corr()
    fig2 = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                     zmin=-1, zmax=1, aspect='auto')
    fig2.update_layout(height=420, margin=dict(t=20, b=20))
    st.plotly_chart(fig2, use_container_width=True)

    # Q2 - scatterplots
    st.subheader("Q2 - Bivariate Scatterplots")
    sel_pair = st.selectbox("Select pair", pair_labels)
    idx      = pair_labels.index(sel_pair)
    pi, pj   = pair_indices[idx]
    a_t, b_t = TICKERS[pi], TICKERS[pj]
    x_lo, x_hi = returns[a_t].quantile(0.02), returns[a_t].quantile(0.98)
    y_lo, y_hi = returns[b_t].quantile(0.02), returns[b_t].quantile(0.98)
    mask = returns[a_t].between(x_lo, x_hi) & returns[b_t].between(y_lo, y_hi)
    corr_val = returns[[a_t, b_t]].corr().iloc[0, 1]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=returns[a_t][mask], y=returns[b_t][mask],
        mode='markers', marker=dict(size=3, color='steelblue', opacity=0.3),
        name='Returns'
    ))
    fig3.update_layout(
        title=f"{a_t} vs {b_t}  (r = {corr_val:.2f})",
        xaxis_title=a_t, yaxis_title=b_t,
        height=420, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(t=40, b=30)
    )
    fig3.update_xaxes(showgrid=True, gridcolor='#eeeeee')
    fig3.update_yaxes(showgrid=True, gridcolor='#eeeeee')
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2
# ══════════════════════════════════════════════════════════════════════════
elif page == "DCC-GARCH Model":
    st.title("DCC-GARCH Model")

    # Q3 - parameters
    st.subheader("Q3 - GARCH(1,1) and DCC Parameter Estimates")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**GARCH(1,1) parameters**")
        garch_df = pd.DataFrame({
            'omega': [garch_params[t]['omega']    for t in TICKERS],
            'alpha': [garch_params[t]['alpha[1]'] for t in TICKERS],
            'beta':  [garch_params[t]['beta[1]']  for t in TICKERS],
        }, index=TICKERS)
        garch_df['alpha+beta'] = garch_df['alpha'] + garch_df['beta']
        st.dataframe(garch_df.round(4), use_container_width=True)
    with col2:
        st.markdown("**DCC parameters**")
        dcc_df = pd.DataFrame({
            'Parameter': ['a (alpha)', 'b (beta)', 'a + b'],
            'Estimate':  [round(a_dcc, 6), round(b_dcc, 6), round(a_dcc + b_dcc, 6)]
        })
        st.dataframe(dcc_df, use_container_width=True, hide_index=True)
        st.caption(f"Stationarity: a + b = {a_dcc+b_dcc:.4f} < 1")

    # Q4 - conditional volatility
    st.subheader("Q4 - Conditional Standard Deviations")
    asset_sel4 = st.multiselect("Select assets", TICKERS, default=TICKERS, key='q4')
    if asset_sel4:
        n4  = len(asset_sel4)
        fig4 = make_subplot_fig(n4)
        for i, ticker in enumerate(asset_sel4, 1):
            fig4.add_trace(go.Scatter(
                x=cond_vol.index, y=cond_vol[ticker],
                mode='lines', line=dict(width=0.7, color=COLOR_MAP[ticker]),
                name=ticker, showlegend=False
            ), row=i, col=1)
            add_covid(fig4, i, 0, float(cond_vol[ticker].max()) * 1.05)
            fig4.update_yaxes(title_text=f"{ticker}  |  Cond. Std Dev (%)",
                              title_font_size=10, row=i, col=1)
        fig4.update_layout(height=180 * n4)
        style_fig(fig4, n4)
        st.plotly_chart(fig4, use_container_width=True)

    # Q5 - DCC correlations
    st.subheader("Q5 - DCC Conditional Correlations")
    sel_pairs = st.multiselect("Select pairs", pair_labels, default=pair_labels[:5])
    if sel_pairs:
        fig5 = go.Figure()
        for label in sel_pairs:
            idx2 = pair_labels.index(label)
            pi, pj = pair_indices[idx2]
            fig5.add_trace(go.Scatter(
                x=dcc_index, y=R_series[:, pi, pj],
                mode='lines', line=dict(width=0.8), name=label
            ))
        fig5.add_vrect(x0=COVID_START, x1=COVID_END,
                       fillcolor="rgba(255,0,0,0.08)", line_width=0,
                       annotation_text="COVID crash", annotation_position="top left")
        fig5.add_hline(y=0, line_width=0.5, line_color='black')
        fig5.update_layout(
            height=420, yaxis=dict(range=[-1.05, 1.05], title="Conditional Correlation"),
            xaxis=dict(tickformat='%Y'), plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(t=20, b=20), legend=dict(orientation='h', y=-0.15)
        )
        fig5.update_xaxes(showgrid=True, gridcolor='#eeeeee')
        fig5.update_yaxes(showgrid=True, gridcolor='#eeeeee')
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("**DCC Conditional Correlation Summary**")
    summary = []
    for pi, pj in pair_indices:
        ts = R_series[:, pi, pj]
        summary.append({'Pair': f"{TICKERS[pi]} vs {TICKERS[pj]}",
                        'Mean': round(ts.mean(), 4), 'Min': round(ts.min(), 4),
                        'Max':  round(ts.max(),  4), 'Std Dev': round(ts.std(), 4)})
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3
# ══════════════════════════════════════════════════════════════════════════
elif page == "Portfolio Analysis":
    st.title("Portfolio Analysis")

    # Q6 - MVP weights
    st.subheader("Q6 - Minimum Variance Portfolio Weights")
    asset_sel6 = st.multiselect("Select assets", TICKERS, default=TICKERS, key='q6')
    if asset_sel6:
        n6   = len(asset_sel6)
        fig6 = make_subplot_fig(n6)
        for i, ticker in enumerate(asset_sel6, 1):
            mean_w = float(mvp_df[ticker].mean())
            fig6.add_trace(go.Scatter(
                x=mvp_df.index, y=mvp_df[ticker],
                mode='lines', line=dict(width=0.7, color=COLOR_MAP[ticker]),
                name=ticker, showlegend=False
            ), row=i, col=1)
            fig6.add_hline(y=mean_w, line_dash='dash', line_color='black',
                           line_width=0.8, row=i, col=1,
                           annotation_text=f"mean={mean_w:.3f}",
                           annotation_position="top right")
            add_covid(fig6, i, float(mvp_df[ticker].min()), float(mvp_df[ticker].max()))
            fig6.update_yaxes(title_text=f"{ticker}  |  Weight",
                              title_font_size=10, row=i, col=1)
        fig6.update_layout(height=180 * n6)
        style_fig(fig6, n6)
        fig6.update_yaxes(zeroline=True, zerolinecolor='#aaaaaa')
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("**Average MVP weights over full sample**")
    st.dataframe(mvp_df.mean().round(4).to_frame(name='Average Weight').T,
                 use_container_width=True)

    # Q7 - MVP returns
    st.subheader("Q7 - MVP Portfolio Returns")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Return (%)", f"{mvp_returns.mean():.4f}")
    c2.metric("Variance",        f"{mvp_returns.var():.4f}")
    c3.metric("Std Dev (%)",     f"{mvp_returns.std():.4f}")
    c4.metric("Sharpe Ratio",    f"{mvp_returns.mean()/mvp_returns.std():.4f}")

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=mvp_returns.index, y=mvp_returns,
                              mode='lines', line=dict(width=0.6, color='steelblue'),
                              name='MVP'))
    fig7.add_vrect(x0=COVID_START, x1=COVID_END,
                   fillcolor="rgba(255,0,0,0.08)", line_width=0,
                   annotation_text="COVID crash", annotation_position="top left")
    fig7.add_hline(y=0, line_width=0.5, line_color='black')
    fig7.update_layout(height=340, yaxis_title="Return (%)",
                       xaxis=dict(tickformat='%Y'),
                       plot_bgcolor='white', paper_bgcolor='white',
                       margin=dict(t=20, b=20))
    fig7.update_xaxes(showgrid=True, gridcolor='#eeeeee')
    fig7.update_yaxes(showgrid=True, gridcolor='#eeeeee')
    st.plotly_chart(fig7, use_container_width=True)

    # Q8 - comparison
    st.subheader("Q8 - MVP vs Equal-Weight Portfolio")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Portfolio Statistics**")
        comp_df = pd.DataFrame({
            'Mean Return (%)': [mvp_returns.mean(),  ew_returns.mean()],
            'Variance':        [mvp_returns.var(),   ew_returns.var()],
            'Std Dev (%)':     [mvp_returns.std(),   ew_returns.std()],
            'Sharpe Ratio':    [mvp_returns.mean()/mvp_returns.std(),
                                ew_returns.mean()/ew_returns.std()],
        }, index=['MVP', 'Equal-Weight']).round(4)
        st.dataframe(comp_df, use_container_width=True)
    with col2:
        var_red = (1 - mvp_returns.var() / ew_returns.var()) * 100
        sharpe_diff = mvp_returns.mean()/mvp_returns.std() - ew_returns.mean()/ew_returns.std()
        st.metric("Variance reduction", f"{var_red:.1f}%")
        st.metric("Sharpe improvement", f"+{sharpe_diff:.4f}")

    # Daily returns comparison
    fig8 = make_subplot_fig(2)
    fig8.add_trace(go.Scatter(x=mvp_returns.index, y=mvp_returns,
                              mode='lines', line=dict(width=0.6, color='steelblue'),
                              name='MVP', showlegend=False), row=1, col=1)
    fig8.add_trace(go.Scatter(x=ew_returns.index, y=ew_returns,
                              mode='lines', line=dict(width=0.6, color='darkorange'),
                              name='Equal-Weight', showlegend=False), row=2, col=1)
    for r in [1, 2]:
        add_covid(fig8, r, -10, 10)
    fig8.update_yaxes(title_text="MVP  |  Return (%)", title_font_size=10, row=1, col=1)
    fig8.update_yaxes(title_text="Equal-Weight  |  Return (%)", title_font_size=10, row=2, col=1)
    fig8.update_layout(height=480)
    style_fig(fig8, 2)
    fig8.update_yaxes(zeroline=True, zerolinecolor='black', zerolinewidth=0.4)
    st.plotly_chart(fig8, use_container_width=True)

    # Cumulative returns
    cumul_mvp = (1 + mvp_returns / 100).cumprod()
    cumul_ew  = (1 + ew_returns  / 100).cumprod()
    fig9 = go.Figure()
    fig9.add_trace(go.Scatter(x=cumul_mvp.index, y=cumul_mvp,
                              mode='lines', line=dict(width=1.2, color='steelblue'),
                              name='MVP'))
    fig9.add_trace(go.Scatter(x=cumul_ew.index, y=cumul_ew,
                              mode='lines', line=dict(width=1.2, color='darkorange',
                              dash='dash'), name='Equal-Weight'))
    fig9.add_vrect(x0=COVID_START, x1=COVID_END,
                   fillcolor="rgba(255,0,0,0.08)", line_width=0,
                   annotation_text="COVID crash", annotation_position="top left")
    fig9.update_layout(
        height=380, yaxis_title="Cumulative Return (base = 1)",
        xaxis=dict(tickformat='%Y'),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(t=20, b=20), legend=dict(orientation='h', y=-0.15)
    )
    fig9.update_xaxes(showgrid=True, gridcolor='#eeeeee')
    fig9.update_yaxes(showgrid=True, gridcolor='#eeeeee')
    st.plotly_chart(fig9, use_container_width=True)
