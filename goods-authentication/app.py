import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Authentication Economics", layout="wide")
st.title("Inbound Luxury Goods Authentication: Automation vs Expert Augmentation Economics")
st.caption(
    "Items the model flags as fake are discarded immediately (no expert review). "
    "Everything else goes to an expert, who is assumed perfect."
)

st.subheader("Assumptions")
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_items = st.number_input(
        "Total items per month", min_value=1, value=16000, step=100
    )
with col2:
    throughput = st.number_input(
        "Expert throughput (items/month)", min_value=1, value=1600, step=10
    )
with col3:
    salary = st.number_input(
        "Expert salary per month", min_value=0.0, value=60000.0, step=1000.0
    )
with col4:
    price = st.number_input(
        "Economic loss per genuine false reject",
        min_value=0.0, value=10000.0, step=1000.0,
        help=(
            "What it actually costs to wrongly discard a genuine item — not "
            "necessarily retail price. Could be COGS, replacement value, "
            "foregone margin, supplier liability, etc."
        ),
    )

expert_cost_per_item = salary / throughput

FIXED_FPR = 0.01
FIXED_FAKE_RATE = 0.05

# Clean red/green divide at zero — saturation encodes magnitude, no yellow
# "neutral" band, since most of the surface is meaningfully positive or negative.
RED_GREEN = [
    [0.0, "rgb(103,0,13)"],
    [0.25, "rgb(214,96,77)"],
    [0.5, "rgb(255,255,255)"],
    [0.75, "rgb(90,174,97)"],
    [1.0, "rgb(0,68,27)"],
]


def net_gain(fake, fpr_val, cost=None):
    """Net monthly gain(+)/loss(-) from using the classification model."""
    if cost is None:
        cost = expert_cost_per_item
    return total_items * (cost * fake + fpr_val * (1 - fake) * (cost - price))


CHART_HEIGHT = 340
CHART_FONT = dict(color="rgba(230, 230, 230, 0.95)", size=13)


def line_fig(x, y, x_title, y_title="Net per month", x_tickformat=None, x_hover_format=".2f"):
    y_pos = np.where(y >= 0, y, 0.0)
    y_neg = np.where(y <= 0, y, 0.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y_neg, mode="lines", line=dict(width=0), fill="tozeroy",
            fillcolor="rgba(178,24,43,0.45)", hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y_pos, mode="lines", line=dict(width=0), fill="tozeroy",
            fillcolor="rgba(27,120,55,0.45)", hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines", line=dict(color="black", width=2),
            showlegend=False,
            hovertemplate=f"{x_title}: " + "%{x:" + x_hover_format + "}<br>"
            + f"{y_title}: " + "%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"))

    xaxis = dict(range=[x.min(), x.max()])
    if x_tickformat:
        xaxis["tickformat"] = x_tickformat

    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis=xaxis,
        margin=dict(t=10, l=10, r=10, b=40),
        height=CHART_HEIGHT,
        font=CHART_FONT,
    )
    return fig


FPR_MAX = 0.10


st.markdown(
    """
    <style>
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
        color: var(--text-color) !important;
        opacity: 0.92 !important;
        font-size: 0.9rem;
    }
    .st-key-chart_columns [data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"]:not(:first-child) {
        border-left: 1px solid rgba(128, 128, 128, 0.4);
        padding-left: 1.5rem;
    }
    .st-key-top_split > div > [data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"]:nth-child(2) {
        border-left: 4px solid rgba(200, 200, 200, 0.9);
        padding-left: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

top_split = st.container(key="top_split")
top_left, top_right = top_split.columns([3, 1])

with top_left:
    st.subheader("Net Gain/Loss From Classification Model")
    st.info(
        "**Counterfeit recall assumed = 100% (best case):** every genuine "
        "fake is caught, either by the model directly or by the expert "
        "reviewing what's left. Real-world recall below 100% would only "
        "make these numbers worse — this is the model's best possible case."
    )
    chart_columns = st.container(key="chart_columns")
    left, middle, right = chart_columns.columns(3)

    with left:
        st.markdown(f"**Fake rate vs net** (FPR fixed at {FIXED_FPR:.0%})")
        fake_rate_1d = np.linspace(0.0, 0.99, 200)
        net_1d_a = net_gain(fake_rate_1d, FIXED_FPR)
        fig_left = line_fig(fake_rate_1d, net_1d_a, "Fake rate (true prevalence)")

        cost_minus_price = expert_cost_per_item - price
        denom = expert_cost_per_item - FIXED_FPR * cost_minus_price
        if denom != 0:
            breakeven_fake = (-FIXED_FPR * cost_minus_price) / denom
            if 0 <= breakeven_fake <= 0.99:
                fig_left.add_vline(
                    x=breakeven_fake, line=dict(color="black", dash="dot", width=1)
                )
                fig_left.add_annotation(
                    x=breakeven_fake, y=0, xref="x", yref="y",
                    text=(
                        f"<b>Breakeven fake rate ≈ {breakeven_fake:.1%}</b><br>"
                        "even assuming 100% fake recall"
                    ),
                    showarrow=True, arrowhead=2, ax=-70, ay=-70,
                    font=dict(size=13, color="black"),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="black", borderwidth=1,
                )

        st.plotly_chart(fig_left, use_container_width=True)
        st.caption(
            f"Holds false positive rate fixed at {FIXED_FPR:.0%} and sweeps the "
            "fake rate. Where the line crosses from red to green is the fake "
            "rate above which the model pays for itself at this FPR."
        )

    with middle:
        st.markdown(f"**False positive rate vs net** (fake rate fixed at {FIXED_FAKE_RATE:.0%})")
        fpr_1d = np.linspace(0.0, FPR_MAX, 200)
        net_1d_b = net_gain(FIXED_FAKE_RATE, fpr_1d)
        fig_middle = line_fig(
            fpr_1d, net_1d_b, "False positive rate",
            x_tickformat=".0%", x_hover_format=".3%",
        )

        if price > expert_cost_per_item:
            breakeven_fixed = (
                expert_cost_per_item * FIXED_FAKE_RATE
                / ((1 - FIXED_FAKE_RATE) * (price - expert_cost_per_item))
            )
            if 0 <= breakeven_fixed <= FPR_MAX:
                fig_middle.add_vline(
                    x=breakeven_fixed, line=dict(color="black", dash="dot", width=1)
                )
                fig_middle.add_annotation(
                    x=breakeven_fixed, y=0, xref="x", yref="y",
                    text=(
                        f"<b>Breakeven FPR ≈ {breakeven_fixed:.4%}</b><br>"
                        "even assuming 100% fake recall"
                    ),
                    showarrow=True, arrowhead=2, ax=70, ay=-70,
                    font=dict(size=13, color="black"),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="black", borderwidth=1,
                )

        st.plotly_chart(fig_middle, use_container_width=True)
        st.caption(
            f"Holds fake rate fixed at {FIXED_FAKE_RATE:.0%} and sweeps the false "
            "positive rate (capped at 10% — the model is already losing money "
            "well before that). The crossing point is the highest FPR the "
            "model can tolerate before it starts losing money."
        )

    with right:
        st.markdown("**Fake rate vs false positive rate** (breakeven map)")

        fake_rate = np.linspace(0.0, 0.99, 150)
        fpr = np.linspace(0.0, FPR_MAX, 150)
        FAKE, FPR = np.meshgrid(fake_rate, fpr)
        NET = net_gain(FAKE, FPR)

        # Signed log10 color mapping: an odd function run through a symmetric
        # zmin/zmax, so equal magnitude on either side of 0 always gets equal
        # saturation. Ticks are thinned once (ascending from 0) and then
        # mirrored, so if a magnitude is shown on one side it's always shown
        # on the other too — no independently-chosen, mismatched tick sets.
        def _log_transform(v):
            return np.sign(v) * np.log10(1 + np.abs(v))

        NET_LOG = _log_transform(NET)
        log_bound = float(np.abs(NET_LOG).max()) or 1.0
        raw_max = float(np.abs(NET).max()) or 1.0

        top_exp = int(np.floor(np.log10(raw_max))) if raw_max >= 1 else 0
        magnitude_candidates = [10**e for e in range(max(top_exp - 3, 0), top_exp + 1)]

        min_gap = 0.12 * (2 * log_bound)
        kept_pos = [0]
        last_t = 0.0
        for m in magnitude_candidates:
            t = _log_transform(m)
            if t - last_t >= min_gap:
                kept_pos.append(m)
                last_t = t

        # The outermost magnitude is where the scale actually bottoms/tops
        # out (full saturation) — always label it, even if it's closer to
        # the next-in tick than the spacing filter would normally allow.
        if magnitude_candidates and magnitude_candidates[-1] not in kept_pos:
            kept_pos.append(magnitude_candidates[-1])

        tick_raw = [-m for m in reversed(kept_pos[1:])] + kept_pos
        tick_vals = [_log_transform(v) for v in tick_raw]

        def _fmt(v):
            for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
                if abs(v) >= div:
                    return f"{v / div:,.0f}{suf}"
            return f"{v:,.0f}"

        tick_text = [_fmt(v) for v in tick_raw]

        fig_map = go.Figure()
        fig_map.add_trace(
            go.Heatmap(
                x=fake_rate,
                y=fpr,
                z=NET_LOG,
                customdata=NET,
                colorscale=RED_GREEN,
                zmid=0,
                zmin=-log_bound,
                zmax=log_bound,
                colorbar=dict(title="Net per<br>month", tickvals=tick_vals, ticktext=tick_text),
                zsmooth="best",
                hovertemplate=(
                    "Fake rate: %{x:.2f}<br>"
                    "False positive rate: %{y:.3%}<br>"
                    "Net per month: %{customdata:,.0f}<extra></extra>"
                ),
            )
        )

        if price > expert_cost_per_item:
            fpr_breakeven = (expert_cost_per_item * fake_rate) / (
                (1 - fake_rate) * (price - expert_cost_per_item)
            )
            fpr_breakeven_clipped = np.clip(fpr_breakeven, 0, FPR_MAX)
            fig_map.add_trace(
                go.Scatter(
                    x=fake_rate,
                    y=fpr_breakeven_clipped,
                    mode="lines",
                    line=dict(color="black", width=2),
                    name="Breakeven (net = 0)",
                    hoverinfo="skip",
                )
            )
        else:
            st.info(
                "The economic loss per genuine false reject is not greater than "
                f"the expert's per-item cost ({expert_cost_per_item:,.2f}), so "
                "the model is a net gain at any false positive rate — there is "
                "no breakeven line."
            )

        fig_map.update_layout(
            xaxis_title="Fake rate (true prevalence)",
            yaxis_title="False positive rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, FPR_MAX], tickformat=".0%"),
            showlegend=False,
            margin=dict(t=10, l=10, r=10, b=40),
            height=CHART_HEIGHT,
            font=CHART_FONT,
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption(
            "Both axes together. The black line is net = 0 (breakeven); above "
            "it false positives cost more than the expert time saved (net "
            "loss), below it the model saves more than it costs (net gain)."
        )

    st.caption(
        f"Expert cost per item: {expert_cost_per_item:,.2f}. All three charts show "
        "the same net gain(+)/loss(-) per month from using the classification "
        "model — green is net gain, red is net loss."
    )

with top_right:
    st.subheader("Net Gain From Expert Augmentation Model")
    st.markdown("**Throughput multiplier vs savings**")

    throughput_mult = np.linspace(1.0, 3.0, 200)
    baseline_cost = total_items * expert_cost_per_item
    scaled_cost = total_items * (salary / (throughput * throughput_mult))
    cost_saving = baseline_cost - scaled_cost

    st.plotly_chart(
        line_fig(
            throughput_mult,
            cost_saving,
            "Throughput multiplier (×)",
            y_title="Cost savings per month",
        ),
        use_container_width=True,
    )
    st.caption(
        "No classification model here — just making the same expert faster "
        "(up to 3×), so every item still gets reviewed. A higher multiplier "
        "lowers the cost per item, so this is pure savings, always net gain."
    )
