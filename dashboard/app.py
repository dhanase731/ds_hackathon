"""
StockSense Dashboard  –  IntelliData 2026
Run:  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

PROCESSED_DIR = BASE_DIR / "data" / "processed"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense | NovaMart",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        border-radius: 12px; padding: 20px; color: white; text-align: center;
    }
    .risk-high   { background: #ff4b4b; color: white; border-radius: 6px; padding: 4px 10px; font-weight: bold; }
    .risk-medium { background: #ffa500; color: white; border-radius: 6px; padding: 4px 10px; font-weight: bold; }
    .risk-low    { background: #21c354; color: white; border-radius: 6px; padding: 4px 10px; font-weight: bold; }
    .section-header { font-size: 1.4rem; font-weight: 700; color: #1e3a5f; border-bottom: 3px solid #2d6a9f; padding-bottom: 6px; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_master():
    p = PROCESSED_DIR / "master_dataset.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["date"])
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_features():
    p = PROCESSED_DIR / "feature_engineered_dataset.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["date"])
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_demand_preds():
    p = PROCESSED_DIR / "demand_predictions.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["date"])
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_stockout_preds():
    p = PROCESSED_DIR / "stockout_predictions.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["date"])
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_recommendations():
    p = PROCESSED_DIR / "recommendations.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_model_metrics():
    """Try to load saved metrics, else return empty."""
    p = PROCESSED_DIR / "model_metrics.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def run_pipeline():
    """Train models and generate all outputs."""
    import subprocess
    script = BASE_DIR / "run_pipeline.py"
    if script.exists():
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(BASE_DIR),
            capture_output=True, text=True
        )
        if result.returncode != 0:
            st.error(f"Pipeline error:\n{result.stderr[-2000:]}")
            return
    st.cache_data.clear()
    st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/warehouse.png", width=80)
    st.title("StockSense")
    st.caption("NovaMart Retail Pvt. Ltd.")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Executive Summary", "📈 Demand Intelligence",
         "⚠️ Inventory Risk", "🎯 Manager Action Centre",
         "🤖 Model Performance", "🔍 Explainability"],
        label_visibility="collapsed"
    )
    st.divider()

    if st.button("🔄 Re-run Pipeline", use_container_width=True):
        with st.spinner("Training models… this may take a minute."):
            run_pipeline()

    st.caption("IntelliData 2026 | Sri Eshwar College of Engineering")

# ── Load data ─────────────────────────────────────────────────────────────────
master = load_master()
features = load_features()
demand_preds = load_demand_preds()
stockout_preds = load_stockout_preds()
recs = load_recommendations()

data_ready = not master.empty

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Summary":
    st.title("📊 Executive Summary")
    st.caption("NovaMart Retail — StockSense Decision Support System")

    if not data_ready:
        st.warning("No processed data found. Click **Re-run Pipeline** in the sidebar.")
        st.stop()

    # KPIs
    revenue = (master["units_sold"] * master["price"]).sum()
    units = master["units_sold"].sum()
    stockout_rate = (master["inventory_level"] < master["demand"]).mean() * 100 if "inventory_level" in master.columns else 0
    inv_value = (master["inventory_level"] * master["price"]).sum() if "inventory_level" in master.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Total Revenue", f"₹{revenue/1e6:.1f}M")
    c2.metric("📦 Units Sold", f"{units:,.0f}")
    c3.metric("⚠️ Stock-out Rate", f"{stockout_rate:.1f}%")
    c4.metric("🏪 Inventory Value", f"₹{inv_value/1e6:.1f}M")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Revenue by Store</p>', unsafe_allow_html=True)
        master["revenue"] = master["units_sold"] * master["price"]
        rev_store = master.groupby("store_id")["revenue"].sum().reset_index()
        fig = px.bar(rev_store, x="store_id", y="revenue", color="store_id",
                     labels={"revenue": "Revenue (₹)", "store_id": "Store"},
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Daily Revenue Trend</p>', unsafe_allow_html=True)
        daily = master.groupby("date")["revenue"].sum().reset_index()
        fig2 = px.line(daily, x="date", y="revenue",
                       labels={"revenue": "Revenue (₹)", "date": "Date"},
                       color_discrete_sequence=["#2d6a9f"])
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<p class="section-header">Promotion vs Normal Sales</p>', unsafe_allow_html=True)
        promo = master.groupby("promotion")["units_sold"].mean().reset_index()
        promo["promotion"] = promo["promotion"].map({0: "Normal", 1: "Promotion"})
        fig3 = px.bar(promo, x="promotion", y="units_sold",
                      color="promotion", color_discrete_map={"Normal": "#6c757d", "Promotion": "#2d6a9f"},
                      labels={"units_sold": "Avg Units Sold"})
        fig3.update_layout(showlegend=False, height=280)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        if not recs.empty:
            st.markdown('<p class="section-header">Risk Distribution</p>', unsafe_allow_html=True)
            risk_counts = recs["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk", "Count"]
            fig4 = px.pie(risk_counts, names="Risk", values="Count",
                          color="Risk",
                          color_discrete_map={"HIGH": "#ff4b4b", "MEDIUM": "#ffa500", "LOW": "#21c354"})
            fig4.update_layout(height=280)
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Run pipeline to see risk distribution.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – DEMAND INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Demand Intelligence":
    st.title("📈 Demand Intelligence")

    if not data_ready:
        st.warning("No processed data found. Click **Re-run Pipeline** in the sidebar.")
        st.stop()

    # Filters
    stores = sorted(master["store_id"].unique())
    products = sorted(master["product_id"].unique())
    sel_store = st.selectbox("Select Store", ["All"] + stores)
    sel_product = st.selectbox("Select Product", ["All"] + products)

    df_filt = master.copy()
    df_filt["revenue"] = df_filt["units_sold"] * df_filt["price"]
    if sel_store != "All":
        df_filt = df_filt[df_filt["store_id"] == sel_store]
    if sel_product != "All":
        df_filt = df_filt[df_filt["product_id"] == sel_product]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Actual Demand Over Time</p>', unsafe_allow_html=True)
        daily_demand = df_filt.groupby("date")["demand"].sum().reset_index()
        fig = px.line(daily_demand, x="date", y="demand",
                      labels={"demand": "Total Demand", "date": "Date"},
                      color_discrete_sequence=["#2d6a9f"])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if not demand_preds.empty:
            st.markdown('<p class="section-header">Actual vs Predicted Demand</p>', unsafe_allow_html=True)
            dp = demand_preds.copy()
            if sel_store != "All":
                dp = dp[dp["store_id"] == sel_store]
            if sel_product != "All":
                dp = dp[dp["product_id"] == sel_product]
            dp_agg = dp.groupby("date")[["demand", "predicted_demand"]].sum().reset_index()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=dp_agg["date"], y=dp_agg["demand"],
                                      name="Actual", line=dict(color="#2d6a9f")))
            fig2.add_trace(go.Scatter(x=dp_agg["date"], y=dp_agg["predicted_demand"],
                                      name="Predicted", line=dict(color="#ff6b35", dash="dash")))
            fig2.update_layout(height=300, legend=dict(orientation="h"))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Train demand model to see predictions.")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<p class="section-header">Weekday vs Weekend Demand</p>', unsafe_allow_html=True)
        df_filt["weekday"] = pd.to_datetime(df_filt["date"]).dt.dayofweek
        df_filt["day_type"] = df_filt["weekday"].apply(lambda x: "Weekend" if x >= 5 else "Weekday")
        wk = df_filt.groupby("day_type")["demand"].mean().reset_index()
        fig3 = px.bar(wk, x="day_type", y="demand",
                      color="day_type",
                      color_discrete_map={"Weekday": "#6c757d", "Weekend": "#2d6a9f"},
                      labels={"demand": "Avg Demand"})
        fig3.update_layout(showlegend=False, height=280)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<p class="section-header">Top 10 Products by Demand</p>', unsafe_allow_html=True)
        top_prod = (df_filt.groupby("product_id")["demand"].sum()
                    .nlargest(10).reset_index())
        fig4 = px.bar(top_prod, x="demand", y="product_id", orientation="h",
                      color="demand", color_continuous_scale="Blues",
                      labels={"demand": "Total Demand", "product_id": "Product"})
        fig4.update_layout(height=280, coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – INVENTORY RISK
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Inventory Risk":
    st.title("⚠️ Inventory Risk")

    if not data_ready:
        st.warning("No processed data found. Click **Re-run Pipeline** in the sidebar.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Stock-out Heatmap (Store × Product)</p>', unsafe_allow_html=True)
        if "inventory_level" in master.columns:
            master["stockout"] = (master["inventory_level"] < master["demand"]).astype(int)
            heat = master.groupby(["store_id", "product_id"])["stockout"].mean().reset_index()
            heat_pivot = heat.pivot(index="product_id", columns="store_id", values="stockout").fillna(0)
            fig = px.imshow(heat_pivot, color_continuous_scale="RdYlGn_r",
                            labels=dict(color="Stockout Rate"),
                            aspect="auto")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Inventory data not available.")

    with col2:
        if not stockout_preds.empty:
            st.markdown('<p class="section-header">Stockout Probability Distribution</p>', unsafe_allow_html=True)
            fig2 = px.histogram(stockout_preds, x="stockout_prob", nbins=30,
                                color_discrete_sequence=["#2d6a9f"],
                                labels={"stockout_prob": "Stockout Probability"})
            fig2.add_vline(x=0.40, line_dash="dash", line_color="orange", annotation_text="Medium")
            fig2.add_vline(x=0.70, line_dash="dash", line_color="red", annotation_text="High")
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Train stockout model to see probability distribution.")

    if not recs.empty:
        st.markdown('<p class="section-header">Risk Summary Table</p>', unsafe_allow_html=True)
        risk_filter = st.multiselect("Filter by Risk Level", ["HIGH", "MEDIUM", "LOW"],
                                     default=["HIGH", "MEDIUM"])
        disp = recs[recs["risk_level"].isin(risk_filter)].copy()

        def color_risk(val):
            colors = {"HIGH": "background-color:#ff4b4b;color:white",
                      "MEDIUM": "background-color:#ffa500;color:white",
                      "LOW": "background-color:#21c354;color:white"}
            return colors.get(val, "")

        styled = disp[["store_id", "product_id", "inventory_level",
                        "next_7_day_demand", "stockout_prob", "risk_level", "reorder_qty"]].style.applymap(
            color_risk, subset=["risk_level"]
        ).format({"stockout_prob": "{:.1%}", "next_7_day_demand": "{:.0f}"})
        st.dataframe(styled, use_container_width=True, height=400)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – MANAGER ACTION CENTRE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Manager Action Centre":
    st.title("🎯 Manager Action Centre")
    st.caption("Recommended replenishment actions — sorted by urgency")

    if recs.empty:
        st.warning("No recommendations found. Click **Re-run Pipeline** in the sidebar.")
        st.stop()

    # Summary cards
    n_high = (recs["risk_level"] == "HIGH").sum()
    n_med = (recs["risk_level"] == "MEDIUM").sum()
    n_low = (recs["risk_level"] == "LOW").sum()
    total_reorder = recs["reorder_qty"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 HIGH Risk", n_high)
    c2.metric("🟡 MEDIUM Risk", n_med)
    c3.metric("🟢 LOW Risk", n_low)
    c4.metric("📦 Total Units to Order", f"{total_reorder:,}")

    st.divider()

    # Recommendation cards for HIGH risk
    st.markdown('<p class="section-header">🔴 Immediate Action Required</p>', unsafe_allow_html=True)
    high_risk = recs[recs["risk_level"] == "HIGH"].head(10)

    if high_risk.empty:
        st.success("No HIGH risk items currently.")
    else:
        for _, row in high_risk.iterrows():
            with st.expander(
                f"🏪 {row['store_id']}  |  📦 {row['product_id']}  |  "
                f"Stockout Prob: {row['stockout_prob']:.0%}  |  Reorder: {row['reorder_qty']} units"
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Stock", int(row["inventory_level"]))
                c2.metric("7-Day Forecast", f"{row['next_7_day_demand']:.0f}")
                c3.metric("Stockout Prob", f"{row['stockout_prob']:.0%}")
                c4.metric("Recommended Order", row["reorder_qty"])
                st.markdown(f"**Key Drivers:** {row.get('top_drivers', '—')}")
                st.error(row.get("action", "RAISE REPLENISHMENT ORDER TODAY"))

    st.divider()
    st.markdown('<p class="section-header">Full Recommendation Table</p>', unsafe_allow_html=True)

    store_filter = st.multiselect("Filter by Store", sorted(recs["store_id"].unique()),
                                  default=sorted(recs["store_id"].unique()))
    risk_filter2 = st.multiselect("Filter by Risk", ["HIGH", "MEDIUM", "LOW"],
                                  default=["HIGH", "MEDIUM", "LOW"])

    disp = recs[recs["store_id"].isin(store_filter) & recs["risk_level"].isin(risk_filter2)]
    st.dataframe(
        disp[["store_id", "product_id", "inventory_level", "next_7_day_demand",
              "stockout_prob", "risk_level", "reorder_qty", "action"]],
        use_container_width=True, height=450
    )

    csv = disp.to_csv(index=False).encode()
    st.download_button("⬇️ Download Recommendations CSV", csv,
                       "recommendations.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    tab1, tab2 = st.tabs(["Demand Forecasting", "Stockout Classification"])

    with tab1:
        st.markdown('<p class="section-header">Demand Model Metrics</p>', unsafe_allow_html=True)
        if not demand_preds.empty:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae = mean_absolute_error(demand_preds["demand"], demand_preds["predicted_demand"])
            rmse = np.sqrt(mean_squared_error(demand_preds["demand"], demand_preds["predicted_demand"]))
            r2 = r2_score(demand_preds["demand"], demand_preds["predicted_demand"])
            mape = np.mean(np.abs((demand_preds["demand"] - demand_preds["predicted_demand"]) /
                                  (demand_preds["demand"] + 1e-9))) * 100

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"{mae:.2f}")
            c2.metric("RMSE", f"{rmse:.2f}")
            c3.metric("R²", f"{r2:.4f}")
            c4.metric("MAPE", f"{mape:.1f}%")

            st.markdown("**Business Interpretation:** MAE tells managers the average error in units. "
                        "R² shows how much variance is explained. MAPE gives percentage accuracy.")

            # Residual plot
            demand_preds["residual"] = demand_preds["demand"] - demand_preds["predicted_demand"]
            fig = px.scatter(demand_preds.sample(min(2000, len(demand_preds))),
                             x="predicted_demand", y="residual",
                             opacity=0.4, color_discrete_sequence=["#2d6a9f"],
                             labels={"predicted_demand": "Predicted", "residual": "Residual"})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(title="Residual Plot", height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train demand model to see metrics.")

    with tab2:
        st.markdown('<p class="section-header">Stockout Classification Metrics</p>', unsafe_allow_html=True)
        if not stockout_preds.empty:
            from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                         f1_score, roc_auc_score, confusion_matrix)
            y_true = stockout_preds["stockout_flag"]
            y_pred = stockout_preds["stockout_pred"]
            y_prob = stockout_preds["stockout_prob"]

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            auc = roc_auc_score(y_true, y_prob)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{acc:.3f}")
            c2.metric("Precision", f"{prec:.3f}")
            c3.metric("Recall", f"{rec:.3f}")
            c4.metric("F1-Score", f"{f1:.3f}")
            c5.metric("ROC-AUC", f"{auc:.3f}")

            st.markdown("**Business Interpretation:** High Recall is critical — we must catch most real stockouts. "
                        "ROC-AUC > 0.8 indicates a reliable classifier.")

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = px.imshow(cm, text_auto=True,
                               labels=dict(x="Predicted", y="Actual"),
                               x=["No Stockout", "Stockout"],
                               y=["No Stockout", "Stockout"],
                               color_continuous_scale="Blues")
            fig_cm.update_layout(title="Confusion Matrix", height=350)
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Train stockout model to see metrics.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 – EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Explainability":
    st.title("🔍 Explainability")
    st.caption("Why is a product at risk? Feature importance from trained models.")

    from explainability import get_demand_feature_importance, get_stockout_feature_importance

    tab1, tab2 = st.tabs(["Demand Model", "Stockout Model"])

    with tab1:
        imp_d = get_demand_feature_importance()
        if not imp_d.empty:
            fig = px.bar(imp_d, x="importance", y="label", orientation="h",
                         color="importance", color_continuous_scale="Blues",
                         labels={"importance": "Importance Score", "label": "Feature"})
            fig.update_layout(height=450, coloraxis_showscale=False,
                               yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Top driver:** The most important feature for demand forecasting is "
                        f"**{imp_d.iloc[0]['label']}** with importance score "
                        f"{imp_d.iloc[0]['importance']:.4f}.")
        else:
            st.info("Train demand model to see feature importance.")

    with tab2:
        imp_s = get_stockout_feature_importance()
        if not imp_s.empty:
            fig2 = px.bar(imp_s, x="importance", y="label", orientation="h",
                          color="importance", color_continuous_scale="Reds",
                          labels={"importance": "Importance Score", "label": "Feature"})
            fig2.update_layout(height=450, coloraxis_showscale=False,
                                yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("**Top driver:** The most important feature for stockout prediction is "
                        f"**{imp_s.iloc[0]['label']}** with importance score "
                        f"{imp_s.iloc[0]['importance']:.4f}.")

            # Example explanation card
            if not recs.empty:
                st.divider()
                st.markdown('<p class="section-header">Sample Explanation Card</p>', unsafe_allow_html=True)
                high = recs[recs["risk_level"] == "HIGH"]
                if not high.empty:
                    row = high.iloc[0]
                    st.info(
                        f"**STORE {row['store_id']} — PRODUCT {row['product_id']}**\n\n"
                        f"Predicted 7-day demand: **{row['next_7_day_demand']:.0f} units**\n\n"
                        f"Current stock: **{int(row['inventory_level'])} units**\n\n"
                        f"Stock-out probability: **{row['stockout_prob']:.0%}** | Risk: **{row['risk_level']}**\n\n"
                        f"Recommended order: **{row['reorder_qty']} units**\n\n"
                        f"**Why?** {row.get('top_drivers', '—')}\n\n"
                        f"**MANAGER ACTION:** {row.get('action', '')}"
                    )
        else:
            st.info("Train stockout model to see feature importance.")
