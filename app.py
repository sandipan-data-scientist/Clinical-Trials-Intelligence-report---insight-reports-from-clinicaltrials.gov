"""
app.py
Clinical Trial Sponsor Intelligence Engine
Streamlit application for share.streamlit.io deployment.

Run locally:  streamlit run app.py
Deploy:       Push to GitHub, connect to share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# local module imports
from src.data_loader      import DataLoader
from src.preprocessor     import DataPreprocessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer    import ModelTrainer
from src.visualizer       import Visualizer
from src.sponsor_dashboard import SponsorDashboard


# page config must be the first Streamlit call
st.set_page_config(
    page_title="Clinical Trial Intelligence Engine",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# custom CSS - keeps the app clean and readable
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { color: #2E86AB; font-weight: 800; }
    h2 { color: #1A1A2E; border-bottom: 2px solid #2E86AB; padding-bottom: 6px; }
    h3 { color: #2E86AB; }
    .metric-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-card .value { font-size: 2rem; font-weight: 800; color: #2E86AB; }
    .metric-card .label { font-size: 0.82rem; color: #666; margin-top: 4px; }
    .chart-caption {
        background: #F0F4F8;
        border-left: 4px solid #2E86AB;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
        color: #444;
        margin-bottom: 16px;
    }
    .section-intro {
        background: #FFFFFF;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.92rem;
        color: #333;
        margin-bottom: 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)


# data pipeline - cached so it only runs once per session
@st.cache_data(show_spinner=False)
def load_and_clean_data():
    loader = DataLoader()
    df_raw = loader.load()
    preprocessor = DataPreprocessor(df_raw)
    df_clean = preprocessor.run()
    return df_clean


@st.cache_resource(show_spinner=False)
def train_model(_df_clean):
    """
    Trains the GradientBoosting model. Cached as a resource so
    the model object is reused across reruns without retraining.
    """
    engineer = FeatureEngineer(df_clean)
    df_feat  = engineer.run()
    le_int, le_dom = engineer.get_encoders()
    feature_cols   = engineer.get_feature_cols()

    trainer = ModelTrainer(df_feat, feature_cols)
    results = trainer.train_all()

    # score every trial with the trained model
    X_all    = df_feat[feature_cols].fillna(0).values
    df_feat["Predicted_Failure_Prob"] = trainer.score_full_dataset(X_all)

    def risk_label(p):
        if p < 0.20:   return "Low Risk"
        elif p < 0.40: return "Moderate Risk"
        return "High Risk"

    df_feat["Risk_Level"] = df_feat["Predicted_Failure_Prob"].apply(risk_label)
    return df_feat, trainer, results, le_int, le_dom, feature_cols


# load data with a progress indicator
with st.spinner("Loading and processing clinical trials data..."):
    try:
        df_clean = load_and_clean_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

with st.spinner("Training predictive model..."):
    df, trainer, model_results, le_int, le_dom, feature_cols = train_model(df_clean)

# shared objects
viz = Visualizer(df)


# sidebar navigation
with st.sidebar:
    st.image(
        "https://crir.ca/wp-content/uploads/2020/05/nihclinicaltrials.png",
        use_container_width=True
    )
    st.markdown("### Navigation")
    section = st.radio(
        "Jump to section",
        options=[
            "Project Overview",
            "Dataset Summary",
            "Trial Landscape",
            "Enrollment Analysis",
            "Failure Analysis",
            "Competitive Landscape",
            "Domain Trends and Forecasts",
            "ML Model Results",
            "Volume Forecast",
            "Sponsor Dashboard",
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(f"**Trials loaded:** {len(df):,}")
    st.markdown(f"**Unique sponsors:** {df['Sponsor'].nunique():,}")
    st.markdown(f"**Years covered:** 2005 - 2025")
    st.markdown("---")
    st.caption("Built by Sandipan Acharjee  |  Data: ClinicalTrials.gov")


# helper functions

def chart_caption(text: str):
    st.markdown(f'<div class="chart-caption">{text}</div>', unsafe_allow_html=True)

def section_intro(text: str):
    st.markdown(f'<div class="section-intro">{text}</div>', unsafe_allow_html=True)

def metric_card(col, value, label):
    col.markdown(
        f'<div class="metric-card"><div class="value">{value}</div>'
        f'<div class="label">{label}</div></div>',
        unsafe_allow_html=True
    )


# main content

if section == "Project Overview":
    st.title("Clinical Trial Sponsor Intelligence Engine")
    st.markdown("#### End-to-End Data Science Platform  |  ClinicalTrials.gov 2005-2025")

    st.markdown("""
    <div class="section-intro">
    This platform analyses <strong>3,792 clinical trials</strong> across <strong>957 sponsors</strong>
    spanning 20 years of global pharmaceutical R&D activity sourced from ClinicalTrials.gov.
    It delivers sponsor-level failure intelligence, therapeutic domain forecasting, competitive
    landscaping, and a machine learning model that predicts trial termination risk. Use the sidebar 
    to navigate through each analysis section.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    metric_card(c1, f"{len(df):,}", "Total Trials")
    metric_card(c2, f"{df['Sponsor'].nunique():,}", "Unique Sponsors")
    metric_card(c3, f"{df['Medical_Domain'].nunique()}", "Therapeutic Domains")
    metric_card(c4, f"{(df['Status_Clean']=='Terminated').sum():,}", "Terminated Trials")
    metric_card(
        c5,
        f"{(df[df['Status_Clean'].isin(['Completed','Terminated'])]['Is_Failed'].mean()*100):.1f}%",
        "Overall Failure Rate"
    )

    st.markdown("### What This Platform Does")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        **Sponsor Intelligence**
        Profiles each of 957 sponsors across trial volume, phase distribution, failure rate,
        enrollment patterns, and therapeutic focus.
        """)
    with col_b:
        st.markdown("""
        **ML Risk Prediction**
        A XGBoost classifier trained on phase, duration, sponsor history, and funder
        signals predicts trial termination probability per trial.
        """)
    with col_c:
        st.markdown("""
        **Strategic Forecasting**
        Linear trend models project trial volume and domain enrollment out to 2045, surfacing
        which therapeutic areas are blooming.
        """)


elif section == "Dataset Summary":
    st.title("Dataset Summary")
    section_intro(
        "A first look at the 23-column dataset: column types, missingness patterns, "
        "and the key distributions that shape every downstream analysis."
    )

    st.markdown("### Column Overview")
    col_info = pd.DataFrame({
        "Column": df.columns.tolist(),
        "Type":   df.dtypes.astype(str).tolist(),
        "Non-Null": df.notnull().sum().tolist(),
        "Missing %": (df.isnull().sum() / len(df) * 100).round(1).tolist(),
    })
    st.dataframe(col_info, use_container_width=True, height=400)

    st.markdown("### Missingness Report")
    missing = (df.isnull().sum() / len(df) * 100).round(2)
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        st.success("No missing values remain after preprocessing.")
    else:
        fig_miss, ax_miss = plt.subplots(figsize=(10, 4))
        ax_miss.barh(missing.index, missing.values, color="#2E86AB", edgecolor="white")
        ax_miss.set_xlabel("Missing (%)")
        ax_miss.set_title("Remaining Missing Values After Cleaning", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_miss, use_container_width=True)
        plt.close(fig_miss)

    chart_caption(
        "Collaborators is the only column with significant missingness and this is expected: "
        "the majority of trials are run by a single sponsor with no external collaborator listed."
    )

    st.markdown("### Sample Records")
    st.dataframe(df.head(5), use_container_width=True)


elif section == "Trial Landscape":
    st.title("Trial Landscape (2005-2025)")
    section_intro(
        "How has clinical trial activity evolved over two decades? "
        "This section shows the year-by-year growth in trial initiation and how "
        "the phase composition of the industry has shifted over time."
    )

    fig = viz.fig_trial_landscape()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    chart_caption(
        "Left: Total trials initiated per year. The post-2020 surge reflects COVID-era "
        "urgency and the accelerated regulatory environment. Right: The stacked area chart "
        "shows that Phase 2 trials dominate the pipeline, while Phase 4 post-market studies "
        "have grown proportionally in recent years."
    )

    # year-over-year change
    st.markdown("### Year-over-Year Trial Volume Change")
    yoy = df.groupby("Start_Year").size().pct_change() * 100
    fig_yoy, ax_yoy = plt.subplots(figsize=(12, 4))
    colors_yoy = ["#3BB273" if v >= 0 else "#E84855" for v in yoy.values]
    ax_yoy.bar(yoy.index, yoy.values, color=colors_yoy, edgecolor="white")
    ax_yoy.axhline(0, color="black", lw=0.8)
    ax_yoy.set_title("Year-over-Year Change in Trial Volume (%)", fontweight="bold")
    ax_yoy.set_xlabel("Year")
    ax_yoy.set_ylabel("YoY Change (%)")
    plt.tight_layout()
    st.pyplot(fig_yoy, use_container_width=True)
    plt.close(fig_yoy)
    chart_caption(
        "A sharp positive spike appears in 2020-2021 driven by COVID vaccine and treatment "
        "trials. Years with negative change reflect normal cyclical slowdowns in new trial starts."
    )


elif section == "Enrollment Analysis":
    st.title("Enrollment Analysis")
    section_intro(
        "Enrollment size is a direct proxy for R&D investment scale. "
        "This section examines the extreme skewness of enrollment data, "
        "why the mean misleads, and how enrollment patterns differ across phases and domains."
    )

    fig = viz.fig_enrollment_distribution()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    raw_skew = df["Enrollment"].skew()
    log_skew = df["Log_Enrollment"].skew()
    chart_caption(
        f"Raw enrollment skewness: {raw_skew:.1f}. A small number of mega-trials with tens of "
        f"thousands of participants pull the mean far above the median. Log transformation "
        f"reduces skewness to {log_skew:.1f}, making statistical analysis valid. "
        "All downstream modeling uses log-transformed enrollment."
    )

    st.markdown("### Enrollment by Phase")
    fig_ep, ax_ep = plt.subplots(figsize=(10, 5))
    phase_order = ["Phase 1","Phase 1/2","Phase 2","Phase 2/3","Phase 3","Phase 4"]
    phase_data  = [
        df[df["Phase_Clean"] == ph]["Enrollment"].dropna().values
        for ph in phase_order if ph in df["Phase_Clean"].values
    ]
    present_phases = [ph for ph in phase_order if ph in df["Phase_Clean"].values]
    ax_ep.boxplot(phase_data, labels=present_phases, patch_artist=True,
                  boxprops=dict(facecolor="#AED9E0"),
                  medianprops=dict(color="#E84855", lw=2))
    ax_ep.set_title("Enrollment Distribution by Phase", fontweight="bold")
    ax_ep.set_ylabel("Enrollment (raw)")
    ax_ep.set_ylim(0, df["Enrollment"].quantile(0.95))
    plt.tight_layout()
    st.pyplot(fig_ep, use_container_width=True)
    plt.close(fig_ep)
    chart_caption(
        "Phase 3 trials have the widest enrollment range: they must be large enough "
        "for statistical significance across primary endpoints, but sponsor budgets and "
        "disease prevalence constrain the upper bound."
    )


elif section == "Failure Analysis":
    st.title("Trial Failure Analysis")
    section_intro(
        "Failure here means a trial was officially terminated before completion. "
        "Understanding where and why trials fail is the core intelligence question "
        "for any R&D strategy team. Historically Phase 2 carries the highest attrition "
        "as it is the first real test of efficacy. Let us see how this dataset compares."
    )

    fig = viz.fig_failure_by_phase()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    chart_caption(
        "Red bars indicate phases where failure rate exceeds 20%, which we use as the "
        "industry risk threshold. The dashed line is the overall average failure rate across "
        "all phases. Phase 2 typically shows higher attrition because it transitions from "
        "safety-focused (Phase 1) to efficacy-focused trials."
    )

    st.markdown("### Failure Rate by Medical Domain")
    df_out = df[df["Status_Clean"].isin(["Completed", "Terminated"])]
    dom_fail = df_out.groupby("Medical_Domain")["Is_Failed"].agg(
        Total="count", Failed="sum"
    ).reset_index()
    dom_fail["Failure_Rate"] = (dom_fail["Failed"] / dom_fail["Total"] * 100).round(1)
    dom_fail = dom_fail.sort_values("Failure_Rate", ascending=True)

    fig_df, ax_df = plt.subplots(figsize=(10, 5))
    colors_df = ["#E84855" if r > 20 else "#3BB273" for r in dom_fail["Failure_Rate"]]
    bars = ax_df.barh(dom_fail["Medical_Domain"], dom_fail["Failure_Rate"],
                      color=colors_df, edgecolor="white")
    for bar, rate in zip(bars, dom_fail["Failure_Rate"]):
        ax_df.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                   f"{rate}%", va="center", fontsize=9)
    ax_df.axvline(df_out["Is_Failed"].mean()*100, color="black", ls="--", lw=1)
    ax_df.set_title("Trial Failure Rate by Medical Domain", fontweight="bold")
    ax_df.set_xlabel("Failure Rate (%)")
    plt.tight_layout()
    st.pyplot(fig_df, use_container_width=True)
    plt.close(fig_df)
    chart_caption(
        "Domains with complex endpoints or heterogeneous patient populations tend to have "
        "higher failure rates. Neurological disorders often show the highest attrition because "
        "primary endpoints like cognitive improvement are difficult to demonstrate statistically."
    )

    # correlation heatmap
    st.markdown("### Feature Correlation with Trial Failure")
    fig_corr = viz.fig_correlation()
    st.pyplot(fig_corr, use_container_width=True)
    plt.close(fig_corr)
    chart_caption(
        "Pearson correlations between numeric features. Values near +1 indicate strong positive "
        "association, -1 indicates strong negative. Most correlations with Is_Failed are weak, "
        "which is expected: trial failure is a multi-causal phenomenon. This is why we use "
        "a non-linear Gradient Boosting model rather than simple regression."
    )


elif section == "Competitive Landscape":
    st.title("Competitive Landscape")
    section_intro(
        "Who are the most active sponsors? Who has the worst failure record? "
        "And how do the top players compare on enrollment scale and trial duration? "
        "This section provides a side-by-side view of the top 10 sponsors by volume."
    )

    top_n = st.slider("Number of top sponsors to compare", 5, 20, 10)

    fig_sl = viz.fig_sponsor_landscape(top_n=top_n)
    st.pyplot(fig_sl, use_container_width=True)
    plt.close(fig_sl)
    chart_caption(
        "Left: Failure rate per sponsor sorted ascending. Red bars exceed the 20% risk threshold. "
        "Right: Each bubble represents a sponsor. X-axis is total trial volume, Y-axis is failure "
        "rate. Bubble size encodes average enrollment. Sponsors in the top-right quadrant run many "
        "large trials with high failure rates, representing the highest R&D risk concentration."
    )

    st.markdown("### Full Competitive Dashboard")
    section_intro(
        "This consolidated panel brings together trial volume, phase composition, enrollment "
        "distributions, duration benchmarks, funder mix, and domain growth signals in a single "
        "comparative view."
    )
    fig_cd = viz.fig_competitive_dashboard(top_n=top_n)
    st.pyplot(fig_cd, use_container_width=True)
    plt.close(fig_cd)


elif section == "Domain Trends and Forecasts":
    st.title("Therapeutic Domain Trends and Forecasts")
    section_intro(
        "Which therapeutic areas are growing fastest? Which are plateauing or contracting? "
        "This section uses Compound Annual Growth Rate (CAGR) as the ranking signal and "
        "applies linear regression to forecast domain-level trial volume. "
        "CAGR = (End Value / Start Value)^(1/years) - 1."
    )

    fig_dt = viz.fig_domain_trends()
    st.pyplot(fig_dt, use_container_width=True)
    plt.close(fig_dt)
    chart_caption(
        "Each line is a therapeutic domain. Rapid growth post-2020 in Infectious disease "
        "is driven almost entirely by COVID-19 trials. Oncology shows a consistent upward "
        "trajectory reflecting the sustained industry investment in cancer drug development."
    )

    fig_cagr = viz.fig_domain_cagr()
    st.pyplot(fig_cagr, use_container_width=True)
    plt.close(fig_cagr)
    chart_caption(
        "CAGR calculated over 2015-2024 for each domain. Green bars indicate growing areas "
        "where new trial initiation is accelerating. Red bars indicate areas where fewer new "
        "trials are being started, which could signal market saturation or pipeline depletion."
    )

    st.markdown("### Domain-Level Enrollment Forecast to 2045")
    df_d = df.copy()
    from sklearn.linear_model import LinearRegression
    domains = sorted(df_d["Medical_Domain"].unique())
    selected_domains = st.multiselect("Select domains to forecast", domains, default=domains[:4])
    if selected_domains:
        fig_fore, axes = plt.subplots(
            len(selected_domains), 1,
            figsize=(14, 4 * len(selected_domains))
        )
        if len(selected_domains) == 1:
            axes = [axes]
        for ax, dom in zip(axes, selected_domains):
            d_data = df_d[df_d["Medical_Domain"] == dom].groupby("Start_Year")["Enrollment"].median()
            d_data = d_data[d_data.index <= 2024]
            if len(d_data) < 3:
                ax.text(0.5, 0.5, f"{dom}: insufficient data",
                        ha="center", va="center", transform=ax.transAxes)
                continue
            t_ = d_data.index.values.reshape(-1, 1)
            y_ = d_data.values
            lr_ = LinearRegression().fit(t_, y_)
            future_ = np.arange(2025, 2046).reshape(-1, 1)
            preds_ = np.maximum(lr_.predict(future_), 0)
            std_   = (y_ - lr_.predict(t_)).std()
            ax.plot(d_data.index, d_data.values, "o-", color="#2E86AB", lw=2, label="Actual")
            ax.plot(future_.flatten(), preds_, "r--", lw=2, label="Forecast")
            ax.fill_between(future_.flatten(), preds_ - 1.96*std_, preds_ + 1.96*std_,
                            alpha=0.15, color="red", label="95% CI")
            ax.set_title(f"{dom}: Median Enrollment Forecast", fontweight="bold", fontsize=11)
            ax.set_xlabel("Year", fontsize=8)
            ax.set_ylabel("Median Enrollment", fontsize=8)
            ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_fore, use_container_width=True)
        plt.close(fig_fore)
        chart_caption(
            "Forecasts use Ordinary Least Squares (OLS) linear regression. The shaded band "
            "is a 95% confidence interval calculated from residual standard deviation. "
            "Treat these as directional signals, not exact predictions."
        )


elif section == "ML Model Results":
    st.title("ML Model: Trial Failure Prediction")
    section_intro(
        "A binary classification model (Logistic Regression) trained to predict whether a trial will be terminated "
        "before completion. We compare three classifiers and select the best by AUC-ROC, "
        "which correctly handles class imbalance. The winning model scores every trial in the "
        "dataset with a failure probability."
    )

    st.markdown("### Model Comparison")
    results_df = pd.DataFrame(model_results).T.reset_index()
    results_df.columns = ["Model", "AUC", "Accuracy", "Precision", "Recall", "F1"]
    results_df = results_df.sort_values("AUC", ascending=False)
    st.dataframe(
        results_df.style
            .highlight_max(subset=["AUC","Recall","F1"], color="#d4edda")
            .format(precision=3),
        use_container_width=True
    )
    chart_caption(
        "AUC-ROC is the primary selection metric. An AUC of 0.5 is equivalent to random guessing; "
        "1.0 is a perfect classifier. Recall on the failure class tells us how many actual "
        "terminations the model correctly flags - critical for risk management applications."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ROC Curves")
        roc_data = trainer.get_roc_data()
        fig_roc = viz.fig_roc_curves(roc_data)
        st.pyplot(fig_roc, use_container_width=True)
        plt.close(fig_roc)

    with col2:
        st.markdown("### Confusion Matrix (Best Model)")
        cm = trainer.get_confusion_matrix()
        fig_cm = viz.fig_confusion_matrix(cm, trainer.best_model_name)
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

    chart_caption(
        "ROC curve: each point represents a different decision threshold. A curve that hugs "
        "the top-left corner is a better classifier. The confusion matrix shows the breakdown "
        "of correct and incorrect predictions for the best model on the held-out test set."
    )

    st.markdown("### Feature Importance")
    imp = trainer.get_feature_importances()
    fig_imp = viz.fig_feature_importance(imp)
    st.pyplot(fig_imp, use_container_width=True)
    plt.close(fig_imp)
    chart_caption(
        "XGBoost feature importances measure how much each variable reduces "
        "prediction error across all decision trees. Sponsor historical failure rate and "
        "phase rank consistently rank highest, confirming that who runs the trial and at what "
        "stage matters more than the disease area or enrollment size."
    )

    st.markdown("### Risk Score Distribution Across All Trials")
    fig_risk, ax_risk = plt.subplots(figsize=(12, 4))
    ax_risk.hist(df["Predicted_Failure_Prob"].dropna(), bins=40,
                 color="#9B5DE5", edgecolor="white", lw=0.5)
    ax_risk.axvline(0.20, color="#FAA916", ls="--", lw=1.5, label="Low/Moderate threshold (20%)")
    ax_risk.axvline(0.40, color="#E84855", ls="--", lw=1.5, label="Moderate/High threshold (40%)")
    ax_risk.set_title("Distribution of Predicted Failure Probability (All Trials)",
                      fontweight="bold")
    ax_risk.set_xlabel("Predicted Failure Probability")
    ax_risk.set_ylabel("Trial Count")
    ax_risk.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_risk, use_container_width=True)
    plt.close(fig_risk)


elif section == "Volume Forecast":
    st.title("Clinical Trial Volume Forecast: 2025-2045")
    section_intro(
        "Using the 20-year historical trend as training data, a linear regression model "
        "projects total trial volume through 2045. The slope of the trend line tells us "
        "how many new trials the industry adds per year on average. The shaded band is a "
        "95% confidence interval based on historical residuals."
    )

    fig_vf = viz.fig_volume_forecast()
    st.pyplot(fig_vf, use_container_width=True)
    plt.close(fig_vf)

    yearly = df.groupby("Start_Year").size().reset_index(name="Count")
    yearly = yearly[yearly["Start_Year"] <= 2024]
    from sklearn.linear_model import LinearRegression
    t = yearly["Start_Year"].values.reshape(-1, 1)
    y = yearly["Count"].values
    lr_v = LinearRegression().fit(t, y)
    slope = lr_v.coef_[0]
    pred_2030 = max(lr_v.predict([[2030]])[0], 0)
    pred_2040 = max(lr_v.predict([[2040]])[0], 0)

    chart_caption(
        f"The model adds approximately {slope:.1f} new trial starts per year. "
        f"Projected 2030 volume: {pred_2030:.0f} trials. "
        f"Projected 2040 volume: {pred_2040:.0f} trials. "
        "These projections assume historical growth patterns continue without major "
        "regulatory or macroeconomic disruptions."
    )

    c1, c2, c3 = st.columns(3)
    metric_card(c1, f"{slope:.1f}", "New Trials Added Per Year (trend)")
    metric_card(c2, f"{pred_2030:.0f}", "Projected Trials in 2030")
    metric_card(c3, f"{pred_2040:.0f}", "Projected Trials in 2040")


elif section == "Sponsor Dashboard":
    st.title("Sponsor Intelligence Dashboard")
    section_intro(
        "Select any of the 957 sponsors in the dataset to generate a fully personalised "
        "intelligence report powered by the trained Gradient Boosting model. "
        "The dashboard includes failure analysis, domain growth trends, age and sex group "
        "distributions, enrollment patterns, risk group breakdown, and a table of the "
        "highest-risk trials in that sponsor's active portfolio. "
        "Download the complete report as a PNG."
    )

    all_sponsors = sorted(df["Sponsor"].dropna().unique().tolist())
    sponsor_name = st.selectbox(
        "Select a Sponsor",
        options=all_sponsors,
        index=all_sponsors.index("Pfizer") if "Pfizer" in all_sponsors else 0
    )

    run_btn = st.button("Generate Dashboard", type="primary", use_container_width=True)

    if run_btn:
        dash = SponsorDashboard(df, sponsor_name)
        if not dash.is_valid():
            st.error(f"No data found for '{sponsor_name}'. Please select a different sponsor.")
        else:
            summary = dash.get_summary()

            # summary metrics row
            st.markdown(f"### {sponsor_name}")
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            metric_card(c1, summary["Total Trials"],         "Total Trials")
            metric_card(c2, summary["Completed"],            "Completed")
            metric_card(c3, summary["Terminated"],           "Terminated")
            metric_card(c4, f"{summary['Failure Rate']}%",  "Failure Rate")
            metric_card(c5, f"{summary['Model Avg Risk']}%","Model Risk Score")
            metric_card(c6, summary["High Risk Trials"],     "High Risk Trials")

            # build the full dashboard figure
            with st.spinner(f"Building full dashboard for {sponsor_name}..."):
                fig_dash = dash.build()

            st.pyplot(fig_dash, use_container_width=True)

            # download button
            png_bytes = SponsorDashboard.to_png_bytes(fig_dash)
            clean_name = sponsor_name.replace(" ", "_").replace("/", "-").replace(",", "")
            st.download_button(
                label="Download Dashboard as PNG",
                data=png_bytes,
                file_name=f"{clean_name}_intelligence_dashboard.png",
                mime="image/png",
                use_container_width=True,
            )
            plt.close(fig_dash)

            # detailed data table below the dashboard
            st.markdown("### Trial-Level Data")
            display_cols = [
                "NCT Number", "Phase_Clean", "Medical_Domain", "Status_Clean",
                "Enrollment", "Duration_Months", "Sex_Clean", "Age_Group_Label",
                "Predicted_Failure_Prob", "Risk_Level"
            ]
            avail_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(
                df[df["Sponsor"] == sponsor_name][avail_cols]
                .sort_values("Predicted_Failure_Prob", ascending=False)
                .reset_index(drop=True),
                use_container_width=True,
                height=350
            )