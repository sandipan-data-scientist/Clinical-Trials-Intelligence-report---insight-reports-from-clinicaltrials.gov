import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import warnings

warnings.filterwarnings("ignore")

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


# step 1: load all saved model artifacts
print("Loading model artifacts...")
model          = joblib.load("model_gb.pkl")
le_intervention = joblib.load("le_intervention.pkl")
le_domain       = joblib.load("le_domain.pkl")
feature_cols    = joblib.load("feature_cols.pkl")
print("Artifacts loaded successfully.")
print(f"Features expected by model: {feature_cols}")

# step 2: load and rebuild the cleaned dataframe
# this replicates the cleaning pipeline from the main notebook
# so the dashboard is fully self-contained and portable

df_raw = pd.read_csv("raw_ct_data.csv")
df_raw["Collaborators"] = df_raw["Collaborators"].fillna("None")
df = df_raw.dropna(subset=["Sponsor", "Phases", "Start Date", "Completion Date",
                             "Enrollment", "Funder Type", "Study Type"]).copy()

df["Start Date"]      = pd.to_datetime(df["Start Date"], dayfirst=True, errors="coerce")
df["Completion Date"] = pd.to_datetime(df["Completion Date"], dayfirst=True, errors="coerce")
df["Duration_Months"] = ((df["Completion Date"] - df["Start Date"]).dt.days / 30.44).round(1)
df = df[df["Duration_Months"] > 0].copy()
df["Start_Year"]      = df["Start Date"].dt.year
df["Log_Enrollment"]  = np.log1p(df["Enrollment"])

# phase mapping
phase_map = {
    "PHASE1": "Phase 1", "EARLY_PHASE1": "Phase 1",
    "PHASE1|PHASE2": "Phase 1/2", "PHASE2": "Phase 2",
    "PHASE2|PHASE3": "Phase 2/3", "PHASE3": "Phase 3", "PHASE4": "Phase 4",
}
df["Phase_Clean"] = df["Phases"].map(phase_map).fillna("Unknown")

phase_rank_map = {
    "Phase 1": 1, "Phase 1/2": 1.5, "Phase 2": 2,
    "Phase 2/3": 2.5, "Phase 3": 3, "Phase 4": 4, "Unknown": 0
}
df["Phase_Rank"] = df["Phase_Clean"].map(phase_rank_map)

status_map = {
    "COMPLETED": "Completed", "TERMINATED": "Terminated",
    "ACTIVE_NOT_RECRUITING": "Active", "UNKNOWN": "Unknown"
}
df["Status_Clean"] = df["Study Status"].map(status_map)
df["Is_Failed"]    = df["Status_Clean"].map({"Terminated": 1, "Completed": 0})

def extract_intervention_type(text):
    if pd.isna(text):
        return "OTHER"
    text = str(text).upper()
    for itype in ["BIOLOGICAL", "DRUG", "DEVICE", "PROCEDURE", "BEHAVIORAL", "DIETARY"]:
        if itype in text:
            return itype
    return "OTHER"

def extract_masking(design):
    if pd.isna(design):
        return "NONE"
    design = str(design).upper()
    if "QUADRUPLE" in design: return "QUADRUPLE"
    elif "TRIPLE" in design:  return "TRIPLE"
    elif "DOUBLE" in design:  return "DOUBLE"
    elif "SINGLE" in design:  return "SINGLE"
    else:                     return "NONE"

def extract_allocation(design):
    if pd.isna(design):
        return "UNKNOWN"
    design = str(design).upper()
    if "RANDOMIZED" in design and "NON_RANDOMIZED" not in design: return "RANDOMIZED"
    elif "NON_RANDOMIZED" in design: return "NON_RANDOMIZED"
    else: return "NA"

DOMAIN_MAP = {
    "Oncology":       ["cancer", "tumor", "carcinoma", "lymphoma", "leukemia",
                       "melanoma", "glioma", "sarcoma", "myeloma", "neoplasm"],
    "Infectious":     ["infection", "hiv", "covid", "influenza", "hepatitis",
                       "malaria", "tuberculosis", "bacterial", "viral", "candida"],
    "Cardiovascular": ["heart", "cardiac", "hypertension", "coronary", "arrhythmia",
                       "stroke", "vascular", "atherosclerosis"],
    "Neurology":      ["alzheimer", "parkinson", "epilepsy", "multiple sclerosis",
                       "neurological", "dementia", "migraine", "autism"],
    "Immunology":     ["autoimmune", "lupus", "rheumatoid", "crohn", "psoriasis",
                       "inflammatory bowel", "allergy", "immunology"],
    "Endocrinology":  ["diabetes", "obesity", "thyroid", "metabolic", "insulin",
                       "endocrine"],
    "Respiratory":    ["asthma", "copd", "lung", "pulmonary", "respiratory", "bronchitis"],
    "Healthy":        ["healthy", "volunteer"],
}

def classify_domain(text):
    if pd.isna(text):
        return "Other"
    text = str(text).lower()
    for domain, keywords in DOMAIN_MAP.items():
        for kw in keywords:
            if kw in text:
                return domain
    return "Other"

df["Intervention_Type"] = df["Interventions"].apply(extract_intervention_type)
df["Masking_Type"]      = df["Study Design"].apply(extract_masking)
df["Allocation_Type"]   = df["Study Design"].apply(extract_allocation)
df["Medical_Domain"]    = df["Conditions"].apply(classify_domain)
df["Is_Randomized"]     = (df["Allocation_Type"] == "RANDOMIZED").astype(int)
df["Years_Since_2005"]  = df["Start_Year"] - 2005

masking_rank_map = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "QUADRUPLE": 4}
funder_rank_map  = {
    "INDUSTRY": 3, "NIH": 2, "FED": 2, "OTHER_GOV": 2,
    "NETWORK": 1, "OTHER": 1, "UNKNOWN": 0, "INDIV": 0
}
df["Masking_Rank"] = df["Masking_Type"].map(masking_rank_map).fillna(0)
df["Funder_Rank"]  = df["Funder Type"].map(funder_rank_map).fillna(0)

sponsor_trial_counts  = df["Sponsor"].value_counts()
df["Sponsor_Size"]    = df["Sponsor"].map(sponsor_trial_counts)

sponsor_fail_rate = (
    df[df["Is_Failed"].notna()]
    .groupby("Sponsor")["Is_Failed"].mean()
)
df["Sponsor_Hist_Fail_Rate"] = df["Sponsor"].map(sponsor_fail_rate).fillna(
    df["Is_Failed"].mean()
)

# encode categories using the saved label encoders
def safe_encode(encoder, value, default=0):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return default

df["Intervention_Code"] = df["Intervention_Type"].apply(
    lambda x: safe_encode(le_intervention, x)
)
df["Domain_Code"] = df["Medical_Domain"].apply(
    lambda x: safe_encode(le_domain, x)
)

print(f"Dataset rebuilt. Total rows: {df.shape[0]}")

# step 3: score every trial in the dataset using the loaded model
# this gives us a model-predicted failure probability per trial

X_all = df[feature_cols].fillna(0).values
df["Predicted_Failure_Prob"] = model.predict_proba(X_all)[:, 1]

# risk bucket based on probability thresholds
def risk_label(p):
    if p < 0.20:   return "Low Risk"
    elif p < 0.40: return "Moderate Risk"
    else:          return "High Risk"

df["Risk_Level"] = df["Predicted_Failure_Prob"].apply(risk_label)

print("Model scoring complete. Sample predictions:")
print(df[["Sponsor", "Phase_Clean", "Medical_Domain",
          "Status_Clean", "Predicted_Failure_Prob", "Risk_Level"]].head(8).to_string(index=False))

# step 4: the main function - generates a full sponsor-level dashboard
# using model scores, actual outcomes, and trial characteristics

df.to_csv("Model_ready_df.csv")

def generate_model_driven_sponsor_dashboard(df, sponsor_name, save_path=None):
    """
    Generates a targeted sponsor intelligence dashboard powered by
    the trained GradientBoosting model.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and model-scored dataframe (must have Predicted_Failure_Prob column).
    sponsor_name : str
        Exact sponsor name from the Sponsor column.
    save_path : str or None
        File path to save the figure. Example: "pfizer_dashboard.png"
        If None, figure is only displayed.

    Usage
    -----
    generate_model_driven_sponsor_dashboard(df, "Pfizer", "pfizer_dashboard.png")
    generate_model_driven_sponsor_dashboard(df, "GlaxoSmithKline")
    generate_model_driven_sponsor_dashboard(df, "Boehringer Ingelheim", "bi_dashboard.png")
    """

    df_s = df[df["Sponsor"] == sponsor_name].copy()

    if df_s.empty:
        print(f"Sponsor '{sponsor_name}' not found in dataset.")
        print("Run Part 1 to see all available sponsor names.")
        return

    # compute summary statistics
    total         = len(df_s)
    completed     = (df_s["Status_Clean"] == "Completed").sum()
    terminated    = (df_s["Status_Clean"] == "Terminated").sum()
    active        = (df_s["Status_Clean"] == "Active").sum()
    known_outcomes = completed + terminated
    failure_rate  = (terminated / known_outcomes * 100) if known_outcomes > 0 else 0
    avg_prob      = df_s["Predicted_Failure_Prob"].mean() * 100
    med_enroll    = df_s["Enrollment"].median()
    med_duration  = df_s["Duration_Months"].median()
    high_risk_pct = (df_s["Risk_Level"] == "High Risk").mean() * 100

    # color palette
    C_BLUE    = "#2E86AB"
    C_RED     = "#E84855"
    C_GREEN   = "#3BB273"
    C_ORANGE  = "#FAA916"
    C_PURPLE  = "#9B5DE5"
    C_BG      = "#F8F9FA"
    C_CARD    = "#FFFFFF"

    fig = plt.figure(figsize=(24, 30))
    fig.patch.set_facecolor(C_BG)
    gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.60, wspace=0.38)
    tp = {"fontsize": 10, "fontweight": "bold", "pad": 9}

    # header: sponsor name and key metrics as a scorecard row
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_facecolor(C_BLUE)
    ax_hdr.axis("off")
    ax_hdr.text(0.5, 0.80, f"Clinical Trial Intelligence Dashboard",
                ha="center", va="center", fontsize=13, color="white",
                transform=ax_hdr.transAxes)
    ax_hdr.text(0.5, 0.45, sponsor_name,
                ha="center", va="center", fontsize=20, fontweight="bold",
                color="white", transform=ax_hdr.transAxes)

    # metric cards inside header
    metrics = [
        ("Total Trials",          f"{total}",           "white"),
        ("Completed",             f"{completed}",        "#90EE90"),
        ("Terminated",            f"{terminated}",       "#FFB6B6"),
        ("Active",                f"{active}",           "#ADD8E6"),
        ("Observed Failure Rate", f"{failure_rate:.1f}%", "#FFD700"),
        ("Model Avg Risk Score",  f"{avg_prob:.1f}%",   "#FFA07A"),
        ("Median Enrollment",     f"{med_enroll:.0f}",  "white"),
        ("Median Duration",       f"{med_duration:.0f}m","white"),
    ]
    x_positions = np.linspace(0.04, 0.96, len(metrics))
    for x, (label, value, color) in zip(x_positions, metrics):
        ax_hdr.text(x, 0.08, value,   ha="center", fontsize=13, fontweight="bold",
                    color=color, transform=ax_hdr.transAxes)
        ax_hdr.text(x, -0.10, label, ha="center", fontsize=7.5, color="#DDDDDD",
                    transform=ax_hdr.transAxes)

    # panel 1: trial volume over time with stacked status
    ax1 = fig.add_subplot(gs[1, :2])
    ax1.set_facecolor(C_CARD)
    status_year = df_s.groupby(["Start_Year", "Status_Clean"]).size().unstack(fill_value=0)
    status_colors = {
        "Completed": C_GREEN, "Terminated": C_RED,
        "Active": C_BLUE, "Unknown": "#AAAAAA"
    }
    bottom = np.zeros(len(status_year))
    for status in ["Completed", "Terminated", "Active", "Unknown"]:
        if status in status_year.columns:
            ax1.bar(status_year.index, status_year[status],
                    bottom=bottom, label=status,
                    color=status_colors.get(status, "#888888"),
                    edgecolor="white", linewidth=0.4)
            bottom += status_year[status].values
    ax1.set_title("Trial Volume Over Time by Outcome Status", **tp)
    ax1.set_xlabel("Year", fontsize=8)
    ax1.set_ylabel("Number of Trials", fontsize=8)
    ax1.legend(fontsize=8, title="Status", loc="upper left")

    # panel 2: model-predicted risk distribution (pie)
    ax2 = fig.add_subplot(gs[1, 2])
    ax2.set_facecolor(C_CARD)
    risk_counts = df_s["Risk_Level"].value_counts()
    risk_colors = {
        "Low Risk": C_GREEN, "Moderate Risk": C_ORANGE, "High Risk": C_RED
    }
    colors_pie = [risk_colors.get(r, "#888888") for r in risk_counts.index]
    wedges, texts, autotexts = ax2.pie(
        risk_counts.values,
        labels=risk_counts.index,
        autopct="%1.0f%%",
        startangle=140,
        colors=colors_pie,
        textprops={"fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax2.set_title("Model-Predicted Risk Distribution", **tp)

    # panel 3: failure rate by phase (actual vs model predicted)
    ax3 = fig.add_subplot(gs[2, :])
    ax3.set_facecolor(C_CARD)
    df_outcome = df_s[df_s["Status_Clean"].isin(["Completed", "Terminated"])].copy()
    phase_order = ["Phase 1", "Phase 1/2", "Phase 2", "Phase 2/3", "Phase 3", "Phase 4", "Unknown"]
    present_phases = [p for p in phase_order if p in df_s["Phase_Clean"].values]

    actual_fail  = []
    model_fail   = []
    phase_labels = []
    for ph in present_phases:
        sub = df_outcome[df_outcome["Phase_Clean"] == ph]
        mod = df_s[df_s["Phase_Clean"] == ph]
        if len(sub) > 0:
            actual_fail.append(sub["Is_Failed"].mean() * 100)
            model_fail.append(mod["Predicted_Failure_Prob"].mean() * 100)
            phase_labels.append(ph)

    x = np.arange(len(phase_labels))
    w = 0.38
    bars1 = ax3.bar(x - w/2, actual_fail, width=w, label="Actual Failure Rate",
                    color=C_RED, edgecolor="white", alpha=0.85)
    bars2 = ax3.bar(x + w/2, model_fail, width=w, label="Model Predicted Risk",
                    color=C_ORANGE, edgecolor="white", alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(phase_labels, fontsize=8)
    ax3.set_title("Actual Failure Rate vs Model Predicted Risk by Phase", **tp)
    ax3.set_ylabel("Rate / Probability (%)", fontsize=8)
    ax3.legend(fontsize=8)
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.0f}%",
                     ha="center", fontsize=7.5, color=C_RED, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.0f}%",
                     ha="center", fontsize=7.5, color=C_ORANGE, fontweight="bold")

    # panel 4: medical domain breakdown with failure overlay
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.set_facecolor(C_CARD)
    dom_counts = df_s["Medical_Domain"].value_counts()
    dom_fail   = df_outcome.groupby("Medical_Domain")["Is_Failed"].mean() * 100
    bar_colors = [
        C_RED if dom_fail.get(d, 0) > 25 else C_BLUE
        for d in dom_counts.index
    ]
    ax4.barh(dom_counts.index, dom_counts.values, color=bar_colors, edgecolor="white")
    ax4.set_title("Medical Domain Focus\n(red = domain failure rate > 25%)", **tp)
    ax4.set_xlabel("Trial Count", fontsize=8)
    ax4.tick_params(axis="y", labelsize=7)

    # panel 5: model predicted failure probability distribution
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.set_facecolor(C_CARD)
    probs = df_s["Predicted_Failure_Prob"].dropna()
    ax5.hist(probs, bins=20, color=C_PURPLE, edgecolor="white", linewidth=0.5)
    ax5.axvline(probs.mean(), color=C_RED, linestyle="--", linewidth=1.5,
                label=f"Mean: {probs.mean()*100:.1f}%")
    ax5.axvline(0.20, color=C_ORANGE, linestyle=":", linewidth=1.2, label="Low/Mod threshold (20%)")
    ax5.axvline(0.40, color=C_RED,    linestyle=":", linewidth=1.2, label="Mod/High threshold (40%)")
    ax5.set_title("Distribution of Predicted Failure Probability", **tp)
    ax5.set_xlabel("Predicted Failure Probability", fontsize=8)
    ax5.set_ylabel("Trial Count", fontsize=8)
    ax5.legend(fontsize=7)

    # panel 6: enrollment trend over time
    ax6 = fig.add_subplot(gs[3, 2])
    ax6.set_facecolor(C_CARD)
    enroll_trend = df_s.groupby("Start_Year")["Enrollment"].median()
    ax6.plot(enroll_trend.index, enroll_trend.values, "o-",
             color=C_GREEN, linewidth=2, markersize=4)
    ax6.fill_between(enroll_trend.index, enroll_trend.values, alpha=0.15, color=C_GREEN)
    ax6.set_title("Median Enrollment Trend Over Time", **tp)
    ax6.set_xlabel("Year", fontsize=8)
    ax6.set_ylabel("Median Enrollment", fontsize=8)

    # panel 7: high risk trials table (top 10 most at-risk active/recent trials)
    ax7 = fig.add_subplot(gs[4, :])
    ax7.set_facecolor(C_CARD)
    ax7.axis("off")
    high_risk_trials = (
        df_s[df_s["Status_Clean"].isin(["Active", "Completed", "Unknown"])]
        .sort_values("Predicted_Failure_Prob", ascending=False)
        .head(10)[["NCT Number", "Phase_Clean", "Medical_Domain",
                   "Status_Clean", "Enrollment", "Predicted_Failure_Prob"]]
        .copy()
    )
    high_risk_trials["Predicted_Failure_Prob"] = (
        high_risk_trials["Predicted_Failure_Prob"] * 100
    ).round(1).astype(str) + "%"
    high_risk_trials.columns = ["NCT Number", "Phase", "Domain",
                                 "Status", "Enrollment", "Model Risk Score"]

    if not high_risk_trials.empty:
        ax7.set_title("Top 10 Highest Risk Trials (Model Flagged)",
                      fontsize=10, fontweight="bold", pad=12, loc="left", x=0.02)
        table = ax7.table(
            cellText=high_risk_trials.values,
            colLabels=high_risk_trials.columns,
            cellLoc="center",
            loc="center",
            bbox=[0, 0.05, 1, 0.88]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#DDDDDD")
            if row == 0:
                cell.set_facecolor(C_BLUE)
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#F2F2F2")
            else:
                cell.set_facecolor("white")
    else:
        ax7.text(0.5, 0.5, "No active/open trials found for flagging.",
                 ha="center", va="center", fontsize=10, transform=ax7.transAxes)

    # panel 8: intervention type mix
    ax8 = fig.add_subplot(gs[5, 0])
    ax8.set_facecolor(C_CARD)
    int_dist = df_s["Intervention_Type"].value_counts()
    ax8.pie(
        int_dist.values,
        labels=int_dist.index,
        autopct="%1.0f%%",
        startangle=90,
        colors=plt.cm.Set2.colors[:len(int_dist)],
        textprops={"fontsize": 8},
        wedgeprops={"edgecolor": "white", "linewidth": 1.2}
    )
    ax8.set_title("Intervention Type Mix", **tp)

    # panel 9: funder type distribution
    ax9 = fig.add_subplot(gs[5, 1])
    ax9.set_facecolor(C_CARD)
    funder_dist = df_s["Funder Type"].value_counts()
    ax9.bar(funder_dist.index, funder_dist.values, color=C_BLUE, edgecolor="white")
    ax9.set_title("Funder Type Distribution", **tp)
    ax9.set_ylabel("Trial Count", fontsize=8)
    ax9.tick_params(axis="x", rotation=30, labelsize=7)

    # panel 10: phase progression insight - how many trials made it from phase 1 to 3
    ax10 = fig.add_subplot(gs[5, 2])
    ax10.set_facecolor(C_CARD)
    phase_funnel = df_s["Phase_Clean"].value_counts().reindex(
        ["Phase 1", "Phase 1/2", "Phase 2", "Phase 2/3", "Phase 3", "Phase 4"]
    ).dropna()
    if not phase_funnel.empty:
        colors_funnel = [C_BLUE, "#4E9DB3", C_GREEN, "#5CC98A", C_ORANGE, C_PURPLE]
        ax10.barh(phase_funnel.index, phase_funnel.values,
                  color=colors_funnel[:len(phase_funnel)], edgecolor="white")
        ax10.set_title("Trial Count by Phase\n(Pipeline Depth)", **tp)
        ax10.set_xlabel("Trial Count", fontsize=8)
        ax10.tick_params(axis="y", labelsize=8)
    else:
        ax10.text(0.5, 0.5, "No phase data available",
                  ha="center", va="center", transform=ax10.transAxes)

    fig.suptitle(
        f"Model-Driven Sponsor Intelligence Dashboard: {sponsor_name}  |  ClinicalTrials.gov 2005-2025",
        fontsize=14, fontweight="bold", y=1.002, x=0.5
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor=C_BG, dpi=130)
        print(f"Dashboard saved: {save_path}")

    plt.show()

    # print a brief narrative summary to accompany the visual
    print()
    print(f"SPONSOR SUMMARY: {sponsor_name}")
    print(f"  Total Trials       : {total}")
    print(f"  Completed          : {completed}")
    print(f"  Terminated         : {terminated}")
    print(f"  Observed Fail Rate : {failure_rate:.1f}%")
    print(f"  Model Avg Risk     : {avg_prob:.1f}%")
    print(f"  Trials flagged High Risk by model: "
          f"{(df_s['Risk_Level'] == 'High Risk').sum()} "
          f"({high_risk_pct:.1f}% of portfolio)")


def generate_sponsor_report(df, sponsor_name, save_path=None):
    """
    Generates a complete sponsor-level intelligence report.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned clinical trials dataframe.
    sponsor_name : str
        Exact sponsor name as it appears in the 'Sponsor' column.
        Example: 'Pfizer', 'GlaxoSmithKline'
    save_path : str or None
        If provided, saves the figure to this path.
        Example: 'pfizer_report.png'

    Returns
    -------
    None. Displays and optionally saves the report figure.
    """

    df_sponsor = df[df["Sponsor"] == sponsor_name].copy()

    if len(df_sponsor) == 0:
        print(f"No data found for sponsor: {sponsor_name}")
        print("Available sponsors:", df["Sponsor"].value_counts().head(20).index.tolist())
        return

    df_outcome = df_sponsor[df_sponsor["Status_Clean"].isin(["Completed", "Terminated"])]
    total = len(df_sponsor)
    completed = (df_sponsor["Status_Clean"] == "Completed").sum()
    terminated = (df_sponsor["Status_Clean"] == "Terminated").sum()
    failure_rate = terminated / max(completed + terminated, 1) * 100
    avg_duration = df_sponsor["Duration_Months"].median()
    avg_enroll = df_sponsor["Enrollment"].median()

    fig = plt.figure(figsize=(22, 26))
    fig.patch.set_facecolor("#FAFAFA")
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.55, wspace=0.35)
    title_props = {"fontsize": 10, "fontweight": "bold", "pad": 8}

    # header metrics
    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis("off")
    metrics = [
        f"Total Trials: {total}",
        f"Completed: {completed}",
        f"Terminated: {terminated}",
        f"Failure Rate: {failure_rate:.1f}%",
        f"Median Duration: {avg_duration:.0f} months",
        f"Median Enrollment: {avg_enroll:.0f}"
    ]
    x_positions = np.linspace(0.05, 0.95, len(metrics))
    for x, metric in zip(x_positions, metrics):
        label, value = metric.split(": ")
        header_ax.text(x, 0.6, value, ha="center", fontsize=16, fontweight="bold",
                       color="#2E86AB")
        header_ax.text(x, 0.2, label, ha="center", fontsize=9, color="#555555")
    header_ax.set_title(f"Sponsor Intelligence Report: {sponsor_name}",
                        fontsize=16, fontweight="bold", pad=10)
    header_ax.axhline(0.0, color="#DDDDDD", linewidth=1)

    # panel 1: trial volume over time
    ax1 = fig.add_subplot(gs[1, :2])
    ts = df_sponsor.groupby("Start_Year").size()
    ax1.bar(ts.index, ts.values, color="#2E86AB", edgecolor="white")
    ax1.set_title("Trial Volume Over Time", **title_props)
    ax1.set_xlabel("Year", fontsize=8)
    ax1.set_ylabel("Trials", fontsize=8)

    # panel 2: phase distribution
    ax2 = fig.add_subplot(gs[1, 2])
    phase_dist = df_sponsor["Phase_Clean"].value_counts()
    ax2.pie(phase_dist.values, labels=phase_dist.index, autopct="%1.0f%%",
            startangle=140, textprops={"fontsize": 8},
            colors=plt.cm.tab10.colors[:len(phase_dist)])
    ax2.set_title("Phase Distribution", **title_props)

    # panel 3: failure rate by phase for this sponsor
    ax3 = fig.add_subplot(gs[2, :])
    if len(df_outcome) > 0:
        phase_fail = df_outcome.groupby("Phase_Clean")["Is_Failed"].agg(
            Total="count", Failed="sum"
        ).reset_index()
        phase_fail["Failure_Rate"] = (phase_fail["Failed"] / phase_fail["Total"] * 100).round(1)
        colors_p = ["#E84855" if r > 20 else "#3BB273" for r in phase_fail["Failure_Rate"]]
        bars = ax3.bar(phase_fail["Phase_Clean"], phase_fail["Failure_Rate"],
                       color=colors_p, edgecolor="white")
        for bar, rate in zip(bars, phase_fail["Failure_Rate"]):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{rate}%", ha="center", fontsize=9)
        ax3.set_title("Failure Rate by Phase", **title_props)
        ax3.set_xlabel("Phase", fontsize=8)
        ax3.set_ylabel("Failure Rate (%)", fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No outcome data", ha="center", va="center", transform=ax3.transAxes)

    # panel 4: medical domain breakdown
    ax4 = fig.add_subplot(gs[3, 0])
    dom_dist = df_sponsor["Medical_Domain"].value_counts()
    dom_dist.plot(kind="barh", ax=ax4, color="#FAA916", edgecolor="white")
    ax4.set_title("Medical Domain Focus", **title_props)
    ax4.set_xlabel("Trials", fontsize=8)
    ax4.tick_params(axis="y", labelsize=7)

    # panel 5: enrollment distribution
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.hist(df_sponsor["Log_Enrollment"].dropna(), bins=20, color="#3BB273", edgecolor="white")
    ax5.axvline(df_sponsor["Log_Enrollment"].median(), color="red", linestyle="--",
                linewidth=1.5, label="Median")
    ax5.set_title("Enrollment Distribution (log scale)", **title_props)
    ax5.set_xlabel("log(1 + Enrollment)", fontsize=8)
    ax5.set_ylabel("Count", fontsize=8)
    ax5.legend(fontsize=7)

    # panel 6: trial duration distribution
    ax6 = fig.add_subplot(gs[3, 2])
    ax6.hist(df_sponsor["Duration_Months"].dropna().clip(upper=df_sponsor["Duration_Months"].quantile(0.95)),
             bins=20, color="#9B5DE5", edgecolor="white")
    ax6.axvline(df_sponsor["Duration_Months"].median(), color="red", linestyle="--",
                linewidth=1.5, label=f"Median: {df_sponsor['Duration_Months'].median():.0f}m")
    ax6.set_title("Trial Duration Distribution", **title_props)
    ax6.set_xlabel("Duration (Months)", fontsize=8)
    ax6.set_ylabel("Count", fontsize=8)
    ax6.legend(fontsize=7)

    # panel 7: intervention types
    ax7 = fig.add_subplot(gs[4, 0])
    int_dist = df_sponsor["Intervention_Type"].value_counts()
    ax7.pie(int_dist.values, labels=int_dist.index, autopct="%1.0f%%",
            startangle=90, textprops={"fontsize": 8},
            colors=plt.cm.Set2.colors[:len(int_dist)])
    ax7.set_title("Intervention Type Mix", **title_props)

    # panel 8: funder type
    ax8 = fig.add_subplot(gs[4, 1])
    funder_dist = df_sponsor["Funder Type"].value_counts()
    ax8.bar(funder_dist.index, funder_dist.values, color="#2E86AB", edgecolor="white")
    ax8.set_title("Funder Type", **title_props)
    ax8.set_xlabel("", fontsize=8)
    ax8.set_ylabel("Count", fontsize=8)
    ax8.tick_params(axis="x", rotation=30, labelsize=7)

    # panel 9: enrollment trend
    ax9 = fig.add_subplot(gs[4, 2])
    enroll_trend = df_sponsor.groupby("Start_Year")["Enrollment"].median()
    ax9.plot(enroll_trend.index, enroll_trend.values, "o-", color="#E84855",
             linewidth=1.5, markersize=4)
    ax9.set_title("Median Enrollment Trend", **title_props)
    ax9.set_xlabel("Year", fontsize=8)
    ax9.set_ylabel("Median Enrollment", fontsize=8)

    fig.suptitle(
        f"Clinical Trial Intelligence Report: {sponsor_name}",
        fontsize=16, fontweight="bold", y=1.005
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor="#FAFAFA")
        print(f"Report saved to: {save_path}")

    plt.show()
    return fig


def sql_query_frametable(df, table_name, database_name, output_file="mysql_queries.txt"):
    import pandas as pd
    import numpy as np
    # -------------------------------
    # STEP 1: Infer MySQL Data Types
    # -------------------------------
    def map_dtype(dtype):
        if pd.api.types.is_integer_dtype(dtype):
            return "INT"
        elif pd.api.types.is_float_dtype(dtype):
            return "FLOAT"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return "DATETIME"
        else:
            return "TEXT"  # safe fallback for strings/mixed

    column_defs = []
    for col in df.columns:
        col_clean = col.replace(" ", "_").replace("-", "_")
        sql_type = map_dtype(df[col].dtype)
        column_defs.append(f"`{col_clean}` {sql_type}")

    create_table_query = f"""
CREATE DATABASE IF NOT EXISTS `{database_name}`;
USE `{database_name}`;

CREATE TABLE IF NOT EXISTS `{table_name}` (
    {',\n    '.join(column_defs)}
);
"""

    # -------------------------------
    # STEP 2: Generate INSERT Queries
    # -------------------------------
    def format_value(val):
        if pd.isna(val):
            return "NULL"
        elif isinstance(val, (int, float, np.integer, np.floating)):
            return str(val)
        elif isinstance(val, pd.Timestamp):
            return f"'{val.strftime('%Y-%m-%d %H:%M:%S')}'"
        else:
            val = str(val).replace("'", "''")  # escape quotes
            return f"'{val}'"

    columns_sql = ", ".join([f"`{col.replace(' ', '_').replace('-', '_')}`" for col in df.columns])

    insert_queries = []
    for _, row in df.iterrows():  # OK for ~4000 rows
        values = ", ".join([format_value(v) for v in row])
        insert_queries.append(f"INSERT INTO `{table_name}` ({columns_sql}) VALUES ({values});")

    # -------------------------------
    # STEP 3: Write to TXT File
    # -------------------------------
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(create_table_query)
        f.write("\n-- INSERT STATEMENTS --\n\n")
        f.write("\n".join(insert_queries))

    print(f"✅ SQL queries generated and saved to {output_file}")

