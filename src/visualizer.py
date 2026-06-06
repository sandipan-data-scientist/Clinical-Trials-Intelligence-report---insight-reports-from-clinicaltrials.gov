"""
Visualizer
All EDA, trend, and ML evaluation charts used by the Streamlit pages.
Each method returns a matplotlib Figure so Streamlit can render it
with st.pyplot(fig) without any global state side-effects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ConfusionMatrixDisplay
from typing import Dict


# shared style applied once at import time
plt.rcParams["font.family"]          = "DejaVu Sans"
plt.rcParams["axes.spines.top"]      = False
plt.rcParams["axes.spines.right"]    = False
plt.rcParams["figure.facecolor"]     = "#FAFAFA"
plt.rcParams["axes.facecolor"]       = "#FAFAFA"

C_BLUE   = "#2E86AB"
C_RED    = "#E84855"
C_GREEN  = "#3BB273"
C_ORANGE = "#FAA916"
C_PURPLE = "#9B5DE5"


class Visualizer:
    """
    Stateless chart factory. All methods accept a dataframe and
    optional parameters, and return a matplotlib Figure.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # 1. Trial Landscape
    def fig_trial_landscape(self) -> plt.Figure:
        df = self.df
        yearly = df.groupby("Start_Year").size().reset_index(name="Count")
        phase_year = df.groupby(["Start_Year", "Phase_Clean"]).size().unstack(fill_value=0)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        axes[0].bar(yearly["Start_Year"], yearly["Count"], color=C_BLUE, edgecolor="white", lw=0.5)
        axes[0].set_title("Total Clinical Trials Initiated Per Year", fontweight="bold", fontsize=12)
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("Number of Trials")

        phase_year.plot(kind="area", stacked=True, ax=axes[1], colormap="tab10", alpha=0.75)
        axes[1].set_title("Trial Volume by Phase Over Time", fontweight="bold", fontsize=12)
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("Number of Trials")
        axes[1].legend(title="Phase", fontsize=8, loc="upper left")
        plt.tight_layout()
        return fig

    # 2. Enrollment distribution
    def fig_enrollment_distribution(self) -> plt.Figure:
        df = self.df
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        clip_val = df["Enrollment"].quantile(0.99)
        axes[0].hist(df["Enrollment"].clip(upper=clip_val), bins=60,
                     color=C_RED, edgecolor="white", lw=0.4)
        axes[0].set_title("Raw Enrollment Distribution (clipped at 99th pct)",
                          fontweight="bold", fontsize=12)
        axes[0].set_xlabel("Enrollment")
        axes[0].set_ylabel("Frequency")

        axes[1].hist(df["Log_Enrollment"].dropna(), bins=50,
                     color=C_GREEN, edgecolor="white", lw=0.4)
        axes[1].set_title("Log-Transformed Enrollment Distribution",
                          fontweight="bold", fontsize=12)
        axes[1].set_xlabel("log(1 + Enrollment)")
        axes[1].set_ylabel("Frequency")
        plt.tight_layout()
        return fig

    # 3. Failure rate by phase
    def fig_failure_by_phase(self) -> plt.Figure:
        df = self.df
        df_out = df[df["Status_Clean"].isin(["Completed", "Terminated"])]
        pf = df_out.groupby("Phase_Clean")["Is_Failed"].agg(
            Total="count", Failed="sum"
        ).reset_index()
        pf["Failure_Rate"] = (pf["Failed"] / pf["Total"] * 100).round(1)
        pf = pf.sort_values("Failure_Rate", ascending=False)
        overall = df_out["Is_Failed"].mean() * 100

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [C_RED if r > 20 else C_GREEN for r in pf["Failure_Rate"]]
        bars = ax.bar(pf["Phase_Clean"], pf["Failure_Rate"], color=colors, edgecolor="white")
        for bar, rate in zip(bars, pf["Failure_Rate"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{rate}%", ha="center", fontsize=9, fontweight="bold")
        ax.axhline(overall, color="black", linestyle="--", lw=1,
                   label=f"Overall avg: {overall:.1f}%")
        ax.set_title("Trial Failure Rate by Phase", fontweight="bold", fontsize=12)
        ax.set_xlabel("Phase")
        ax.set_ylabel("Failure Rate (%)")
        ax.legend()
        plt.tight_layout()
        return fig

    # 4. Sponsor landscape
    def fig_sponsor_landscape(self, top_n: int = 15) -> plt.Figure:
        df = self.df
        top_sponsors = df["Sponsor"].value_counts().head(top_n).index.tolist()
        df_top = df[df["Sponsor"].isin(top_sponsors)]
        df_out = df_top[df_top["Status_Clean"].isin(["Completed", "Terminated"])]

        stats = df_out.groupby("Sponsor").agg(
            Total_Trials=("Is_Failed", "count"),
            Failed_Trials=("Is_Failed", "sum"),
            Avg_Enrollment=("Enrollment", "mean"),
        ).reset_index()
        stats["Failure_Rate"] = (stats["Failed_Trials"] / stats["Total_Trials"] * 100).round(1)
        stats = stats.sort_values("Failure_Rate", ascending=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        colors = [C_RED if r > 20 else C_BLUE for r in stats["Failure_Rate"]]
        axes[0].barh(stats["Sponsor"], stats["Failure_Rate"], color=colors, edgecolor="white")
        axes[0].axvline(stats["Failure_Rate"].mean(), color="black", linestyle="--", lw=1,
                        label=f"Avg: {stats['Failure_Rate'].mean():.1f}%")
        axes[0].set_title(f"Failure Rate: Top {top_n} Sponsors", fontweight="bold", fontsize=12)
        axes[0].set_xlabel("Failure Rate (%)")
        axes[0].legend(fontsize=8)

        sc = axes[1].scatter(
            stats["Total_Trials"], stats["Failure_Rate"],
            s=stats["Avg_Enrollment"].clip(upper=2000) / 5,
            c=stats["Failure_Rate"], cmap="RdYlGn_r",
            edgecolors="white", lw=0.7, alpha=0.85
        )
        for _, row in stats.iterrows():
            axes[1].annotate(row["Sponsor"].split()[0],
                             (row["Total_Trials"], row["Failure_Rate"]), fontsize=7)
        axes[1].set_title("Volume vs Failure Rate\n(bubble size = avg enrollment)",
                          fontweight="bold", fontsize=12)
        axes[1].set_xlabel("Total Trials")
        axes[1].set_ylabel("Failure Rate (%)")
        plt.colorbar(sc, ax=axes[1], label="Failure Rate (%)")
        plt.tight_layout()
        return fig

    # 5. Medical domain growth
    def fig_domain_trends(self) -> plt.Figure:
        df = self.df
        domain_year = df.groupby(["Start_Year", "Medical_Domain"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(14, 6))
        for col in domain_year.columns:
            ax.plot(domain_year.index, domain_year[col], marker="o", markersize=3,
                    lw=1.5, label=col)
        ax.set_title("Clinical Trial Volume by Medical Domain Over Time",
                     fontweight="bold", fontsize=12)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Trials")
        ax.legend(title="Domain", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        return fig

    # 6. Correlation heatmap
    def fig_correlation(self) -> plt.Figure:
        df = self.df
        cols = ["Enrollment", "Log_Enrollment", "Duration_Months", "Start_Year", "Is_Failed"]
        corr = df[cols].dropna().corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask,
                    ax=ax, linewidths=0.5, vmin=-1, vmax=1)
        ax.set_title("Correlation Matrix: Numeric Features", fontweight="bold", fontsize=12)
        plt.tight_layout()
        return fig

    # 7. ROC curves
    def fig_roc_curves(self, roc_data: Dict) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, data in roc_data.items():
            ax.plot(data["fpr"], data["tpr"], lw=2,
                    label=f"{name} (AUC={data['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_title("ROC Curves: Model Comparison", fontweight="bold", fontsize=12)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=9)
        plt.tight_layout()
        return fig

    # 8. Confusion matrix
    def fig_confusion_matrix(self, cm: np.ndarray, model_name: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=["Completed", "Terminated"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(f"Confusion Matrix: {model_name}", fontweight="bold", fontsize=12)
        plt.tight_layout()
        return fig

    # 9. Feature importance
    def fig_feature_importance(self, importances: pd.Series) -> plt.Figure:
        imp = importances.sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(imp.index, imp.values, color=C_BLUE, edgecolor="white")
        ax.set_title("Feature Importance (Gradient Boosting)", fontweight="bold", fontsize=12)
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        return fig

    # 10. Volume forecast
    def fig_volume_forecast(self) -> plt.Figure:
        df = self.df
        yearly = df.groupby("Start_Year").size().reset_index(name="Count")
        yearly = yearly[yearly["Start_Year"] <= 2024]
        t = yearly["Start_Year"].values.reshape(-1, 1)
        y = yearly["Count"].values
        lr = LinearRegression().fit(t, y)
        future = np.arange(2025, 2046).reshape(-1, 1)
        preds  = np.maximum(lr.predict(future), 0)
        std_e  = (y - lr.predict(t)).std()

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(yearly["Start_Year"], yearly["Count"], color=C_BLUE, alpha=0.7, label="Actual")
        ax.plot(future.flatten(), preds, "r--", lw=2, marker="o", markersize=4, label="Forecast")
        ax.fill_between(future.flatten(), preds - 1.96*std_e, preds + 1.96*std_e,
                        alpha=0.15, color="red", label="95% CI")
        ax.set_title("Clinical Trial Volume Forecast: 2025-2045", fontweight="bold", fontsize=12)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Trials")
        ax.legend()
        plt.tight_layout()
        return fig

    # 11. Domain CAGR
    def fig_domain_cagr(self) -> plt.Figure:
        df = self.df
        growth = []
        for domain in df["Medical_Domain"].unique():
            d = df[df["Medical_Domain"] == domain].groupby("Start_Year").size()
            d = d[d.index.isin(range(2015, 2025))]
            if len(d) >= 3 and d.iloc[0] > 0:
                cagr = (d.iloc[-1] / d.iloc[0]) ** (1 / max(len(d)-1, 1)) - 1
                growth.append({"Domain": domain, "CAGR": cagr * 100})
        if not growth:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "Insufficient data for CAGR calculation",
                    ha="center", va="center", transform=ax.transAxes)
            return fig
        gdf = pd.DataFrame(growth).sort_values("CAGR", ascending=False)
        colors = [C_GREEN if c > 0 else C_RED for c in gdf["CAGR"]]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(gdf["Domain"], gdf["CAGR"], color=colors, edgecolor="white")
        ax.axvline(0, color="black", lw=1)
        for bar, val in zip(bars, gdf["CAGR"]):
            ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=8)
        ax.set_title("Domain CAGR 2015-2024: Which Therapeutic Areas Are Growing?",
                     fontweight="bold", fontsize=12)
        ax.set_xlabel("Compound Annual Growth Rate (%)")
        plt.tight_layout()
        return fig

    # 12. Competitive dashboard (multi-panel)
    def fig_competitive_dashboard(self, top_n: int = 10) -> plt.Figure:
        df = self.df
        top_sponsors = df["Sponsor"].value_counts().head(top_n).index.tolist()
        df_top = df[df["Sponsor"].isin(top_sponsors)]
        df_out = df_top[df_top["Status_Clean"].isin(["Completed", "Terminated"])]

        fig = plt.figure(figsize=(22, 20))
        fig.patch.set_facecolor("#F8F9FA")
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38)
        tp = {"fontsize": 9, "fontweight": "bold", "pad": 7}

        # vol by sponsor
        ax1 = fig.add_subplot(gs[0, :2])
        vol = df_top["Sponsor"].value_counts().reindex(top_sponsors)
        ax1.bar(range(len(top_sponsors)), vol.values, color=C_BLUE, edgecolor="white")
        ax1.set_xticks(range(len(top_sponsors)))
        ax1.set_xticklabels([s.split()[0] for s in top_sponsors], rotation=35, fontsize=7)
        ax1.set_title("Total Trials by Sponsor", **tp)
        ax1.set_ylabel("Trial Count", fontsize=8)

        # failure rate
        ax2 = fig.add_subplot(gs[0, 2])
        fr = df_out.groupby("Sponsor")["Is_Failed"].mean() * 100
        fr = fr.reindex(top_sponsors).sort_values(ascending=True)
        col_fr = [C_RED if v > 20 else C_GREEN for v in fr.values]
        fr.plot(kind="barh", ax=ax2, color=col_fr, edgecolor="white")
        ax2.axvline(fr.mean(), color="black", ls="--", lw=1)
        ax2.set_title("Failure Rate (%)", **tp)
        ax2.tick_params(axis="y", labelsize=6)

        # phase distribution
        ax3 = fig.add_subplot(gs[1, :])
        pd_cross = df_top.groupby(["Sponsor", "Phase_Clean"]).size().unstack(fill_value=0)
        pd_cross.plot(kind="bar", stacked=True, ax=ax3, colormap="tab10",
                      edgecolor="white", lw=0.3)
        ax3.set_title("Phase Distribution by Sponsor", **tp)
        ax3.tick_params(axis="x", rotation=35, labelsize=7)
        ax3.legend(title="Phase", fontsize=7, bbox_to_anchor=(1.01, 1))

        # enrollment box
        ax4 = fig.add_subplot(gs[2, :2])
        data = [df_top[df_top["Sponsor"] == s]["Log_Enrollment"].dropna().values
                for s in top_sponsors]
        ax4.boxplot(data, labels=[s.split()[0] for s in top_sponsors],
                    patch_artist=True,
                    boxprops=dict(facecolor="#AED9E0"),
                    medianprops=dict(color="red", lw=1.5))
        ax4.set_title("Enrollment Distribution by Sponsor (log scale)", **tp)
        ax4.tick_params(axis="x", rotation=35, labelsize=7)

        # duration
        ax5 = fig.add_subplot(gs[2, 2])
        dur = df_top.groupby("Sponsor")["Duration_Months"].median().reindex(top_sponsors).sort_values()
        dur.plot(kind="barh", ax=ax5, color=C_ORANGE, edgecolor="white")
        ax5.set_title("Median Trial Duration (Months)", **tp)
        ax5.tick_params(axis="y", labelsize=6)

        # funder type
        ax6 = fig.add_subplot(gs[3, 0])
        fd = df["Funder Type"].value_counts()
        ax6.pie(fd.values, labels=fd.index, autopct="%1.0f%%", startangle=140,
                textprops={"fontsize": 7},
                colors=plt.cm.Set3.colors[:len(fd)],
                wedgeprops={"edgecolor": "white"})
        ax6.set_title("Funder Type Distribution", **tp)

        # domain CAGR inline
        ax7 = fig.add_subplot(gs[3, 1:])
        growth = []
        for domain in df["Medical_Domain"].unique():
            d = df[df["Medical_Domain"] == domain].groupby("Start_Year").size()
            d = d[d.index.isin(range(2015, 2025))]
            if len(d) >= 3 and d.iloc[0] > 0:
                cagr = (d.iloc[-1] / d.iloc[0]) ** (1 / max(len(d)-1, 1)) - 1
                growth.append({"Domain": domain, "CAGR": cagr * 100})
        if growth:
            gdf = pd.DataFrame(growth).sort_values("CAGR", ascending=True)
            c7 = [C_GREEN if v > 0 else C_RED for v in gdf["CAGR"]]
            ax7.barh(gdf["Domain"], gdf["CAGR"], color=c7, edgecolor="white")
            ax7.axvline(0, color="black", lw=1)
        ax7.set_title("Domain CAGR 2015-2024", **tp)

        fig.suptitle("Clinical Trials Competitive Intelligence Dashboard (2005-2025)",
                     fontsize=14, fontweight="bold", y=1.005)
        plt.tight_layout()
        return fig