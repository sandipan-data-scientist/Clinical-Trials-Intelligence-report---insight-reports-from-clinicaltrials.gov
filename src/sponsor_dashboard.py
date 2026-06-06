"""
SponsorDashboard
Generates a full multi-panel model-driven intelligence report
for a single named sponsor. Designed for Streamlit rendering and
PNG download. Uses OOP to keep all state scoped to one sponsor.
"""

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression

C_BLUE   = "#2E86AB"
C_RED    = "#E84855"
C_GREEN  = "#3BB273"
C_ORANGE = "#FAA916"
C_PURPLE = "#9B5DE5"
C_BG     = "#FFFFFF"
C_WHITE  = "#FFFFFF"


class SponsorDashboard:
    """
    Builds a targeted analytics dashboard for a single sponsor.

    Usage
    -----
    dash = SponsorDashboard(df_scored, "Pfizer")
    fig  = dash.build()
    png  = dash.to_png_bytes(fig)
    """

    def __init__(self, df: pd.DataFrame, sponsor_name: str):
        self.sponsor_name = sponsor_name
        self.df_all   = df
        self.df_s     = df[df["Sponsor"] == sponsor_name].copy()
        self.df_out   = self.df_s[
            self.df_s["Status_Clean"].isin(["Completed", "Terminated"])
        ].copy()

    def is_valid(self) -> bool:
        return len(self.df_s) > 0

    def get_summary(self) -> dict:
        total     = len(self.df_s)
        completed = (self.df_s["Status_Clean"] == "Completed").sum()
        terminated = (self.df_s["Status_Clean"] == "Terminated").sum()
        active    = (self.df_s["Status_Clean"] == "Active").sum()
        known     = completed + terminated
        fail_rate = (terminated / known * 100) if known > 0 else 0
        avg_risk  = self.df_s["Predicted_Failure_Prob"].mean() * 100 \
            if "Predicted_Failure_Prob" in self.df_s.columns else 0
        high_risk_n = (self.df_s.get("Risk_Level", pd.Series()) == "High Risk").sum()
        return {
            "Total Trials":       total,
            "Completed":          completed,
            "Terminated":         terminated,
            "Active":             active,
            "Failure Rate":       round(fail_rate, 1),
            "Model Avg Risk":     round(avg_risk, 1),
            "High Risk Trials":   int(high_risk_n),
            "Median Enrollment":  round(self.df_s["Enrollment"].median(), 0),
            "Median Duration":    round(self.df_s["Duration_Months"].median(), 0),
        }

    def build(self) -> plt.Figure:
        df_s   = self.df_s
        df_out = self.df_out
        name   = self.sponsor_name
        summ   = self.get_summary()

        fig = plt.figure(figsize=(24, 38))
        fig.patch.set_facecolor(C_BG)
        gs = gridspec.GridSpec(8, 3, figure=fig, hspace=0.60, wspace=0.38)
        tp = {"fontsize": 9, "fontweight": "bold", "pad": 8}

        # header scorecard
        ax_hdr = fig.add_subplot(gs[0, :])
        ax_hdr.set_facecolor(C_BLUE)
        ax_hdr.axis("off")
        ax_hdr.text(0.5, 0.80, "Clinical Trial Sponsor Intelligence Report",
                    ha="center", va="center", fontsize=21, color="#8E000C",
                    transform=ax_hdr.transAxes)
        ax_hdr.text(0.5, 0.45, name, ha="center", va="center",
                    fontsize=27, fontweight="bold", color="#850052",
                    transform=ax_hdr.transAxes)
        metric_items = [
            ("Total Trials",     str(summ["Total Trials"]),     "#540D34"),
            ("Completed",        str(summ["Completed"]),         "#008E00"),
            ("Terminated",       str(summ["Terminated"]),        "#B80202"),
            ("Active",           str(summ["Active"]),            "#0084B0"),
            ("Failure Rate",     f"{summ['Failure Rate']}%",    "#6E5E02"),
            ("Model Risk Score", f"{summ['Model Avg Risk']}%",  "#AB3405"),
            ("Median Enrollment",str(int(summ["Median Enrollment"])), "#09008E"),
            ("Median Duration",  f"{int(summ['Median Duration'])}m","#6A0068"),
        ]
        xpos = np.linspace(0.04, 0.96, len(metric_items))
        for x, (label, value, color) in zip(xpos, metric_items):
            ax_hdr.text(x, 0.08, value, ha="center", fontsize=21, fontweight="bold",
                        color=color, transform=ax_hdr.transAxes)
            ax_hdr.text(x, -0.12, label, ha="center", fontsize=14, color="#000000",
                        transform=ax_hdr.transAxes)

        # panel 1: trial volume by year stacked by status
        ax1 = fig.add_subplot(gs[1, :2])
        ax1.set_facecolor(C_WHITE)
        sty = df_s.groupby(["Start_Year", "Status_Clean"]).size().unstack(fill_value=0)
        status_colors = {"Completed": C_GREEN, "Terminated": C_RED,
                         "Active": C_BLUE, "Unknown": "#AAAAAA"}
        bottom = np.zeros(len(sty))
        for status in ["Completed", "Terminated", "Active", "Unknown"]:
            if status in sty.columns:
                ax1.bar(sty.index, sty[status], bottom=bottom,
                        label=status, color=status_colors[status], edgecolor="white", lw=0.4)
                bottom += sty[status].values
        ax1.set_title("Trial Volume Over Time by Outcome", **tp)
        ax1.set_xlabel("Year", fontsize=8)
        ax1.set_ylabel("Trials", fontsize=8)
        ax1.legend(fontsize=7)

        # panel 2: phase distribution
        ax2 = fig.add_subplot(gs[1, 2])
        ax2.set_facecolor(C_WHITE)
        ph = df_s["Phase_Clean"].value_counts()
        ax2.pie(ph.values, labels=ph.index, autopct="%1.0f%%", startangle=140,
                textprops={"fontsize": 8},
                colors=plt.cm.tab10.colors[:len(ph)],
                wedgeprops={"edgecolor": "white", "lw": 1.2})
        ax2.set_title("Phase Distribution", **tp)

        # panel 3: actual vs model predicted failure by phase
        ax3 = fig.add_subplot(gs[2, :])
        ax3.set_facecolor(C_WHITE)
        phase_order = ["Phase 1","Phase 1/2","Phase 2","Phase 2/3","Phase 3","Phase 4","Unknown"]
        present = [p for p in phase_order if p in df_s["Phase_Clean"].values]
        actual, predicted, labels = [], [], []
        for ph_name in present:
            sub = df_out[df_out["Phase_Clean"] == ph_name]
            mod = df_s[df_s["Phase_Clean"] == ph_name]
            if len(sub) > 0:
                actual.append(sub["Is_Failed"].mean() * 100)
                predicted.append(
                    mod["Predicted_Failure_Prob"].mean() * 100
                    if "Predicted_Failure_Prob" in mod.columns else 0
                )
                labels.append(ph_name)
        x = np.arange(len(labels))
        w = 0.38
        b1 = ax3.bar(x - w/2, actual, width=w, label="Actual Failure Rate",
                     color=C_RED, edgecolor="white", alpha=0.85)
        b2 = ax3.bar(x + w/2, predicted, width=w, label="Model Predicted Risk",
                     color=C_ORANGE, edgecolor="white", alpha=0.85)
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, fontsize=8)
        ax3.set_title("Actual Failure Rate vs Model Risk Score by Phase", **tp)
        ax3.set_ylabel("Rate (%)", fontsize=8)
        ax3.legend(fontsize=8)
        for bar in b1:
            h = bar.get_height()
            if h > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, h + 0.4,
                         f"{h:.0f}%", ha="center", fontsize=7, color=C_RED, fontweight="bold")

        # panel 4: therapeutic domain distribution + growth forecast
        ax4 = fig.add_subplot(gs[3, :2])
        ax4.set_facecolor(C_WHITE)
        dom_counts = df_s["Medical_Domain"].value_counts()
        dom_lr_slopes = {}
        for dom in dom_counts.index:
            dts = df_s[df_s["Medical_Domain"] == dom].groupby("Start_Year").size()
            if len(dts) >= 3:
                t_ = dts.index.values.reshape(-1, 1)
                lr_ = LinearRegression().fit(t_, dts.values)
                dom_lr_slopes[dom] = lr_.coef_[0]
            else:
                dom_lr_slopes[dom] = 0
        growth_signals = [dom_lr_slopes.get(d, 0) for d in dom_counts.index]
        bar_colors = [C_GREEN if g > 0 else C_RED for g in growth_signals]
        bars = ax4.barh(dom_counts.index, dom_counts.values, color=bar_colors, edgecolor="white")
        for bar, slope in zip(bars, growth_signals):
            trend_label = f"trend: +{slope:.1f}/yr" if slope > 0 else f"trend: {slope:.1f}/yr"
            ax4.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                     trend_label, va="center", fontsize=7,
                     color=C_GREEN if slope > 0 else C_RED)
        ax4.set_title("Therapeutic Domain Distribution\n(green = growing trend, red = declining)",
                      **tp)
        ax4.set_xlabel("Trial Count", fontsize=8)
        ax4.tick_params(axis="y", labelsize=7)

        # panel 5: sex group distribution
        ax5 = fig.add_subplot(gs[3, 2])
        ax5.set_facecolor(C_WHITE)
        sex_dist = df_s["Sex_Clean"].value_counts()
        sex_colors = {"All Sexes": C_BLUE, "Male Only": "#4ECDC4", "Female Only": "#FF6B9D",
                      "Not Specified": "#AAAAAA"}
        pie_colors = [sex_colors.get(s, C_BLUE) for s in sex_dist.index]
        ax5.pie(sex_dist.values, labels=sex_dist.index, autopct="%1.0f%%",
                startangle=90, colors=pie_colors,
                textprops={"fontsize": 8},
                wedgeprops={"edgecolor": "white", "lw": 1.2})
        ax5.set_title("Sex Group Distribution", **tp)

        # panel 6: age group distribution
        ax6 = fig.add_subplot(gs[4, 0])
        ax6.set_facecolor(C_WHITE)
        age_counts = df_s["Age_Group_Label"].value_counts()
        age_colors_map = plt.cm.Set2.colors[:len(age_counts)]
        ax6.barh(age_counts.index, age_counts.values, color=age_colors_map, edgecolor="white")
        ax6.set_title("Age Group Distribution", **tp)
        ax6.set_xlabel("Trial Count", fontsize=8)
        ax6.tick_params(axis="y", labelsize=7)

        # panel 7: enrollment trend over time
        ax7 = fig.add_subplot(gs[4, 1])
        ax7.set_facecolor(C_WHITE)
        enroll_trend = df_s.groupby("Start_Year")["Enrollment"].median()
        ax7.plot(enroll_trend.index, enroll_trend.values, "o-", color=C_GREEN, lw=2, markersize=4)
        ax7.fill_between(enroll_trend.index, enroll_trend.values, alpha=0.15, color=C_GREEN)
        ax7.set_title("Median Enrollment Trend Over Time", **tp)
        ax7.set_xlabel("Year", fontsize=8)
        ax7.set_ylabel("Median Enrollment", fontsize=8)

        # panel 8: enrollment growth rate year over year
        ax8 = fig.add_subplot(gs[4, 2])
        ax8.set_facecolor(C_WHITE)
        if len(enroll_trend) >= 2:
            yoy_growth = enroll_trend.pct_change() * 100
            yoy_growth = yoy_growth.dropna()
            bar_c = [C_GREEN if v >= 0 else C_RED for v in yoy_growth.values]
            ax8.bar(yoy_growth.index, yoy_growth.values, color=bar_c, edgecolor="white")
            ax8.axhline(0, color="black", lw=0.8)
        ax8.set_title("Enrollment Year-over-Year Growth Rate (%)", **tp)
        ax8.set_xlabel("Year", fontsize=8)
        ax8.set_ylabel("YoY Change (%)", fontsize=8)

        # panel 9: trial risk groups from model
        ax9 = fig.add_subplot(gs[5, 0])
        ax9.set_facecolor(C_WHITE)
        if "Risk_Level" in df_s.columns:
            risk_counts = df_s["Risk_Level"].value_counts()
            risk_c = {"Low Risk": C_GREEN, "Moderate Risk": C_ORANGE, "High Risk": C_RED}
            pie_rc = [risk_c.get(r, C_BLUE) for r in risk_counts.index]
            wedges, texts, ats = ax9.pie(
                risk_counts.values, labels=risk_counts.index, autopct="%1.0f%%",
                colors=pie_rc, textprops={"fontsize": 8},
                wedgeprops={"edgecolor": "white", "lw": 1.2}
            )
            for at in ats:
                at.set_fontweight("bold")
        else:
            ax9.text(0.5, 0.5, "Model scores not available",
                     ha="center", va="center", transform=ax9.transAxes, fontsize=8)
        ax9.set_title("Trial Risk Groups (Model Predicted)", **tp)

        # panel 10: domain forecast for top 3 domains
        ax10 = fig.add_subplot(gs[5, 1:])
        ax10.set_facecolor(C_WHITE)
        top3_domains = dom_counts.head(3).index.tolist()
        forecast_colors = [C_BLUE, C_GREEN, C_ORANGE]
        for i, dom in enumerate(top3_domains):
            dts = df_s[df_s["Medical_Domain"] == dom].groupby("Start_Year").size()
            if len(dts) < 3:
                continue
            t_ = dts.index.values.reshape(-1, 1)
            lr_ = LinearRegression().fit(t_, dts.values)
            future_ = np.arange(dts.index.max()+1, 2036).reshape(-1, 1)
            fcast_ = np.maximum(lr_.predict(future_), 0)
            ax10.plot(dts.index, dts.values, "o-", color=forecast_colors[i],
                      lw=1.5, markersize=3, label=f"{dom} (actual)")
            ax10.plot(future_.flatten(), fcast_, "--", color=forecast_colors[i],
                      lw=1.5, alpha=0.7, label=f"{dom} (forecast)")
        ax10.set_title("Top 3 Domain Growth Forecast", **tp)
        ax10.set_xlabel("Year", fontsize=8)
        ax10.set_ylabel("Trials", fontsize=8)
        ax10.legend(fontsize=7)

        # panel 11: intervention type mix
        ax11 = fig.add_subplot(gs[6, 0])
        ax11.set_facecolor(C_WHITE)
        int_dist = df_s["Intervention_Type"].value_counts()
        ax11.pie(int_dist.values, labels=int_dist.index, autopct="%1.0f%%",
                 startangle=90, colors=plt.cm.Set2.colors[:len(int_dist)],
                 textprops={"fontsize": 8}, wedgeprops={"edgecolor": "white"})
        ax11.set_title("Intervention Type Mix", **tp)

        # panel 12: funder type
        ax12 = fig.add_subplot(gs[6, 1])
        ax12.set_facecolor(C_WHITE)
        funder_dist = df_s["Funder Type"].value_counts()
        ax12.bar(funder_dist.index, funder_dist.values, color=C_BLUE, edgecolor="white")
        ax12.set_title("Funder Type Distribution", **tp)
        ax12.set_ylabel("Count", fontsize=8)
        ax12.tick_params(axis="x", rotation=30, labelsize=7)

        # panel 13: duration distribution
        ax13 = fig.add_subplot(gs[6, 2])
        ax13.set_facecolor(C_WHITE)
        dur_data = df_s["Duration_Months"].dropna()
        dur_data = dur_data.clip(upper=dur_data.quantile(0.95))
        ax13.hist(dur_data, bins=20, color=C_PURPLE, edgecolor="white")
        ax13.axvline(df_s["Duration_Months"].median(), color=C_RED, ls="--", lw=1.5,
                     label=f"Median: {df_s['Duration_Months'].median():.0f}m")
        ax13.set_title("Trial Duration Distribution", **tp)
        ax13.set_xlabel("Duration (Months)", fontsize=8)
        ax13.set_ylabel("Count", fontsize=8)
        ax13.legend(fontsize=7)

        # panel 14: top 10 high risk trials table
        ax14 = fig.add_subplot(gs[7, :])
        ax14.set_facecolor(C_WHITE)
        ax14.axis("off")
        if "Predicted_Failure_Prob" in df_s.columns:
            high_risk = (
                df_s[df_s["Status_Clean"].isin(["Active", "Unknown", "Completed"])]
                .sort_values("Predicted_Failure_Prob", ascending=False)
                .head(10)[["NCT Number", "Phase_Clean", "Medical_Domain",
                            "Status_Clean", "Enrollment", "Predicted_Failure_Prob"]]
                .copy()
            )
            high_risk["Predicted_Failure_Prob"] = (
                high_risk["Predicted_Failure_Prob"] * 100
            ).round(1).astype(str) + "%"
            high_risk.columns = ["NCT Number","Phase","Domain","Status","Enrollment","Risk Score"]
            if not high_risk.empty:
                ax14.set_title("Top 10 Model-Flagged High Risk Trials",
                               fontsize=9, fontweight="bold", pad=10, loc="left", x=0.01)
                tbl = ax14.table(
                    cellText=high_risk.values,
                    colLabels=high_risk.columns,
                    cellLoc="center", loc="center",
                    bbox=[0, 0.05, 1, 0.88]
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(8)
                for (row, col), cell in tbl.get_celld().items():
                    cell.set_edgecolor("#DDDDDD")
                    if row == 0:
                        cell.set_facecolor(C_BLUE)
                        cell.set_text_props(color="white", fontweight="bold")
                    elif row % 2 == 0:
                        cell.set_facecolor("#F2F2F2")
                    else:
                        cell.set_facecolor("white")

        fig.suptitle(
            f"Sponsor Intelligence Dashboard: {name}  |  ClinicalTrials.gov 2005-2025",
            fontsize=14, fontweight="bold", y=1.002
        )
        plt.tight_layout()
        return fig

    @staticmethod
    def to_png_bytes(fig: plt.Figure) -> bytes:
        """Converts a matplotlib figure to PNG bytes for Streamlit download button."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor=C_BG, dpi=150)
        buf.seek(0)
        return buf.getvalue()