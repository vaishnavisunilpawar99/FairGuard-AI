import os
import json
import numpy as np
import pandas as pd
from .services import drift_service

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> dict:
    """Calculate feature and data drift using both mathematical PSI/KS testing
    and Evidently report generation (if installed).
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    ref = ref_df.copy()
    cur = cur_df.copy()
    for c in ref.columns:
        if ref[c].dtype == "object" or str(ref[c].dtype) == "category":
            ref[c] = ref[c].astype(str)
            cur[c] = cur[c].astype(str)

    # 1. Native mathematical PSI & KS test evaluation (SRS FR-08 & Sec 7.3)
    math_results = drift_service.evaluate_dataset_drift(ref, cur)
    html_path = os.path.join(ARTIFACT_DIR, "drift.html")

    # 2. Attempt Evidently HTML report generation if available
    evidently_success = False
    try:
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
        except ImportError:
            from evidently import Report
            from evidently.presets import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=ref, current_data=cur)
        if hasattr(snapshot, "save_html"):
            snapshot.save_html(html_path)
        else:
            report.save_html(html_path)
        evidently_success = True
    except Exception as exc:
        # Generate standalone interactive HTML report if Evidently is unavailable
        _generate_native_drift_html(math_results, html_path)

    math_results["html_path"] = html_path
    math_results["evidently_available"] = evidently_success
    return math_results


def _generate_native_drift_html(drift_data: dict, output_path: str):
    """Generate self-contained HTML report displaying PSI and KS drift results."""
    rows_html = ""
    for feat, data in drift_data.get("per_feature", {}).items():
        drift_color = "#ef4444" if data.get("drift_detected") else "#22c55e"
        drift_badge = "DRIFT DETECTED" if data.get("drift_detected") else "STABLE"
        rows_html += f"""
        <tr style="border-bottom: 1px solid #e5e7eb;">
            <td style="padding: 10px; font-weight: 500;">{feat}</td>
            <td style="padding: 10px;">{data.get('type', 'N/A')}</td>
            <td style="padding: 10px;">{data.get('psi', 0.0):.4f} ({data.get('psi_category', '')})</td>
            <td style="padding: 10px;">{data.get('p_value', 1.0):.4f}</td>
            <td style="padding: 10px;"><span style="background: {drift_color}20; color: {drift_color}; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 12px;">{drift_badge}</span></td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>FairGuard AI - Data Drift Audit</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; background: #f9fafb; color: #111827; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 14px; }}
        th {{ padding: 10px; background: #f3f4f6; border-bottom: 2px solid #e5e7eb; }}
    </style>
</head>
<body>
    <div class="card">
        <h2>FairGuard AI — Data Drift Audit Report</h2>
        <p><strong>Overall Status:</strong> {drift_data.get('status', 'Stable')} | 
           <strong>Share of Drifted Features:</strong> {drift_data.get('share_drifted', 0.0)*100:.1f}% | 
           <strong>Max PSI:</strong> {drift_data.get('max_psi', 0.0):.4f}</p>
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Type</th>
                    <th>PSI (Population Stability)</th>
                    <th>p-value (KS / Chi2)</th>
                    <th>Drift Status</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def make_shifted(df: pd.DataFrame) -> pd.DataFrame:
    shifted = df.copy()
    if "age" in shifted.columns:
        shifted["age"] = (shifted["age"].astype(float) + 5).clip(upper=90)
    if "hours-per-week" in shifted.columns:
        shifted["hours-per-week"] = (shifted["hours-per-week"].astype(float) + 3).clip(upper=99)
    return shifted
