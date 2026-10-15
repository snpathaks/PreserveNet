"""
app/app.py
----------
PreserveNet Gradio Demo UI.

Theme & Aesthetics:
  - Dark, premium, "live data" feel
  - Background: #0B0B0F
  - Surfaces: #15151B
  - Borders: #2A2A33 (1px subtle)
  - Text: #F5F5F7, Muted: #9A9AA5
  - Primary Accent: Amber Gold #FFB400 (hover #FFC933)
  - Semantic Status: Green #3DDC97 (preserved), Red #FF5C5C (changed)
  - Rounded corners: 14px
  - Font: Inter
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Guarantee imports from current directory and parent repository
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if (REPO_ROOT / 'PreserveNet').exists() and str(REPO_ROOT / 'PreserveNet') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'PreserveNet'))

import gradio as gr
from PIL import Image

# Import the inference wrapper
from inference import Predictor, IMAGENETTE_CLASSES


# =============================================================================
# Custom CSS (Single unified string passed to gr.Blocks)
# =============================================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global Reset & Typography */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    box-sizing: border-box;
}

body, html, .gradio-container {
    background-color: #0B0B0F !important;
    color: #F5F5F7 !important;
    margin: 0;
    padding: 0;
}

/* Container limits & spacing */
.gradio-container {
    max-width: 1380px !important;
    margin: 0 auto !important;
    padding: 24px 20px 48px 20px !important;
}

/* Header Section */
.pn-header {
    margin-bottom: 24px;
    padding: 12px 4px;
    border-bottom: 1px solid #2A2A33;
}

.pn-header h1 {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #F5F5F7;
    margin: 0 0 6px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.pn-header .wordmark-net {
    color: #FFB400;
    text-shadow: 0 0 20px rgba(255, 180, 0, 0.35);
}

.pn-header .tagline {
    font-size: 15px;
    color: #9A9AA5;
    font-weight: 400;
    margin: 0;
}

/* Surfaces & Cards */
.gr-panel, .gr-box, .gr-form, .block, .pn-card {
    background-color: #15151B !important;
    border: 1px solid #2A2A33 !important;
    border-radius: 14px !important;
}

/* Labels */
label span, .block-title, .gr-form > label {
    color: #9A9AA5 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Primary Reduce Button */
#reduce-btn, button.primary {
    background: #FFB400 !important;
    color: #0B0B0F !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border: 1px solid #FFB400 !important;
    border-radius: 14px !important;
    padding: 12px 24px !important;
    cursor: pointer !important;
    transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 2px 10px rgba(255, 180, 0, 0.25) !important;
}

#reduce-btn:hover, button.primary:hover {
    background: #FFC933 !important;
    border-color: #FFC933 !important;
    box-shadow: 0 0 20px rgba(255, 180, 0, 0.5) !important;
    transform: translateY(-1px);
}

#reduce-btn:active, button.primary:active {
    transform: translateY(0);
}

/* Slider Custom Styling */
input[type="range"] {
    accent-color: #FFB400 !important;
}

input[type="range"]::-webkit-slider-thumb {
    background: #FFB400 !important;
    box-shadow: 0 0 10px rgba(255, 180, 0, 0.5) !important;
}

input[type="range"]::-moz-range-thumb {
    background: #FFB400 !important;
    box-shadow: 0 0 10px rgba(255, 180, 0, 0.5) !important;
}

/* Image Panels */
.image-panels-row {
    gap: 16px !important;
    margin-bottom: 20px !important;
}

.image-panels-row .gr-image {
    background: #15151B !important;
    border: 1px solid #2A2A33 !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}

/* Image containers */
.image-container, .image-frame {
    border-radius: 12px !important;
    background: #0E0E12 !important;
}

/* Stats Strip Styling */
.stats-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin: 18px 0;
    padding: 14px 18px;
    background: #15151B;
    border: 1px solid #2A2A33;
    border-radius: 14px;
}

.stat-card {
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 4px 8px;
}

.stat-title {
    font-size: 11px;
    font-weight: 700;
    color: #9A9AA5;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.stat-num {
    font-size: 20px;
    font-weight: 700;
    color: #F5F5F7;
    display: flex;
    align-items: baseline;
    gap: 6px;
}

.stat-denom {
    font-size: 14px;
    color: #9A9AA5;
    font-weight: 500;
}

.stat-badge-pct {
    font-size: 13px;
    color: #FFB400;
    font-weight: 600;
}

.amber-glow {
    color: #FFB400;
    font-weight: 800;
}

/* Status Badges */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    width: fit-content;
}

.status-pill.badge-preserved {
    background: rgba(61, 220, 151, 0.12);
    color: #3DDC97;
    border: 1px solid rgba(61, 220, 151, 0.4);
    box-shadow: 0 0 12px rgba(61, 220, 151, 0.2);
}

.status-pill.badge-changed {
    background: rgba(255, 92, 92, 0.12);
    color: #FF5C5C;
    border: 1px solid rgba(255, 92, 92, 0.4);
    box-shadow: 0 0 12px rgba(255, 92, 92, 0.2);
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

.badge-preserved .status-dot {
    background: #3DDC97;
    box-shadow: 0 0 6px #3DDC97;
}

.badge-changed .status-dot {
    background: #FF5C5C;
    box-shadow: 0 0 6px #FF5C5C;
}

/* Prediction Cards Grid */
.pred-cards-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 14px;
}

.pred-card {
    background: #15151B;
    border: 1px solid #2A2A33;
    border-radius: 14px;
    padding: 18px 20px;
}

.pred-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid #22222B;
}

.card-title {
    font-size: 14px;
    font-weight: 700;
    color: #F5F5F7;
    letter-spacing: -0.01em;
}

.card-badge {
    font-size: 12px;
    font-weight: 600;
    color: #9A9AA5;
    background: #1C1C24;
    padding: 3px 10px;
    border-radius: 6px;
    border: 1px solid #2A2A33;
}

.card-badge.highlight {
    color: #FFB400;
    border-color: rgba(255, 180, 0, 0.3);
    background: rgba(255, 180, 0, 0.08);
}

.pred-rows-container {
    display: flex;
    flex-direction: column;
    gap: 14px;
}

.pred-row {
    display: flex;
    flex-direction: column;
    gap: 5px;
}

.pred-row-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.class-label {
    font-size: 13.5px;
    font-weight: 500;
    color: #F5F5F7;
    display: flex;
    align-items: center;
    gap: 6px;
}

.winner-row .class-label {
    font-weight: 700;
    color: #FFFFFF;
}

.class-pct {
    font-size: 13px;
    font-weight: 700;
    color: #FFB400;
    font-variant-numeric: tabular-nums;
}

.progress-track {
    width: 100%;
    height: 7px;
    background: #22222B;
    border-radius: 4px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #FFB400 0%, #FFC933 100%);
    border-radius: 4px;
    box-shadow: 0 0 8px rgba(255, 180, 0, 0.35);
    transition: width 0.3s ease;
}

/* Agreement dots */
.agree-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
}

.agree-dot.preserved {
    background: #3DDC97;
    box-shadow: 0 0 5px #3DDC97;
}

.agree-dot.changed {
    background: #FF5C5C;
    box-shadow: 0 0 5px #FF5C5C;
}

/* Examples gallery styling */
.gallery {
    border-radius: 12px !important;
    background: #15151B !important;
}

/* Responsive / Mobile layout */
@media (max-width: 900px) {
    .image-panels-row {
        flex-direction: column !important;
    }
    .pred-cards-grid {
        grid-template-columns: 1fr !important;
    }
    .stats-strip {
        grid-template-columns: 1fr !important;
        gap: 12px;
    }
}
"""


# =============================================================================
# HTML Render Helpers
# =============================================================================

def render_stats_html(kept: int, total: int, agree: bool, kl: float) -> str:
    pct = round((kept / total) * 100) if total else 0
    if agree:
        badge_cls = "badge-preserved"
        status_text = "&check; Prediction preserved"
    else:
        badge_cls = "badge-changed"
        status_text = "&times; Prediction changed"

    return f"""
    <div class="stats-strip">
        <div class="stat-card">
            <div class="stat-title">Patches Retained</div>
            <div class="stat-num">
                <span class="amber-glow">{kept}</span>
                <span class="stat-denom">/ {total}</span>
                <span class="stat-badge-pct">({pct}%)</span>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-title">KL Divergence</div>
            <div class="stat-num">
                <span class="amber-glow">{kl:.4f}</span>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-title">Prediction Invariance</div>
            <div class="status-pill {badge_cls}">
                <span class="status-dot"></span>
                <span class="status-text">{status_text}</span>
            </div>
        </div>
    </div>
    """


def render_predictions_html(
    full_probs: dict[str, float],
    red_probs: dict[str, float],
    kept: int,
    total: int,
    agree: bool,
) -> str:
    top3_full = sorted(full_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_red = sorted(red_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    pct = round((kept / total) * 100) if total else 0

    def build_rows(items: list[tuple[str, float]], is_reduced: bool) -> str:
        html_rows = []
        for rank, (cls_name, prob) in enumerate(items, 1):
            pct_val = prob * 100.0
            is_winner = (rank == 1)
            row_cls = "winner-row" if is_winner else ""
            dot_html = ""
            if is_winner and is_reduced:
                dot_type = "preserved" if agree else "changed"
                dot_html = f'<span class="agree-dot {dot_type}"></span>'

            html_rows.append(f"""
                <div class="pred-row {row_cls}">
                    <div class="pred-row-header">
                        <span class="class-label">{rank}. {cls_name} {dot_html}</span>
                        <span class="class-pct">{pct_val:.1f}%</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: {min(100.0, max(1.5, pct_val)):.1f}%;"></div>
                    </div>
                </div>
            """)
        return "".join(html_rows)

    full_rows_html = build_rows(top3_full, is_reduced=False)
    red_rows_html = build_rows(top3_red, is_reduced=True)

    return f"""
    <div class="pred-cards-grid">
        <div class="pred-card">
            <div class="pred-card-header">
                <div class="card-title">Full Image (Unmasked)</div>
                <div class="card-badge">196 patches (100%)</div>
            </div>
            <div class="pred-rows-container">
                {full_rows_html}
            </div>
        </div>
        <div class="pred-card">
            <div class="pred-card-header">
                <div class="card-title">Reduced Image (Sparsified)</div>
                <div class="card-badge highlight">{kept} patches ({pct}%)</div>
            </div>
            <div class="pred-rows-container">
                {red_rows_html}
            </div>
        </div>
    </div>
    """


# =============================================================================
# Main Demo Callback
# =============================================================================

def run_preservenet(img: Any, retention: float | int):
    """
    Executes inference via Predictor.run(img, retention) and formats outputs.
    """
    if img is None:
        gr.Warning("Please upload an image or choose one of the examples below.")
        empty_stats = """
        <div class="stats-strip">
            <div class="stat-card"><div class="stat-title">Status</div><div class="stat-num" style="font-size:14px;color:#9A9AA5;">No image selected</div></div>
        </div>
        """
        empty_preds = """
        <div class="pred-cards-grid">
            <div class="pred-card"><div class="card-title">Awaiting Input</div></div>
            <div class="pred-card"><div class="card-title">Awaiting Input</div></div>
        </div>
        """
        return None, None, None, empty_stats, empty_preds

    # Call inference engine matching the required contract:
    # Predictor.run(img, retention) -> dict of:
    # orig, overlay, reduced, full_probs, red_probs, kept, total, agree, kl
    result = Predictor.run(img, retention)

    orig_img = result['orig']
    overlay_img = result['overlay']
    reduced_img = result['reduced']
    full_probs = result['full_probs']
    red_probs = result['red_probs']
    kept = result['kept']
    total = result['total']
    agree = result['agree']
    kl = result['kl']

    stats_html = render_stats_html(kept, total, agree, kl)
    preds_html = render_predictions_html(full_probs, red_probs, kept, total, agree)

    return orig_img, overlay_img, reduced_img, stats_html, preds_html


# =============================================================================
# Build Gradio Blocks Application
# =============================================================================

def build_app() -> gr.Blocks:
    # Locate example images
    ex_dog = APP_DIR / 'examples' / 'dog_springer.jpg'
    ex_church = APP_DIR / 'examples' / 'church.jpg'
    ex_parachute = APP_DIR / 'examples' / 'parachute.jpg'

    examples = []
    for p, r in [(ex_dog, 25), (ex_church, 30), (ex_parachute, 20)]:
        if p.is_file():
            examples.append([str(p), r])

    with gr.Blocks(
        title="PreserveNet — Prediction-Preserving Image Reduction",
        css=CUSTOM_CSS,
    ) as demo:
        # Header
        gr.HTML("""
        <div class="pn-header">
            <h1>Preserve<span class="wordmark-net">Net</span></h1>
            <p class="tagline">Spatial Redundancy Probing: Measuring classifier prediction invariance under extreme 16×16 patch drop</p>
        </div>
        """)

        with gr.Row():
            # Left Column: Inputs & Controls
            with gr.Column(scale=4):
                image_input = gr.Image(
                    type="pil",
                    label="Input Image",
                    elem_id="image-input",
                    height=280,
                )

                retention_slider = gr.Slider(
                    minimum=5,
                    maximum=100,
                    step=5,
                    value=25,
                    label="Patches retained (%)",
                    elem_id="retention-slider",
                )

                run_btn = gr.Button(
                    "Reduce & classify",
                    variant="primary",
                    elem_id="reduce-btn",
                )

                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=[image_input, retention_slider],
                        label="Clickable Examples",
                        examples_per_page=3,
                    )

            # Right Column: Visual Outputs & Predictions
            with gr.Column(scale=7):
                # Three equal image panels side by side
                with gr.Row(equal_height=True, elem_classes=["image-panels-row"]):
                    out_orig = gr.Image(
                        label="Original",
                        interactive=False,
                        height=210,
                    )
                    out_overlay = gr.Image(
                        label="Selected patches overlay",
                        interactive=False,
                        height=210,
                    )
                    out_reduced = gr.Image(
                        label="Model input",
                        interactive=False,
                        height=210,
                    )

                # Stats Strip
                stats_output = gr.HTML(
                    value=render_stats_html(49, 196, True, 0.0),
                    elem_id="stats-strip-container",
                )

                # Top-3 Prediction Cards (Full vs Reduced)
                preds_output = gr.HTML(
                    value="""
                    <div class="pred-cards-grid">
                        <div class="pred-card">
                            <div class="pred-card-header">
                                <div class="card-title">Full Image (Unmasked)</div>
                                <div class="card-badge">196 patches (100%)</div>
                            </div>
                            <div style="color:#9A9AA5;font-size:13px;padding:12px 0;">Upload an image to inspect top predictions.</div>
                        </div>
                        <div class="pred-card">
                            <div class="pred-card-header">
                                <div class="card-title">Reduced Image (Sparsified)</div>
                                <div class="card-badge highlight">25% retention</div>
                            </div>
                            <div style="color:#9A9AA5;font-size:13px;padding:12px 0;">Upload an image to inspect top predictions.</div>
                        </div>
                    </div>
                    """,
                    elem_id="predictions-container",
                )

        # Connect actions
        run_btn.click(
            fn=run_preservenet,
            inputs=[image_input, retention_slider],
            outputs=[out_orig, out_overlay, out_reduced, stats_output, preds_output],
        )

        # Trigger on example image load if available
        if examples:
            demo.load(
                fn=run_preservenet,
                inputs=[
                    gr.State(str(ex_dog) if ex_dog.is_file() else None),
                    gr.State(25),
                ],
                outputs=[out_orig, out_overlay, out_reduced, stats_output, preds_output],
            )

    return demo


# Create application instance
demo = build_app()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CUSTOM_CSS,
        show_error=True,
    )
