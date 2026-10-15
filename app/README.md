---
title: PreserveNet
emoji: ⚡
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Probing spatial redundancy in vision classifiers via patch reduction
---

# PreserveNet — Prediction-Preserving Image Reduction

How many 16×16 image patches can be removed before a classifier changes its prediction?

PreserveNet probes spatial redundancy in deep vision models by learning a lightweight **PatchScorer** that ranks and retains only the most critical spatial patches while preserving classifier predictions and confidence distributions.

## How It Works
1. **Upload an Image**: Pick one of the provided Imagenette examples or upload your own 224×224 image.
2. **Set Retention Rate**: Use the slider to specify how many patches to retain (5% to 100%, step 5%).
3. **Inspect the Visuals & Metrics**:
   - **Original**: The unmasked input image.
   - **Selected patches overlay**: Dimmed dropped patches (15% brightness) with kept patches tinted amber (25%).
   - **Model input**: Sparsified image fed into the classifier.
   - **Stats strip**: Retained patch count (out of 196), KL divergence, and Prediction Preserved/Changed status badge.
   - **Confidence cards**: Side-by-side top-3 predictions comparing full image vs reduced image.

## Technology Stack
- **Reducer**: PreserveNet lightweight PatchScorer CNN (~110K parameters)
- **Classifier**: Zero-shot ResNet-18 ImageNet backbone
- **UI**: Gradio 6.0 with custom dark theme and amber accents
