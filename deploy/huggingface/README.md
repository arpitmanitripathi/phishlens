---
title: PhishLens
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8000
pinned: false
short_description: Phishing URL detection with LightGBM + Char-CNN
---

# PhishLens

**Real-time phishing URL detection with an explainable LightGBM + CNN ensemble.**

Paste a link and PhishLens says whether it looks like phishing, gives a risk score from 0 to 100
and explains why. The link is never opened: only the text of the address is analysed.

- **Models:** LightGBM on 29 URL features + a character-level CNN (ONNX), soft voting 50 / 50
- **Test set** (5,902 URLs from websites never seen in training): **92.7 % F1**,
  93.9 % precision, 91.6 % recall, 6.0 % false-positive rate
- **API:** `POST /api/check` with `{"url": "..."}` · docs at `/docs`

Code, data pipeline and full report: https://github.com/arpitmanitripathi/phishlens
