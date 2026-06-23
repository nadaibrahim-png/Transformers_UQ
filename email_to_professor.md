Subject: Re: ICTP Presentation — Abstract, Technical Summary & Review Plan

Dear Professor,

Thank you very much for your congratulations and for agreeing to guide me through this presentation. It means a great deal to have your support.

As requested, please find below the three items that summarise my contribution:

---

**1. Abstract submitted to ICTP**

*Title: TOWARD ROBUST DEEP LEARNING: EVALUATING UNCERTAINTY IN TRANSFORMER-BASED ARCHITECTURES FOR SCIENTIFIC DATA*

Deep Learning (DL) models, specifically Convolutional Neural Networks (CNNs) and Transformers, have demonstrated significant efficacy across various scientific domains. However, their integration into critical research is often constrained by the "black-box" nature of these models and their tendency to produce overconfident predictions when processing noisy or out-of-distribution datasets. This study investigates the implementation of robust DL architectures, focusing on the fine-tuning of Transformer models to optimize predictive accuracy while ensuring rigorous model calibration. We analyze the propagation of uncertainty through deep layers and evaluate the impact of targeted feature selection on overall model stability. By comparing standard architectures with uncertainty-aware configurations, this research establishes a methodology for developing more reliable and interpretable computational pipelines. Preliminary results indicate that incorporating Bayesian-inspired uncertainty estimation significantly enhances the robustness of Transformer models in high-dimensional scientific environments, offering a scalable path toward more sustainable and portable machine learning solutions.

---

**2. Technical approach — 30-second summary**

- **Dataset:** MAGIC Gamma Telescope (astroparticle physics) — classifying cosmic ray events as gamma signal vs. hadron background
- **Model:** Transformer encoder for tabular scientific data (TabTransformer), implemented in PyTorch
- **Experiment A — Calibration:**
  - Baseline: standard inference → quantify overconfidence using Expected Calibration Error (ECE) and reliability diagrams
  - MC Dropout (Gal & Ghahramani, 2016): 30 stochastic forward passes at test time → improved ECE
  - Temperature Scaling (Guo et al., 2017): single-parameter post-hoc fix → best calibration, no retraining needed
  - OOD evaluation: model assigns high predictive entropy to corrupted inputs, confirming uncertainty awareness
- **Experiment B — Active Learning:**
  - Pool-based active learning loop comparing three acquisition strategies: Random (baseline), Maximum Entropy, and BALD — Bayesian Active Learning by Disagreement (Gal et al., 2017)
  - BALD repurposes MC Dropout uncertainty to select the most epistemically informative samples for labeling
  - Key result: entropy and BALD-based selection reach high accuracy with significantly fewer labeled samples than random selection
- **Codebase:** Modular Python project (data / model / training / evaluation / active_learning / visualization), hosted on GitHub, experiments reproducible on Google Colab

---

**3. Review plan**

| Date | Milestone |
|------|-----------|
| Now – 4 July | Study key papers + run experiments (ECE, reliability diagrams, learning curves) |
| 7 July | Send you the first draft of poster layout and key figures for feedback |
| 14 July | Incorporate your feedback + finalize poster content |
| 18 July | Send final poster for your review before printing |
| 23 July | Present at ICTP, Trieste |

I will send you the initial draft of the poster layout and the main experimental figures on **Monday, 7 July** — I would be very grateful for your feedback on the technical results and overall clarity before I finalize the design.

Thank you again for your time and support. I look forward to making you proud at ICTP.

Best regards,
Nada Ibrahim Abdelrahman Elhassan
M.Sc. Data Science Student
University of Messina, Italy
MAECI Scholar | Erasmus+ Grantee
