# E-commerce Returns Prediction Challenge

## Overview
This repository contains my solution to a technical challenge focused on predicting product returns in an e-commerce context. The goal was not just to build a model, but to demonstrate a structured, production-aware approach: evaluating a misleading baseline, improving it with clear hypotheses, validating results properly, and planning for real-world deployment.

The emphasis throughout the exercise is on interpretability, methodological rigor, and business relevance rather than model complexity.

## Approach
I started from a simple logistic regression baseline that appeared strong in terms of accuracy but completely failed to identify returns. After validating this behavior, I focused on improving the model in a controlled and explainable way. The main steps included fixing data leakage in preprocessing, handling class imbalance, applying light regularization tuning, and validating generalization through consistent train/test evaluation.

All changes were incremental and driven by explicit hypotheses rather than blind optimization.

## Key Results
The baseline model achieved roughly 75% accuracy but had zero recall for returns, making it unusable in practice. The improved model intentionally trades accuracy for business-relevant signal, achieving approximately 51% recall on returns while maintaining stable performance between training and test data. This confirms that the improvements generalize and do not overfit.

## Business Perspective
The improved model enables early identification of high-risk orders, which can support targeted actions such as manual review, fulfillment adjustments, or proactive customer communication. Even with moderate precision, the ability to catch a meaningful portion of returns can translate into reduced logistics and handling costs when applied selectively.

## Deployment Readiness
The solution is production-oriented and includes persisted model and preprocessing artifacts, clear monitoring metrics, retraining guidelines, and rollback criteria. This allows the model to be deployed safely, monitored continuously, and iterated on with low operational risk.


## Files Description
- **lastname_firstname_challenge.ipynb**: End-to-end notebook with code, results, and reasoning.
- **summary.md**: Executive summary covering approach, findings, business impact, and deployment recommendations.
- **Model artifacts (.pkl)**: Trained model and preprocessing components required for reproducible inference.

## Final Notes
This project reflects how I approach applied machine learning problems: understanding the baseline first, making deliberate trade-offs, validating rigorously, and always aligning technical decisions with business impact and deployment realities.




