# Gaussian Mixture Modeling for Parkinson’s Disease Stratification

This project applies **unsupervised machine learning** to explore latent structure in Parkinson’s disease (PD) using nonlinear biomedical voice features. Gaussian Mixture Modeling (GMM) is used to determine whether clinically diagnosed groups emerge naturally from acoustic biomarkers.

---

## Dataset

Data are drawn from:

> *'Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection', 
Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM. 
BioMedical Engineering OnLine 2007, 6:23 (26 June 2007)*

- 195 total voice recordings  
- 23 Parkinson’s disease participants  
- 31 healthy controls  
- ~6 recordings per subject  

---

## Features Used

The model focuses on established nonlinear speech biomarkers:

- **RPDE** — nonlinear dynamical complexity  
- **DFA** — fractal scaling exponent  
- **spread1, spread2** — frequency variation measures  
- **PPE** — pitch period entropy  

Only variables with ≤10% missingness were retained.

---

## Methodology

1. **Preprocessing**
   - Numeric coercion and cleaning
   - Removal of invalid values
   - Feature standardization (StandardScaler)

2. **Exploratory Analysis**
   - Pair plots and kernel density estimations (KDE)

3. **Gaussian Mixture Modeling**
   - Models trained across K = 1–10 components
   - Full covariance structure
   - Model selection using **BIC and AIC**
   - BIC-optimal K used for final model

4. **Evaluation**
   - Probabilistic cluster assignments
   - Diagnosis × cluster cross-tabulation
   - Normalized Mutual Information (NMI)
   - Adjusted Rand Index (ARI)

A demonstration subject is also scaled and probabilistically assigned to the learned model.

---

## Technologies

- Python  
- pandas, NumPy  
- scikit-learn  
- Matplotlib, Seaborn  

---

## Professional Relevance

This project demonstrates:

- Unsupervised learning for biomedical data
- Statistical model selection (BIC/AIC)
- Probabilistic clustering and interpretation
- Research-oriented data analysis workflows

Designed as an exploratory modeling project, not a clinical diagnostic tool.
