import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

original_file_path = r'parkinsons_raw_data\parkinsons.data'
data_df = pd.read_csv(original_file_path)
data_df.to_csv(r'parkinsons_raw_data\parkinsons_data.csv', index=False)

file_path = r'parkinsons_raw_data\parkinsons_data.csv'

raw_df = pd.read_csv(file_path)
numeric_df = raw_df.apply(pd.to_numeric, errors='coerce').replace(-9999, np.nan)

nan_pct = numeric_df.isna().mean().sort_values()
good_var = nan_pct[nan_pct <= 0.1]
print(good_var)

target_var = [
    'RPDE',
    'DFA',
    'spread1',
    'spread2',
    'PPE',
]

# Select just the target columns
var_df = numeric_df[target_var]

# Keep only rows where ALL target vars are present
non_na = var_df.dropna()
X = non_na.to_numpy(dtype=float)

# Scale
X_df = non_na[target_var].astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

scaled_df = pd.DataFrame(X_scaled, columns=target_var, index=X_df.index)

# Visual Representation of standardized data

g = sns.pairplot(
    data=scaled_df, 
    diag_kind='kde'
    )
g.map_upper(sns.kdeplot)
plt.show()

# Model selection via BIC/AIC
bic, aic = [], []
gmms = {}

for k in range(1, 11):
    gmm_k = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    gmm_k.fit(X_scaled)
    bic.append(gmm_k.bic(X_scaled))
    aic.append(gmm_k.aic(X_scaled))
    gmms[k] = gmm_k

bic = np.array(bic)
bic_min = bic.min()
bic_std = bic.std()

acceptable = np.where(bic <= bic_min + bic_std)[0] + 1
print("Acceptable K values:", acceptable)

best_k = int(np.argmin(bic)) + 1   # since k starts at 1
best_gmm = gmms[best_k]

means = best_gmm.means_
covariances = best_gmm.covariances_
weights = best_gmm.weights_

# Ks for range 1,11
Ks = np.arange(1, len(bic)+1)

# BIC/AIC Selection Visualization
plt.figure(figsize=(7,4))
plt.plot(Ks, bic, marker='o', label='BIC')
plt.plot(Ks, aic, marker='o', label='AIC')
plt.xlabel("Number of components (K)")
plt.ylabel("Criterion (lower = better)")
plt.title("GMM model selection")
plt.xticks(Ks)
plt.legend()
plt.tight_layout()
plt.show()

# BIC
plt.figure(figsize=(7,4))
plt.plot(Ks, bic, marker='o')
plt.xticks(Ks)
plt.xlabel("K")
plt.ylabel("BIC (lower is better)")
plt.title("BIC vs K")
plt.axvline(Ks[int(np.argmin(bic))], linestyle='--', linewidth=1)
plt.tight_layout()
plt.show()

# AIC
plt.figure(figsize=(7,4))
plt.plot(Ks, aic, marker='o')
plt.xticks(Ks)
plt.xlabel("K")
plt.ylabel("AIC (lower is better)")
plt.title("AIC vs K")
plt.axvline(Ks[int(np.argmin(aic))], linestyle='--', linewidth=1)
plt.tight_layout()
plt.show()

# Save clusters/probabilities from best_gmm
best_labels = best_gmm.predict(X_scaled)
best_probs = best_gmm.predict_proba(X_scaled)

for j in range(best_k):
    numeric_df.loc[non_na.index, f'cluster_{j}_prob'] = best_probs[:, j]

# Predict a new subject using the BEST model - DEMONSTRATION ONLY
new_subject = pd.DataFrame([{
    'RPDE':np.average(numeric_df['RPDE']),
    'DFA':np.average(numeric_df['DFA']),
    'spread1':np.average(numeric_df['spread1']),
    'spread2':np.average(numeric_df['spread2']),
    'PPE':np.average(numeric_df['PPE']),
    }], 
    columns=target_var)
new_scaled = scaler.transform(new_subject)

print("new_cluster:", best_gmm.predict(new_scaled))
print("new_prob:", best_gmm.predict_proba(new_scaled))

counts = np.bincount(best_labels, minlength=best_k)
print("cluster counts: ", counts)
print("cluster proportions: ", counts / counts.sum())
print(f'Optimal K: {best_k}\n\n---\n')


####################################################################################


# Pull diagnoses for same rows used in GMM
dx = numeric_df.loc[non_na.index, "status"]

# Crosstab diagnosis x cluster
ct = pd.crosstab(
    dx, 
    best_labels,
    normalize="index"
    )
print(ct)


# Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(ct, annot=True)
plt.title("DX_GROUP vs GMM cluster (row-normalized)")
plt.ylabel("DX_GROUP")
plt.xlabel("GMM cluster")
plt.show()

# Quantify Alignment
dx_arr = dx.to_numpy()
mask = ~np.isnan(dx_arr)
print("NMI:", normalized_mutual_info_score(dx_arr[mask], best_labels[mask]))
print("ARI:", adjusted_rand_score(dx_arr[mask], best_labels[mask]))


print("dx length:", len(dx), " | dx non-null:", dx.notna().sum(), " | dx unique:", dx.nunique(dropna=True))
print("dx value counts:\n", dx.value_counts(dropna=False).head(10))
