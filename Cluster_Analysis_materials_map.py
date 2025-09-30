import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import constants
from adjustText import adjust_text  # neu importieren
import pandas as pd
from adjustText import adjust_text

#Sk Learn
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set good theme
sns.set_theme(style="whitegrid", context="talk")


# ---------- load & convert ----------
df = pd.read_csv("material_wavelength_gain_full.csv")

#Chosing features with only numerical values

X = df.drop(columns=['Material','Wavelength_nm'])

# Implementing standardization with Scalerclass

scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)
#print(scaled_features[:5])

# Implementing k means
kmeans = KMeans(init = 'random',
                n_clusters = 3,
                n_init = 10,
                max_iter = 300,
                random_state = 42
                )

kmeans.fit(scaled_features)

# --- Elbow + Silhouette zur Wahl von k ---
K = range(2, 10)  # teste k=2..9 (bei Bedarf anpassen)
sse = []
sil_scores = {}

for k in K:
    km = KMeans(
        n_clusters=k,
        init="random",
        n_init=10,
        max_iter=300,
        random_state=42
    )
    labels = km.fit_predict(scaled_features)
    sse.append(km.inertia_)
    sil_scores[k] = silhouette_score(scaled_features, labels)

# Elbow plot
plt.figure(figsize=(7, 4))
plt.plot(list(K), sse, "o", color="steelblue", linewidth=2, markersize=7)

plt.xlabel("Number of clusters k", fontsize=12)
plt.ylabel("SSE (Within-Cluster Sum of Squares)", fontsize=12)
plt.title("Elbow Method for Optimal k")

plt.xticks(list(K))
plt.grid(True, alpha=0.5)

plt.tight_layout()
plt.savefig("Elbow_Method.pdf")
plt.show()

# Silhouette plot
plt.figure(figsize=(7, 4))
plt.plot(list(sil_scores.keys()), list(sil_scores.values()), marker= 'o')

plt.xlabel("Number of clusters k", fontsize=12)
plt.ylabel("Silhouette score", fontsize=12)
plt.title("Silhouette Analysis for Optimal k (higher is better)")

plt.xticks(list(K))
plt.grid(True, alpha=0.5)

plt.tight_layout()
plt.savefig("Silhouette_Analysis.pdf")
plt.show()

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

scores = {"k": [], "silhouette": [], "calinski": [], "davies": []}

for k in range(2, 11):  # Beispielbereich
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    labels = kmeans.labels_

    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)

    scores["k"].append(k)
    scores["silhouette"].append(sil)
    scores["calinski"].append(ch)
    scores["davies"].append(db)

# Normalisieren (DBI invertieren, weil niedriger besser)
sil_norm = (np.array(scores["silhouette"]) - min(scores["silhouette"])) / (
            max(scores["silhouette"]) - min(scores["silhouette"]))
ch_norm = (np.array(scores["calinski"]) - min(scores["calinski"])) / (max(scores["calinski"]) - min(scores["calinski"]))
db_norm = 1 - (np.array(scores["davies"]) - min(scores["davies"])) / (max(scores["davies"]) - min(scores["davies"]))

combined = sil_norm + ch_norm + db_norm

best_k = scores["k"][np.argmax(combined)]

print("Bester k nach Multi-Metrik:", best_k)


# Set your range of k
k_values = list(range(2, 11))  # tweak as needed

# Storage
sse = []           # for Elbow (WCSS / inertia)
sil = []           # Silhouette (higher better)
ch = []            # Calinski–Harabasz (higher better)
dbi = []          # Davies–Bouldin (lower better)

# Compute metrics
for k in k_values:
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    sse.append(km.inertia_)
    sil.append(silhouette_score(X, labels))
    ch.append(calinski_harabasz_score(X, labels))
    dbi.append(davies_bouldin_score(X, labels))


plt.figure(figsize=(6,4))
plt.plot(k_values, ch, marker = 'o')
plt.title('Calinski–Harabasz vs k (higher is better)')
plt.xlabel('Number of clusters k'); plt.ylabel('Calinski–Harabasz')
plt.tight_layout()
plt.savefig("calinsky_k.pdf")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(k_values, dbi, marker='o')
plt.title('Davies–Bouldin for optimal k (lower is better)')
plt.xlabel('Number of clusters k'); plt.ylabel('Davies–Bouldin')
plt.tight_layout()
plt.savefig("davies_bouldin_k.pdf")
plt.show()

# --- KMeans mit gewähltem k ---
kmeans = KMeans(
    n_clusters=best_k,
    init="random",
    n_init=10,
    max_iter=300,
    random_state=42
)
kmeans.fit(scaled_features)

# === 1) PCA auf 2D ===
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(scaled_features)

# Loadings als Vektoren (für Biplot nimmt man meist components_*sqrt(var))
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)  # shape: (n_features, 2)
load_df = pd.DataFrame(loadings, index=X.columns, columns=["PCA1","PCA2"])

# Ergebnisse zurück ins df
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]
df["Cluster"] = kmeans.labels_

# === 2) Plot vorbereiten ===
plt.figure(figsize=(12, 8))

# Punkte + Clusterfarben
sns.scatterplot(
    data=df, x="PCA1", y="PCA2",
    hue="Cluster", palette="bright", s=40, edgecolor="black", linewidth=0.2, alpha=0.85
)

# Materialnamen mit Overlap-Vermeidung
texts = []
for _, row in df.iterrows():
    texts.append(plt.text(row["PCA1"], row["PCA2"], row["Material"], fontsize=8))
adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.6))

# === 3) Feature-Vektoren (Biplot-Pfeile) ===
# Skalierung der Pfeile für angenehme Länge
vector_scale = 2.5  # bei Bedarf anpassen


# Mapping: Originalname -> Anzeige im Biplot
feature_labels = {
    "Gain": "Gain",
    "Band_Gap_eV": "Band Gap",
    "Density_g_cm3": "Electron Density",
    "Lattice_parameter_A": "Lattice Parameter"
}

for feat, (vx, vy) in load_df.iterrows():
    label = feature_labels.get(feat, feat)  # falls nicht im Mapping, nimm Original
    plt.arrow(
        0, 0, vx * vector_scale, vy * vector_scale,
        color="crimson", alpha=0.7,
        width=0.003, head_width=0.08, head_length=0.1,
        length_includes_head=True
    )
    plt.text(
        vx * vector_scale * 1.1, vy * vector_scale * 1.1, label,
        fontsize=10, color="crimson",
        bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3")
    )
# === 4) Achsen / Titel ===
var_ratio = pca.explained_variance_ratio_
plt.xlabel(f"PCA 1  ({var_ratio[0]*100:.1f}% Variance)")
plt.ylabel(f"PCA 2  ({var_ratio[1]*100:.1f}% Variance)")
plt.title("Materials Map using PCA-Biplot with K-Means Clustering")
plt.axhline(0, color="lightgray", lw=0.7)
plt.axvline(0, color="lightgray", lw=0.7)
plt.legend([],[],frameon=False)
plt.tight_layout()
plt.savefig("PCABiplot.pdf")
plt.show()



plt.close()

# Mapping für neue Labels
rename = {
    "Gain": "Gain",
    "Lattice_parameter_A": "Lattice Parameter",
    "Band_Gap_eV": "Band Gap",
    "Density_g_cm3": "Density"
}

# DataFrame mit umbenannten Spalten
df_renamed = df.rename(columns=rename)

# Korrelation berechnen
corr = df_renamed[rename.values()].corr()

# Heatmap zeichnen
plt.figure(figsize=(6,5))
sns.set_theme(style="white", context="talk")

ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap="flare",
    annot=True, fmt=".2f", annot_kws={"size":10},
    linewidths=0.5, linecolor="white",
    square=True, cbar_kws={"shrink":0.8, "label":"Correlation"}
)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
plt.title("Feature Correlation")
plt.tight_layout()
plt.savefig("Correlation.pdf")
plt.show()






# Optional: Loadings-Tabelle ansehen
print("\nLoadings (skalierte Beiträge der Features auf die PCs):")
print(load_df.round(3))



