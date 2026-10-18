# ==========================================================================
# TUGAS BESAR MACHINE LEARNING TEKNIK INDUSTRI (PENGGANTI UTS)
# Anggota Kelompok: Fitri Izzati / I0324008 & Galuh Chandra Maulida / I0324009
# ==========================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme()
RANDOM_STATE = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_analitika_data") if "__file__" in globals() else "output_analitika_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FILE_NAME = "garments_worker_productivity.csv"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
LOCAL_CSV = os.path.join(SCRIPT_DIR, FILE_NAME)
if os.path.exists(LOCAL_CSV):
    FILE_NAME = LOCAL_CSV

# Jika file belum ada di runtime, minta pengguna mengunggahnya (Google Colab).
if not os.path.exists(FILE_NAME):
    try:
        from google.colab import files
        print("Unggah file garments_worker_productivity.csv:")
        uploaded = files.upload()
        FILE_NAME = list(uploaded.keys())[0]
    except ImportError:
        raise FileNotFoundError(
            "Letakkan garments_worker_productivity.csv di folder yang sama dengan notebook ini."
        )

df = pd.read_csv(FILE_NAME)
print("Dataset:", FILE_NAME)
print("--- Display 5 Data Teratas ---")
df.head()
print("Ukuran data (baris, kolom):", df.shape)
print("\nTipe data:")
print(df.dtypes)
print("\nNilai kosong per kolom:")
print(df.isnull().sum())
print("\nBaris duplikat:", int(df.duplicated().sum()))
print("\nIsi kolom department (mentah):", df["department"].unique().tolist())
print("\nStatistik deskriptif:")
df.describe().T
# 4.1 Rapikan teks kategorikal
df.columns = df.columns.str.strip()
for col in df.select_dtypes(exclude="number").columns:
    df[col] = df[col].str.strip()
df["department"] = df["department"].replace({"sweing": "sewing"})
print("department setelah dirapikan:\n", df["department"].value_counts())

# 4.2 Duplikat
n_dup = int(df.duplicated().sum())
df = df.drop_duplicates().copy()
print("\nDuplikat dihapus:", n_dup)

# 4.3 Target kosong / tidak valid
target = "actual_productivity"
df[target] = pd.to_numeric(df[target], errors="coerce")
n_bad_target = int(df[target].isna().sum())
df = df.dropna(subset=[target]).copy()
print("Baris dengan target kosong dihapus:", n_bad_target)

# 4.4 Pemeriksaan targeted_productivity
# Nilai di bawah 0.3 tidak otomatis tidak valid, sehingga tidak dihapus tanpa dasar sumber.
df["targeted_productivity"] = pd.to_numeric(df["targeted_productivity"], errors="coerce")
print("\nNilai terkecil targeted_productivity:", df["targeted_productivity"].min())
print("Nilai kosong targeted_productivity:", int(df["targeted_productivity"].isna().sum()))

# 4.5 Pola nilai kosong wip
print("\nPersentase wip kosong per departemen:")
print((df.groupby("department")["wip"].apply(lambda s: s.isna().mean() * 100)).round(1))

print("\nUkuran data setelah cleansing:", df.shape)
# 4.6 Penandaan outlier dengan IQR (tidak dihapus)
numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != "team"]
rows = []
for col in numeric_cols:
    v = df[col].dropna()
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    rows.append({"Variabel": col, "Batas_Bawah": lo, "Batas_Atas": hi,
                 "Jumlah_Outlier": int(((v < lo) | (v > hi)).sum()),
                 "Persen_Outlier": round(((v < lo) | (v > hi)).mean() * 100, 1)})
outlier_df = pd.DataFrame(rows)
outlier_df.to_csv(f"{OUTPUT_DIR}/ringkasan_outlier_iqr.csv", index=False)
outlier_df
# Grafik 1: distribusi target
plt.figure(figsize=(9, 5))
sns.histplot(df[target], bins=25, kde=True, color="steelblue")
plt.title("Distribusi Produktivitas Aktual")
plt.xlabel("Produktivitas Aktual")
plt.ylabel("Jumlah Observasi")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_distribusi_produktivitas.png", dpi=200)
plt.show()
print(df[target].describe().round(4))
# Grafik 2: target vs aktual
plt.figure(figsize=(8, 5.5))
sns.scatterplot(data=df, x="targeted_productivity", y=target, hue="department", alpha=0.6)
plt.title("Target Produktivitas vs Produktivitas Aktual")
plt.xlabel("Target Produktivitas")
plt.ylabel("Produktivitas Aktual")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_target_vs_aktual.png", dpi=200)
plt.show()
# Grafik 3: heatmap korelasi (variabel numerik, tanpa ID tim)
corr = df[numeric_cols].corr()
plt.figure(figsize=(11, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Korelasi Antarvariabel Numerik")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_heatmap_korelasi.png", dpi=200)
plt.show()
print("Korelasi dengan target:")
print(corr[target].drop(target).sort_values().round(3))
# Grafik 4: produktivitas per departemen dan per jumlah perubahan style
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x="department", y=target, ax=axes[0])
axes[0].set_title("Produktivitas Aktual per Departemen")
axes[0].set_xlabel("Departemen")
axes[0].set_ylabel("Produktivitas Aktual")
sns.boxplot(data=df, x="no_of_style_change", y=target, ax=axes[1])
axes[1].set_title("Produktivitas Aktual per Jumlah Perubahan Style")
axes[1].set_xlabel("Jumlah Perubahan Style")
axes[1].set_ylabel("Produktivitas Aktual")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_boxplot_kategori.png", dpi=200)
plt.show()
print(df.groupby("department")[target].agg(["count", "mean", "median"]).round(4))
print(df.groupby("no_of_style_change")[target].agg(["count", "mean"]).round(4))
model_df = df.copy()
# Parsing tanggal dibuat eksplisit; baris dengan tanggal tidak terbaca tidak otomatis dihapus.
parsed_date = pd.to_datetime(model_df["date"], errors="coerce")
if parsed_date.isna().any():
    print("Peringatan: tanggal yang gagal dibaca:", int(parsed_date.isna().sum()))
model_df["month"] = parsed_date.dt.month
model_df = model_df.drop(columns=["date"])
model_df["team"] = model_df["team"].astype("string")

# targeted_productivity dipertahankan sebagai fitur dengan asumsi target kerja sudah diketahui
# sebelum produktivitas aktual diprediksi. Jika tidak tersedia saat prediksi, keluarkan kolom ini.
X = model_df.drop(columns=[target])
y = model_df[target].astype(float)

categorical_features = [c for c in ["quarter", "department", "day", "team"] if c in X.columns]
numeric_features = [c for c in X.columns if c not in categorical_features]
# Pastikan daftar fitur numerik benar-benar numerik; kolom teks lain dimasukkan sebagai kategorikal.
for col in X.columns:
    if col not in categorical_features and not pd.api.types.is_numeric_dtype(X[col]):
        categorical_features.append(col)
numeric_features = [c for c in X.columns if c not in categorical_features]
print("Fitur numerik    :", numeric_features)
print("Fitur kategorikal:", categorical_features)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE)

print(f"\nTraining   : {len(X_train)} ({len(X_train)/len(X):.1%})")
print(f"Validation : {len(X_valid)} ({len(X_valid)/len(X):.1%})")
print(f"Testing    : {len(X_test)} ({len(X_test)/len(X):.1%})")
# Cek multikolinearitas fitur numerik (VIF) pada training set
vif_rows = []
if len(numeric_features) >= 2:
    num_train = SimpleImputer(strategy="median").fit_transform(X_train[numeric_features])
    num_train = StandardScaler().fit_transform(num_train)
    for j, name in enumerate(numeric_features):
        others = np.delete(num_train, j, axis=1)
        r2_j = LinearRegression().fit(others, num_train[:, j]).score(others, num_train[:, j])
        vif = np.inf if np.isclose(1 - r2_j, 0.0) else 1 / (1 - r2_j)
        vif_rows.append({"Fitur": name, "VIF": round(float(vif), 2) if np.isfinite(vif) else np.inf})
else:
    print("VIF tidak dihitung karena fitur numerik kurang dari dua.")
vif_df = pd.DataFrame(vif_rows)
if not vif_df.empty:
    vif_df = vif_df.sort_values("VIF", ascending=False)
    vif_df.to_csv(f"{OUTPUT_DIR}/hasil_vif.csv", index=False)
    print("\nHasil VIF:")
    print(vif_df.to_string(index=False))
def build_pipeline(regressor):
    numeric_tf = Pipeline([("imputer", SimpleImputer(strategy="median")),
                           ("scaler", StandardScaler())])
    categorical_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                               ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", numeric_tf, numeric_features),
                             ("cat", categorical_tf, categorical_features)])
    return Pipeline([("pre", pre), ("reg", regressor)])

def calc_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {"MAE": mean_absolute_error(actual, predicted), "MSE": mse,
            "RMSE": np.sqrt(mse), "R2": r2_score(actual, predicted)}

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

baseline = build_pipeline(DummyRegressor(strategy="mean")).fit(X_train, y_train)
ols = build_pipeline(LinearRegression()).fit(X_train, y_train)

param_grid = {"reg__alpha": [0.01, 0.1, 1, 3, 10, 30, 100, 300]}
grid = GridSearchCV(build_pipeline(Ridge()), param_grid, cv=cv,
                    scoring="neg_root_mean_squared_error")
grid.fit(X_train, y_train)
ridge = grid.best_estimator_
print("Alpha Ridge terbaik (CV training):", grid.best_params_["reg__alpha"])
print("RMSE CV Ridge terbaik            :", round(-grid.best_score_, 4))

ols_cv = -cross_val_score(build_pipeline(LinearRegression()), X_train, y_train, cv=cv,
                          scoring="neg_root_mean_squared_error")
print(f"RMSE CV OLS                      : {ols_cv.mean():.4f} (std {ols_cv.std():.4f})")

models = {"Baseline (rata-rata)": baseline, "Linear Regression (OLS)": ols,
          "Ridge (alpha terbaik)": ridge}
val_table = pd.DataFrame({name: calc_metrics(y_valid, m.predict(X_valid))
                          for name, m in models.items()}).T
print("\n=== PERBANDINGAN MODEL PADA VALIDATION SET ===")
print(val_table.round(4))

best_name = val_table.drop(index="Baseline (rata-rata)")["RMSE"].idxmin()
final_model = models[best_name]
print("\nModel terpilih berdasarkan RMSE validasi:", best_name)
rows = []
for stage, (Xs, ys) in {"Training": (X_train, y_train), "Validation": (X_valid, y_valid),
                        "Testing": (X_test, y_test)}.items():
    rows.append({"Tahap": stage, **calc_metrics(ys, final_model.predict(Xs))})
metrics_df = pd.DataFrame(rows).set_index("Tahap")
print(f"=== METRIK MODEL TERPILIH: {best_name} ===")
print(metrics_df.round(4))
metrics_df.to_csv(f"{OUTPUT_DIR}/hasil_evaluasi_model.csv")

base_test = calc_metrics(y_test, baseline.predict(X_test))
print("\nPembanding baseline rata-rata pada test set:", {k: round(v, 4) for k, v in base_test.items()})

test_pred = final_model.predict(X_test)
within_01 = np.mean(np.abs(y_test - test_pred) <= 0.10) * 100
print(f"\nProporsi prediksi test dengan selisih <= 0,10: {within_01:.1f}%")

pred_out = pd.DataFrame({"Aktual": y_test.to_numpy(), "Prediksi": test_pred})
pred_out["Selisih"] = pred_out["Aktual"] - pred_out["Prediksi"]
pred_out["Selisih_Absolut"] = pred_out["Selisih"].abs()
pred_out.to_csv(f"{OUTPUT_DIR}/hasil_prediksi_data_uji.csv", index=False)
resid = y_test.to_numpy() - test_pred
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

lo, hi = min(y_test.min(), test_pred.min()), max(y_test.max(), test_pred.max())
axes[0].scatter(y_test, test_pred, alpha=0.6)
axes[0].plot([lo, hi], [lo, hi], "--", color="red")
axes[0].set_title("Aktual vs Prediksi (Test Set)")
axes[0].set_xlabel("Produktivitas Aktual")
axes[0].set_ylabel("Produktivitas Prediksi")

axes[1].scatter(test_pred, resid, alpha=0.6)
axes[1].axhline(0, linestyle="--", color="red")
axes[1].set_title("Residual vs Prediksi")
axes[1].set_xlabel("Produktivitas Prediksi")
axes[1].set_ylabel("Residual")

stats.probplot(resid, dist="norm", plot=axes[2])
axes[2].set_title("Q-Q Plot Residual")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_evaluasi_residual.png", dpi=200)
plt.show()

# Galat per departemen pada test set
err = pd.DataFrame({"department": X_test["department"].to_numpy(), "abs_err": np.abs(resid)})
print("MAE test per departemen:")
print(err.groupby("department")["abs_err"].agg(["count", "mean"]).round(4))
feature_names = final_model.named_steps["pre"].get_feature_names_out()
feature_names = [n.replace("num__", "").replace("cat__", "") for n in feature_names]
coef_df = pd.DataFrame({"Fitur": feature_names,
                        "Koefisien": final_model.named_steps["reg"].coef_})
coef_df["Abs"] = coef_df["Koefisien"].abs()
coef_df = coef_df.sort_values("Abs", ascending=False).drop(columns="Abs").reset_index(drop=True)
coef_df.to_csv(f"{OUTPUT_DIR}/koefisien_model.csv", index=False)

# Koefisien tersedia untuk Linear Regression/Ridge, dan model terpilih selalu salah satunya.
top = coef_df.head(12).iloc[::-1]
plt.figure(figsize=(9, 6))
colors = ["tab:green" if v > 0 else "tab:red" for v in top["Koefisien"]]
plt.barh(top["Fitur"], top["Koefisien"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title(f"12 Koefisien Terbesar ({best_name})")
plt.xlabel("Koefisien (hijau: menaikkan, merah: menurunkan produktivitas)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_koefisien_terbesar.png", dpi=200)
plt.show()

coef_df.head(12).round(4)
print("Analisis selesai. Berkas keluaran:")
for name in sorted(os.listdir(OUTPUT_DIR)):
    print("-", name)