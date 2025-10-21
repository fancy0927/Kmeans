# -*- coding: utf-8 -*-
"""
供应商画像：PCA降维“操作过程”——逐步代码与对应结果展示
顺序：数据准备→特征选择→标准化→相关性→PCA→解释方差→载荷→选k→得分→可视化→Meta。
Run:
    python supplier_pca_stepwise.py
"""
from pathlib import Path
import pandas as pd, numpy as np

base = Path(".")
data_file = base / "supplier_profile_dataset_stepwise.csv"
if data_file.exists():
    print("检测到已有文件，直接读取：", data_file)
    df = pd.read_csv(data_file, encoding="utf-8-sig")
else:
    print("未找到文件，自动生成示例数据……")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

np.random.seed(7)
base = Path(".")

# 第0步：合成数据
n = 100
categories = np.random.choice(["原材料","零部件","物流","服务外包"], size=n, p=[0.35,0.35,0.15,0.15])
regions = np.random.choice(["华北","华东","华南","西南","东北","海外"], size=n, p=[0.18,0.28,0.22,0.12,0.1,0.1])
latent = np.random.multivariate_normal(mean=[0,0,0],
                                       cov=[[1.0,0.3,0.2],[0.3,1.0,0.25],[0.2,0.25,1.0]], size=n)
CostEff, Reliability, StrategicValue = latent[:,0], latent[:,1], latent[:,2]

price_index        = np.clip(70 + 10*(-CostEff) + np.random.normal(0,3,n), 50, 110)
quality_defect_ppm = np.clip(800 + 300*(-Reliability) + np.random.normal(0,80,n), 50, 2000)
on_time_rate       = np.clip(0.80 + 0.12*Reliability + np.random.normal(0,0.04,n), 0.5, 0.999)
lead_time_days     = np.clip(20 + 5*(-Reliability) + np.random.normal(0,2,n), 3, 45)
response_hours     = np.clip(24 + 6*(-Reliability) + np.random.normal(0,3,n), 2, 72)
flexibility_score  = np.clip(60 + 15*Reliability + 5*StrategicValue + np.random.normal(0,5,n), 20, 100)
financial_stability= np.clip(0.6 + 0.2*StrategicValue + np.random.normal(0,0.08,n), 0.1, 0.99)
esg_score          = np.clip(55 + 20*StrategicValue + np.random.normal(0,6,n), 20, 100)
cooperation_years  = np.clip(3 + 2*StrategicValue + np.random.normal(0,1.5,n), 0, 15)
dispute_count      = np.clip(2 + 1*(-Reliability) + np.random.normal(0,1,n), 0, 10)
innovation_score   = np.clip(50 + 18*StrategicValue + np.random.normal(0,6,n), 10, 100)

df = pd.DataFrame({
    "供应商": [f"S{i+1:03d}" for i in range(n)],
    "品类": categories,
    "区域": regions,
    "价格指数": np.round(price_index,2),
    "质量缺陷PPM": np.round(quality_defect_ppm,0).astype(int),
    "准时交付率": np.round(on_time_rate,4),
    "平均交期_天": np.round(lead_time_days,1),
    "响应时长_小时": np.round(response_hours,1),
    "柔性能力评分": np.round(flexibility_score,1),
    "财务稳健度": np.round(financial_stability,4),
    "ESG评分": np.round(esg_score,1),
    "合作年限": np.round(cooperation_years,1),
    "纠纷次数": np.round(dispute_count,0).astype(int),
    "创新能力评分": np.round(innovation_score,1),
})
df.to_csv(base/"supplier_profile_dataset_stepwise.csv", index=False, encoding="utf-8-sig")
print("【第0步】已保存：supplier_profile_dataset_stepwise.csv")

# 第1步：特征选择
features = ["价格指数","质量缺陷PPM","准时交付率","平均交期_天","响应时长_小时",
            "柔性能力评分","财务稳健度","ESG评分","合作年限","纠纷次数","创新能力评分"]
X = df[features].copy()
print("【第1步】特征列个数：", len(features))

# 第2步：标准化
scaler = StandardScaler()
X_std = scaler.fit_transform(X.values)
std_df = pd.DataFrame(X_std, columns=features)
check_std = pd.DataFrame({"均值": std_df.mean(), "标准差": std_df.std(ddof=0)})
check_std.to_csv(base/"supplier_std_check.csv", encoding="utf-8-sig")
print("【第2步】已保存：supplier_std_check.csv（均值≈0、标准差≈1）")

# 第3步：相关性
corr = X.corr().round(3)
corr.to_csv(base/"supplier_corr_matrix.csv", encoding="utf-8-sig")
print("【第3步】已保存：supplier_corr_matrix.csv")

# 第4步：PCA拟合（全部PC）
pca = PCA(n_components=len(features), svd_solver="full")
scores_all = pca.fit_transform(X_std)
explained_df = pd.DataFrame({
    "主成分": [f"PC{i+1}" for i in range(len(features))],
    "单个解释方差比": np.round(pca.explained_variance_ratio_, 4),
    "累计解释方差比": np.round(np.cumsum(pca.explained_variance_ratio_), 4)
})
explained_df.to_csv(base/"supplier_pca_explained_stepwise.csv", index=False, encoding="utf-8-sig")
print("【第4步】已保存：supplier_pca_explained_stepwise.csv")

# 第5步：载荷矩阵
loadings = pd.DataFrame(pca.components_.T, index=features, columns=[f"PC{i+1}" for i in range(len(features))])
loadings.to_csv(base/"supplier_pca_loadings_stepwise.csv", encoding="utf-8-sig")
print("【第5步】已保存：supplier_pca_loadings_stepwise.csv")

# 第6步：选k并导出得分
k = int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.85) + 1)
scores_k = pd.DataFrame(scores_all[:, :k], columns=[f"PC{i+1}_得分" for i in range(k)])
scores_out = pd.concat([df[["供应商","品类","区域"]], scores_k], axis=1)
scores_out.to_csv(base/"supplier_pca_scores_stepwise.csv", index=False, encoding="utf-8-sig")
print(f"【第6步】k={k}（累计解释方差≥85%），已保存：supplier_pca_scores_stepwise.csv")

# 第7步：可视化
from matplotlib.pyplot import figure, ylabel, arrow, savefig, annotate, title, tight_layout, plot, grid, scatter, \
    xlabel, text, close
figure()
plot(range(1, len(features) + 1), pca.explained_variance_ratio_, marker="o")
title("Scree Plot（各主成分解释方差比）")
xlabel("主成分序号")
ylabel("解释方差比")
grid(True)
tight_layout()
savefig(base / "supplier_pca_scree_plot_stepwise.png", dpi=160)
close()

figure()
scatter(scores_all[:,0], scores_all[:,1])
for i, name in enumerate(df["供应商"]):
    if i % 5 == 0:
        annotate(name, (scores_all[i,0], scores_all[i,1]), fontsize=8)

scale = 2.2
for j, feat in enumerate(features):
    arrow(0, 0, loadings.iloc[j,0] * scale, loadings.iloc[j,1] * scale, head_width=0.08, length_includes_head=True)
    text(loadings.iloc[j,0] * scale * 1.08, loadings.iloc[j,1] * scale * 1.08, feat, fontsize=8)

title("Biplot: PC1 vs PC2")
xlabel("PC1 得分")
ylabel("PC2 得分")
grid(True)
tight_layout()
savefig(base / "supplier_pca_biplot_stepwise.png", dpi=160)
close()
print("【第7步】已保存：scree & biplot")

# 第8步：Meta
meta = {
    "样本量": n,
    "特征数": len(features),
    "选取PC个数_k(≥85%)": k,
    "累计解释方差_k": round(float(np.cumsum(pca.explained_variance_ratio_)[k-1]), 4)
}
pd.DataFrame(meta, index=[0]).to_csv(base/"supplier_pca_meta_stepwise.csv", index=False, encoding="utf-8-sig")
print("【第8步】已保存：supplier_pca_meta_stepwise.csv")

print("全部完成。请查看当前目录下输出文件。")
