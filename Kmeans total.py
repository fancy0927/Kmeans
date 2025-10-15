import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 步骤1: 数据清洗函数
def clean_financial_data(df):
    """
    清洗财务数据
    解释: 这个函数负责处理原始数据中的各种问题，确保数据质量适合聚类分析
    """
    # 创建数据副本 - 避免修改原始数据
    df_clean = df.copy()

    # 清洗年份列 - 移除"年"字并转换为整数
    # 解释: 原始数据中有些年份带有"年"字(如"2020年")，需要统一格式
    df_clean['年份'] = df_clean['年份'].astype(str).str.replace('年', '').astype(int)

    # 处理数值列中的空值和异常值
    numeric_cols = ['资产总额', '负债总额', '营业收入', '净利润']

    # 使用中位数填充缺失值
    # 解释: 中位数对异常值不敏感，比均值更适合处理财务数据
    imputer = SimpleImputer(strategy='median')
    df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])

    # 移除可能的重复行（基于企业ID和年份）
    # 解释: 确保每条记录的唯一性，避免重复数据影响聚类结果
    df_clean = df_clean.drop_duplicates(subset=['企业ID', '年份'])

    # 计算一些有用的财务比率
    # 解释: 财务比率能更好地反映企业财务状况，比绝对值更有意义
    df_clean['资产负债率'] = df_clean['负债总额'] / df_clean['资产总额']
    df_clean['净利润率'] = df_clean['净利润'] / df_clean['营业收入']
    df_clean['资产周转率'] = df_clean['营业收入'] / df_clean['资产总额']

    # 处理无穷大值和异常比率
    # 解释: 当分母为0时会产生无穷大值，需要处理
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    # 对于比率列，用0填充NaN（例如，当营业收入为0时，净利润率为无穷大）
    ratio_cols = ['资产负债率', '净利润率', '资产周转率']
    df_clean[ratio_cols] = df_clean[ratio_cols].fillna(0)

    # 限制比率在合理范围内（例如，资产负债率在0-2之间，净利润率在-1到1之间）
    # 解释: 防止极端值对聚类产生过大影响
    df_clean['资产负债率'] = df_clean['资产负债率'].clip(0, 2)
    df_clean['净利润率'] = df_clean['净利润率'].clip(-1, 1)
    df_clean['资产周转率'] = df_clean['资产周转率'].clip(0, 5)

    return df_clean


# 步骤2: 确定最优聚类数量
def find_optimal_clusters(data, max_k=10):
    """
    使用肘部法则找到最优聚类数量
    解释: 肘部法则通过计算不同K值对应的簇内平方和，找到拐点作为最佳K值
    """
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)  # inertia_是簇内平方和

    return inertias


# 步骤3: 执行KMeans聚类
def perform_kmeans_clustering(df, n_clusters=3):
    """
    执行KMeans聚类分析
    解释: 这个函数负责标准化数据并执行KMeans聚类算法
    """
    # 选择用于聚类的特征
    features = ['资产总额', '负债总额', '营业收入', '净利润',
                '资产负债率', '净利润率', '资产周转率']

    # 标准化数据
    # 解释: KMeans对特征尺度敏感，必须标准化确保每个特征有相同的重要性
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    # 执行KMeans聚类
    # 解释:
    # - n_clusters: 聚类数量
    # - random_state: 随机种子，确保结果可重现
    # - n_init: 初始化次数，选择最佳初始中心
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    # 将聚类结果添加到数据框
    df['Cluster'] = clusters

    return df, kmeans, scaler, scaled_data


# 步骤4: 可视化聚类结果
def visualize_clusters(df, scaled_data, kmeans):
    """
    可视化聚类结果
    解释: 使用多种图表展示聚类结果，帮助理解数据分布和聚类特征
    """
    # 使用PCA进行降维以便可视化
    # 解释: 原始数据有7个维度，无法直接可视化，PCA将其降为2维
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]

    # 创建可视化图表 - 2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 聚类散点图
    # 解释: 展示企业在降维后的空间中的分布，不同颜色代表不同聚类
    scatter = axes[0, 0].scatter(df['PCA1'], df['PCA2'], c=df['Cluster'],
                                 cmap='viridis', alpha=0.7)
    axes[0, 0].set_title('KMeans聚类结果 (PCA降维)')
    axes[0, 0].set_xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[0, 0].set_ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.colorbar(scatter, ax=axes[0, 0])

    # 2. 聚类规模条形图
    # 解释: 显示每个聚类包含的企业数量，了解聚类大小分布
    cluster_counts = df['Cluster'].value_counts().sort_index()
    axes[0, 1].bar(cluster_counts.index, cluster_counts.values, color='skyblue')
    axes[0, 1].set_title('各聚类企业数量')
    axes[0, 1].set_xlabel('聚类')
    axes[0, 1].set_ylabel('企业数量')
    for i, count in enumerate(cluster_counts.values):
        axes[0, 1].text(i, count, str(count), ha='center', va='bottom')

    # 3. 聚类特征均值热力图
    # 解释: 显示每个聚类在主要财务指标上的平均值，使用对数变换使差异更明显
    cluster_means = df.groupby('Cluster')[['资产总额', '负债总额', '营业收入', '净利润']].mean()
    # 对数变换以便更好地显示
    cluster_means_log = np.log1p(cluster_means)
    sns.heatmap(cluster_means_log, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1, 0])
    axes[1, 0].set_title('各聚类财务指标均值（对数变换后）')

    # 4. 资产负债率分布
    # 解释: 比较不同聚类的资产负债率分布，了解各聚类的财务杠杆状况
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]['资产负债率']
        axes[1, 1].hist(cluster_data, alpha=0.6, label=f'Cluster {cluster}', bins=20)
    axes[1, 1].set_title('各聚类资产负债率分布')
    axes[1, 1].set_xlabel('资产负债率')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    return fig


# 步骤5: 分析聚类特征
def analyze_cluster_characteristics(df):
    """
    分析每个聚类的特征
    解释: 提供每个聚类的详细统计信息，帮助理解各类企业的财务特征
    """
    print("=" * 50)
    print("各聚类特征分析")
    print("=" * 50)

    features = ['资产总额', '负债总额', '营业收入', '净利润',
                '资产负债率', '净利润率', '资产周转率']

    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        print(f"\n聚类 {cluster} (共{len(cluster_data)}家企业):")
        print(f"平均资产总额: {cluster_data['资产总额'].mean():,.2f}")
        print(f"平均负债总额: {cluster_data['负债总额'].mean():,.2f}")
        print(f"平均营业收入: {cluster_data['营业收入'].mean():,.2f}")
        print(f"平均净利润: {cluster_data['净利润'].mean():,.2f}")
        print(f"平均资产负债率: {cluster_data['资产负债率'].mean():.2%}")
        print(f"平均净利润率: {cluster_data['净利润率'].mean():.2%}")
        print(f"平均资产周转率: {cluster_data['资产周转率'].mean():.2f}")


# 主函数 - 整合所有步骤
def main():
    """
    主函数
    解释: 这是程序的入口点，按顺序执行所有步骤
    """
    # 步骤1: 读取数据
    file_path = r'D:\1\财务数据清洗案例数据集.xlsx'
    print("正在读取数据...")
    try:
        df = pd.read_excel(file_path)
        print(f"成功读取数据，共{len(df)}行记录")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

    # 步骤2: 数据清洗
    print("正在进行数据清洗...")
    df_clean = clean_financial_data(df)
    print(f"数据清洗完成，剩余{len(df_clean)}行记录")

    # 步骤3: 确定最优聚类数量
    print("正在确定最优聚类数量...")
    features = ['资产总额', '负债总额', '营业收入', '净利润',
                '资产负债率', '净利润率', '资产周转率']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean[features])

    inertias = find_optimal_clusters(scaled_data)

    # 绘制肘部法则图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(inertias) + 1), inertias, 'bo-')
    plt.xlabel('聚类数量')
    plt.ylabel('簇内平方和')
    plt.title('肘部法则 - 确定最优聚类数量')
    plt.grid(True)
    plt.show()

    # 根据肘部法则选择聚类数量（这里选择4）
    # 解释: 肘部法则图中，拐点通常是最佳聚类数，这里我们选择4
    n_clusters = 3
    print(f"选择聚类数量: {n_clusters}")

    # 步骤4: 执行聚类分析
    print(f"正在进行KMeans聚类 (k={n_clusters})...")
    df_clustered, kmeans, scaler, scaled_data = perform_kmeans_clustering(df_clean, n_clusters)

    # 步骤5: 可视化结果
    print("正在生成可视化图表...")
    visualize_clusters(df_clustered, scaled_data, kmeans)

    # 步骤6: 分析聚类特征
    analyze_cluster_characteristics(df_clustered)

    # 步骤7: 保存结果
    output_path = r'D:\1\聚类分析结果.xlsx'
    df_clustered.to_excel(output_path, index=False)
    print(f"\n聚类分析结果已保存到 '{output_path}'")

    return df_clustered


# 运行主程序
if __name__ == "__main__":
    # 解释: 这是Python程序的入口点，当直接运行此脚本时执行main函数
    result_df = main()