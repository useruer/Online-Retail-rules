# ==========================
# 2. 关联规则与用户分群（严格符合图表规范版）
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# 中文显示配置（确保所有中文标签正常显示，避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# 2.1 关联规则挖掘（Apriori算法，前5条强关联规则整齐对齐）
# --------------------------
print("=== 2.1 关联规则挖掘 ===")
# 加载清洗后的数据
try:
    df_clean = pd.read_csv('df_clean.csv', parse_dates=['InvoiceDate'])
except FileNotFoundError:
    # 兼容独立运行，执行基础清洗
    df = pd.read_csv('Online Retail.csv', encoding='latin1')
    df_clean = df.dropna(subset=['CustomerID']).copy()
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], errors='coerce')
    df_clean = df_clean.dropna(subset=['InvoiceDate'])
    df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']
    df_clean.to_csv('df_clean.csv', index=False)

# 构造交易数据集（按订单分组获取商品列表）
df_clean = df_clean[df_clean['Description'].notna()]
transactions = df_clean.groupby('InvoiceNo')['Description'].apply(lambda x: list(x.unique())).values.tolist()

# 交易数据编码
te = TransactionEncoder()
te_matrix = te.fit_transform(transactions)
transaction_df = pd.DataFrame(te_matrix, columns=te.columns_)

# Apriori挖掘频繁项集（最小支持度=0.01）
frequent_itemsets = apriori(transaction_df, min_support=0.01, use_colnames=True)

# 生成关联规则，前5条强关联规则
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
if len(rules) > 0:
    top5_rules = rules.sort_values('confidence', ascending=False).head(5).copy()
    # 规则格式整理（限制长度+统一格式）
    top5_rules['antecedents'] = top5_rules['antecedents'].apply(
        lambda x: ', '.join(list(x))[:30] + '...' if len(', '.join(list(x))) > 30 else ', '.join(list(x))
    )
    top5_rules['consequents'] = top5_rules['consequents'].apply(
        lambda x: ', '.join(list(x))[:30] + '...' if len(', '.join(list(x))) > 30 else ', '.join(list(x))
    )
    # 规范数值格式
    output_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    output_df = top5_rules[output_cols].copy()
    output_df['support'] = output_df['support'].round(6)
    output_df['confidence'] = output_df['confidence'].round(4)
    output_df['lift'] = output_df['lift'].round(2)
    output_df.index = [f'规则{i + 1}' for i in range(len(output_df))]

    print("\n前5条强关联规则（整齐对齐）：")
    print(output_df.to_string(justify='left', col_space=12))
else:
    print("\n未挖掘到满足条件的强关联规则")

# 频繁项集热力图
plt.figure(figsize=(12, 6))
top10_itemsets = frequent_itemsets.nlargest(10, 'support')
itemset_names = [', '.join(list(itemset))[:20] + '...' for itemset in top10_itemsets['itemsets']]
support_values = top10_itemsets['support'].values
im = plt.imshow(support_values.reshape(1, -1), cmap='YlOrRd', aspect='auto')

# 图表规范配置
plt.colorbar(im, label='支持度（无单位）')
plt.xticks(range(len(itemset_names)), itemset_names, rotation=45, ha='right')
plt.yticks([0], ['频繁项集'])
plt.xlabel('商品组合', fontsize=12)
plt.ylabel('项集类型', fontsize=12)
plt.title('图2-1 前10个频繁项集支持度热力图', fontsize=14, pad=20)
plt.legend(['支持度热力分布'], loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig('frequent_itemsets_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n图2-1 前10个频繁项集支持度热力图 已保存")

# --------------------------
# 2.2 基于RFM特征的K-Means用户分群
# --------------------------
print("\n=== 2.2 K-Means用户分群 ===")
# 加载用户特征数据
user_features = pd.read_csv('user_features.csv')
cluster_features = user_features[['Recency', 'Frequency', 'Monetary']].copy()

# 数据标准化
scaler = StandardScaler()
cluster_features_scaled = scaler.fit_transform(cluster_features)
cluster_features_scaled = pd.DataFrame(cluster_features_scaled, columns=cluster_features.columns)

# 执行K-Means聚类
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
user_features['Cluster'] = kmeans.fit_predict(cluster_features_scaled)

# 三维散点图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = ['red', 'blue', 'green']
user_labels = ['高价值活跃用户', '中价值潜力用户', '高流失风险用户']
for cluster_id, color, label in zip(range(3), colors, user_labels):
    cluster_data = user_features[user_features['Cluster'] == cluster_id]
    ax.scatter(
        cluster_data['Recency'],
        cluster_data['Frequency'],
        cluster_data['Monetary'],
        c=color, label=label, alpha=0.6, s=50
    )

# 图表规范配置
ax.set_xlabel('最近购买天数（单位：天）', fontsize=12)
ax.set_ylabel('购买次数（单位：次）', fontsize=12)
ax.set_zlabel('总消费金额（单位：元）', fontsize=12)
ax.set_title('图2-2 RFM三维聚类散点图', fontsize=14, pad=20)
plt.legend(title='用户分群', loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig('rfm_3d_clustering.png', dpi=300, bbox_inches='tight')
plt.close()
print("图2-2 RFM三维聚类散点图 已保存")

# 雷达图
cluster_means = cluster_features_scaled.groupby(user_features['Cluster']).mean()
features = ['最近购买天数', '购买次数', '总消费金额']
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, polar=True)
for cluster_id, color, label in zip(range(3), colors, user_labels):
    values = cluster_means.loc[cluster_id].tolist()
    values += values[:1]  # 闭合数据
    ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
    ax.fill(angles, values, alpha=0.1, color=color)

# 图表规范配置
ax.set_xticks(angles[:-1])
ax.set_xticklabels([f'{feat}（标准化）' for feat in features])
ax.set_ylabel('标准化特征值（无单位）', fontsize=12)
ax.set_title('图2-3 用户分群雷达图', fontsize=14, pad=20)
plt.legend(title='用户分群', loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig('user_cluster_radar.png', dpi=300, bbox_inches='tight')
plt.close()
print("图2-3 用户分群雷达图 已保存")

# 输出用户分群定义表
original_cluster_means = user_features.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
cluster_definition = pd.DataFrame({
    '用户群ID': range(3),
    '用户标签': user_labels,
    '用户数量': user_features['Cluster'].value_counts().sort_index().values,
    '平均最近购买天数（天）': original_cluster_means['Recency'].round(1).values,
    '平均购买次数（次）': original_cluster_means['Frequency'].round(1).values,
    '平均总消费金额（元）': original_cluster_means['Monetary'].round(2).values
})
print("\n用户分群定义表：")
print(cluster_definition.to_string(index=False, justify='center'))

# 保存核心结果文件
user_clusters = pd.merge(user_features, cluster_definition[['用户群ID', '用户标签']],
                         left_on='Cluster', right_on='用户群ID').drop('用户群ID', axis=1)
user_clusters.to_csv('user_clusters.csv', index=False)
frequent_itemsets.to_csv('frequent_itemsets.csv', index=False)

print("\n核心文件生成完成：user_clusters.csv、frequent_itemsets.csv")
print("规范图表文件生成完成：frequent_itemsets_heatmap.png、rfm_3d_clustering.png、user_cluster_radar.png")