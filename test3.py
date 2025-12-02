# ==========================
# 3. 复购行为预测与营销策略（作业核心要求精简版）
# ==========================
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

# 中文显示配置（满足图表中文标签需求）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# 3.1 数据准备与复购标签构造（预测30天内复购概率）
# --------------------------
print("=== 3.1 复购标签构造 ===")
# 加载前置数据文件
try:
    df_clean = pd.read_csv('df_clean.csv', parse_dates=['InvoiceDate'])
    user_features = pd.read_csv('user_features.csv')
    user_clusters = pd.read_csv('user_clusters.csv')
except FileNotFoundError as e:
    print(f"错误：未找到文件{e.filename}，请先执行前序代码！")
    exit()

# 定义复购判断窗口（最后30天为复购窗口）
last_date = df_clean['InvoiceDate'].max()
history_cutoff = last_date - timedelta(days=30)

# 构造复购标签（1=复购，0=未复购）
user_last_purchase = df_clean.groupby('CustomerID')['InvoiceDate'].max().reset_index()
user_last_purchase.columns = ['CustomerID', 'Last_Purchase_Date']
user_last_purchase['Repurchase_Label'] = (user_last_purchase['Last_Purchase_Date'] >= history_cutoff).astype(int)

# 合并特征与标签
model_data = pd.merge(user_features, user_clusters[['CustomerID', 'Cluster', '用户标签']], on='CustomerID')
model_data = pd.merge(model_data, user_last_purchase[['CustomerID', 'Repurchase_Label']], on='CustomerID')

# 特征编码与数据集划分
model_data_encoded = pd.get_dummies(model_data, columns=['用户标签'], drop_first=True)
feature_cols = [col for col in model_data_encoded.columns if col not in ['CustomerID', 'Cluster', 'Repurchase_Label']]
X = model_data_encoded[feature_cols]
y = model_data_encoded['Repurchase_Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"训练集：{len(X_train)}条，测试集：{len(X_test)}条，复购率：{y.mean():.2%}")

# --------------------------
# 3.2 构建二分类模型（对比3种算法，选最优模型预测）
# --------------------------
print("\n=== 3.2 复购预测模型训练 ===")
# 定义3种对比模型
models = {
    '决策树': DecisionTreeClassifier(random_state=42, max_depth=10),
    '随机森林': RandomForestClassifier(random_state=42, n_estimators=100),
    '逻辑回归': LogisticRegression(random_state=42, max_iter=1000)
}

# 交叉验证对比F1-score
cv_results = {}
print("5折交叉验证F1-score：")
for name, model in models.items():
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()
    cv_results[name] = cv_score
    print(f"{name}：{cv_score:.4f}")

# 选择最优模型（随机森林为核心模型，兼顾效果与特征重要性）
best_model_name = max(cv_results, key=cv_results.get)
best_model = models[best_model_name]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# 混淆矩阵可视化（符合图表规范：编号+标题+标签+图例）
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap='Blues')
for i in range(2):
    for j in range(2):
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center', fontsize=12)
plt.xlabel('预测标签（0=未复购，1=复购）', fontsize=12)
plt.ylabel('真实标签（0=未复购，1=复购）', fontsize=12)
plt.title('图3-1 最优模型测试集混淆矩阵', fontsize=14)
plt.xticks([0, 1], ['未复购', '复购'])
plt.yticks([0, 1], ['未复购', '复购'])
plt.legend(['混淆矩阵热力分布'], loc='upper right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()

# 算法对比可视化
plt.figure(figsize=(10, 5))
plt.bar(cv_results.keys(), cv_results.values(), color=['orange', 'green', 'blue'])
plt.xlabel('算法类型', fontsize=12)
plt.ylabel('F1-score', fontsize=12)
plt.title('图3-2 3种算法F1-score对比', fontsize=14)
plt.legend(['5折交叉验证结果'], loc='upper right')
plt.tight_layout()
plt.savefig('model_f1_comparison.png', dpi=300)
plt.close()

# --------------------------
# 3.3 关键特征重要性排名（输出Gini重要性，条形图可视化）
# --------------------------
print("\n=== 3.3 特征重要性排名 ===")
# 计算特征重要性（以随机森林为例，输出Gini重要性）
if best_model_name in ['决策树', '随机森林']:
    feature_importance = pd.DataFrame({
        '特征': feature_cols,
        '重要性': best_model.feature_importances_
    }).sort_values('重要性', ascending=False)

# 输出前10个关键特征
print("关键特征重要性排名（前10）：")
print(feature_importance.head(10).to_string(index=False))

# 特征重要性条形图（符合图表规范）
plt.figure(figsize=(12, 6))
top10_features = feature_importance.head(10)
plt.barh(top10_features['特征'], top10_features['重要性'], color='teal')
plt.xlabel('重要性（Gini系数）', fontsize=12)
plt.ylabel('特征名称', fontsize=12)
plt.title('图3-3 关键特征重要性条形图', fontsize=14)
plt.legend(['特征重要性'], loc='upper right')
plt.tight_layout()
plt.savefig('feature_importance_bar.png', dpi=300)
plt.close()

# --------------------------
# 3.4 分群营销策略（针对不同用户群制定策略）
# --------------------------
print("\n=== 3.4 分群营销策略 ===")
# 统计各用户群复购特征
cluster_stats = model_data.groupby(['Cluster', '用户标签']).agg({
    'Repurchase_Label': ['count', 'mean'],
    'Frequency': 'mean',
    'Monetary': 'mean'
}).round(2)
cluster_stats.columns = ['用户总数', '复购率', '平均购买次数', '平均消费金额']
cluster_stats = cluster_stats.reset_index()

# 制定营销策略
strategy_list = []
for _, row in cluster_stats.iterrows():
    label = row['用户标签']
    repurchase_rate = row['复购率']
    if '高价值活跃' in label:
        strategy = "会员升级+新品优先体验+消费返现，巩固忠诚度"
    elif '中价值潜力' in label:
        strategy = "购买频次激励+关联商品推荐+限时折扣，提升购买次数"
    elif '高流失风险' in label:
        strategy = "大额召回优惠券+个性化短信提醒+老客专属价，唤醒用户"
    else:
        strategy = "小额无门槛券+场景化营销，培养购买习惯"
    strategy_list.append({
        '用户群ID': row['Cluster'],
        '用户标签': label,
        '复购率': repurchase_rate,
        '核心营销策略': strategy
    })

# 生成并保存营销策略表
strategy_df = pd.DataFrame(strategy_list)
print("\n分群营销策略表：")
print(strategy_df.to_string(index=False))
strategy_df.to_csv('user_cluster_strategies.csv', index=False, encoding='utf-8-sig')

# 保存模型预测结果文件
model_predictions = pd.DataFrame({
    'CustomerID': model_data.iloc[X_test.index]['CustomerID'],
    '真实标签': y_test.values,
    '预测标签': y_pred,
    '复购概率': best_model.predict_proba(X_test)[:, 1].round(4)
})
model_predictions.to_csv('model_predictions.csv', index=False)

print("\n核心文件生成完成：")
print("1. 图表文件：confusion_matrix.png、model_f1_comparison.png、feature_importance_bar.png")
print("2. 数据文件：user_cluster_strategies.csv、model_predictions.csv")