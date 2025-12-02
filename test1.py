import pandas as pd
from datetime import timedelta

# 1. 数据清洗（按要求处理缺失值、异常值）
df = pd.read_csv('Online Retail.csv', encoding='latin1')
# 删除CustomerID缺失记录
df_clean = df.dropna(subset=['CustomerID']).copy()
# 剔除负数量、0单价的异常订单
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
# 时间格式转换并删除解析失败记录
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], errors='coerce')
df_clean = df_clean.dropna(subset=['InvoiceDate'])
# 计算订单总金额
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']
# 输出清洗后完整数据集（作业要求交付）
df_clean.to_csv('cleaned_dataset.csv', index=False)

# 2. 构造用户级特征（RFM指标 + 购买时段偏好）
# 2.1 计算RFM核心指标
last_date = df_clean['InvoiceDate'].max()
reference_date = last_date + timedelta(days=1)
rfm_df = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency：最近购买天数
    'InvoiceNo': 'nunique',  # Frequency：购买次数
    'TotalAmount': 'sum'  # Monetary：总消费金额
}).reset_index()
rfm_df.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# 2.2 计算晨间/夜间购买时段偏好
df_clean['Hour'] = df_clean['InvoiceDate'].dt.hour
time_preference = df_clean.groupby('CustomerID').agg({
    'Hour': [
        lambda x: sum((x >= 6) & (x < 12)),  # 晨间6 - 12点订单数
        lambda x: sum((x >= 18) & (x < 24)),  # 夜间18 - 24点订单数
        'count'  # 总订单数
    ]
}).reset_index()
time_preference.columns = ['CustomerID', 'Morning_Orders', 'Night_Orders', 'Total_Orders']
# 计算时段占比（避免0除）
time_preference['Morning_Ratio'] = time_preference['Morning_Orders'] / time_preference['Total_Orders'].replace(0, 1)
time_preference['Night_Ratio'] = time_preference['Night_Orders'] / time_preference['Total_Orders'].replace(0, 1)

# 2.3 合并用户级特征并保存
user_features = pd.merge(rfm_df, time_preference[['CustomerID', 'Morning_Ratio', 'Night_Ratio']], on='CustomerID')
user_features.to_csv('user_features.csv', index=False)

# 3. 构造商品级特征（购买频次、平均订单量）
product_features = df_clean.groupby('StockCode').agg({
    'InvoiceNo': 'nunique',  # 商品被购买频次
    'Quantity': 'mean'  # 商品平均订单量
}).reset_index()
product_features.columns = ['StockCode', 'Purchase_Frequency', 'Avg_Order_Quantity']
product_features.to_csv('product_features.csv', index=False)

# 输出关键验证信息
print(f"清洗后数据集记录数：{len(df_clean)}")
print(f"用户特征数据量：{len(user_features)}条")
print(f"商品特征数据量：{len(product_features)}条")
print("\n清洗后数据集前3条：")
print(df_clean[['InvoiceNo', 'CustomerID', 'Quantity', 'TotalAmount']].head(3))