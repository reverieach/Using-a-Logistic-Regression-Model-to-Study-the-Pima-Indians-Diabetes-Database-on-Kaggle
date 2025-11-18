# 北京邮电大学神经网络与深度学习实验一

## 实验总览

本实验是一个**机器学习回归问题**，目标是使用逻辑回归模型分析糖尿病数据集中特征与诊断结果的关系，并进行准确率评估。

### 数据集信息
- **数据集名称**：Pima Indians Diabetes Database
- **数据集规模**：768条样本，9个特征
- **任务类型**：二分类问题（患糖尿病 vs 未患糖尿病）
- **下载地址**：https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### 主要评分点
| 任务 | 分值 |
|------|------|
| 特征工程方法说明 | 2分 |
| 逻辑回归模型+散点图 | 2分 |
| 数据集划分与准确率测试 | 2分 |
| 特征分析与关联性排序 | 4分 |
| **总分** | **10分** |

---

## 实验步骤详解

### 第一步：环境准备与数据下载

#### 1.1 安装必要库
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

#### 1.2 下载数据集
从 [Kaggle官网](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) 下载 `diabetes.csv` 文件，保存到工作目录。

#### 1.3 数据集字段说明

| 序号 | 字段名 | 数据类型 | 字段描述 |
|------|--------|--------|---------|
| 1 | Pregnancies | Integer | 怀孕次数 |
| 2 | Glucose | Integer | 口服葡萄糖耐量试验中血浆葡萄糖浓度（2小时） |
| 3 | BloodPressure | Integer | 舒张压(mm Hg) |
| 4 | SkinThickness | Integer | 三头肌皮褶厚度(mm) |
| 5 | Insulin | Integer | 2小时血清胰岛素(uU/mL) |
| 6 | BMI | Integer | 体重指数(体重kg/(身高m)²) |
| 7 | DiabetesPedigreeFunction | Float | 糖尿病谱系功能 |
| 8 | Age | Integer | 年龄(岁) |
| 9 | Outcome | Integer | 目标变量(0=无糖尿病, 1=有糖尿病) |

---

### 第二步：数据加载与基本探索

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv('diabetes.csv')

# 查看基本信息
print(f"数据集形状: {df.shape}")
print(df.head())
print(df.describe())
print(df.dtypes)
print(df.isnull().sum())  # 检查缺失值
```

**关键观察**：
- 数据集包含768条样本和8个特征
- 部分特征（如Glucose、BloodPressure等）存在值为0的异常记录
- 原始数据没有缺失值，但医学上这些特征不应为0

---

### 第三步：特征工程 - 数据清洗与处理

#### 3.1 异常值处理

根据医学常识，以下特征在医学上不可能为0：
- Glucose（血糖）
- BloodPressure（血压）
- SkinThickness（皮褶厚度）
- Insulin（胰岛素）
- BMI（体重指数）

```python
df_clean = df.copy()

# 将医学上不可能为0的值设为NaN
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in zero_columns:
    zero_count = (df_clean[col] == 0).sum()
    print(f"{col}: {zero_count}个0值被识别为缺失值")
    df_clean[col] = df_clean[col].replace(0, np.nan)
```

#### 3.2 缺失值处理

使用**均值填充法**（mean imputation），这是最常见的方法：

```python
for col in zero_columns:
    nan_count = df_clean[col].isnull().sum()
    if nan_count > 0:
        mean_val = df_clean[col].mean()
        df_clean[col].fillna(mean_val, inplace=True)
        print(f"{col}: {nan_count}个缺失值已用均值{mean_val:.2f}填充")
```

**为什么选择均值填充？**
- 保留数据的统计特性
- 不改变特征的均值
- 简单易用，适合数据量不大的情况
- 对于连续数值特征效果较好

#### 3.3 其他特征工程方法

根据PPT内容，可选的方法包括：
- **Binarizer**：对定量特征进行二值化
- **StandardScaler**：标准化（后续在建模前使用）
- **MinMaxScaler**：归一化到[0,1]区间

---

### 第四步：特征分析与相关性挖掘

#### 4.1 计算特征相关系数

```python
# 计算各特征与Outcome的相关系数
correlations = df_clean.corr()['Outcome'].sort_values(ascending=False)
print(correlations)

# 可视化
plt.figure(figsize=(8, 5))
correlations.plot(kind='barh', color='steelblue')
plt.title('各特征与糖尿病诊断的相关系数')
plt.xlabel('相关系数')
plt.ylabel('特征')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
```

**相关性解读**：
- 相关系数接近1：正相关（特征值越大，患糖尿病概率越高）
- 相关系数接近0：无相关性
- 相关系数接近-1：负相关（特征值越大，患糖尿病概率越低）

---

### 第五步：特征标准化

在训练模型前，需要对特征进行标准化：

```python
from sklearn.preprocessing import StandardScaler

# 分离特征和目标变量
X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']

# 使用StandardScaler进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 标准化后特征均值为0，方差为1
print(X_scaled.describe())
```

**为什么要标准化？**
- 消除特征间的量纲差异
- 加速梯度下降收敛
- 提高逻辑回归等模型的性能

---

### 第六步：数据集划分

#### 6.1 根据学号要求划分

- **奇数学号**：train_size = 0.7, 0.75, 0.8
- **偶数学号**：train_size = 0.75, 0.8, 0.85

```python
from sklearn.model_selection import train_test_split

# 这里以偶数学号为例
train_sizes = [0.75, 0.8, 0.85]
results = {}

for train_size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        train_size=train_size, 
        random_state=42  # 固定随机种子保证结果可重复
    )
    print(f"train_size={train_size}: 训练集{X_train.shape[0]}, 测试集{X_test.shape[0]}")
    results[train_size] = (X_train, X_test, y_train, y_test)
```

**参数说明**：
- `train_size`：训练集比例
- `random_state`：随机种子，固定可以保证结果可复现

---

### 第七步：逻辑回归模型训练

#### 7.1 模型训练与评估

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 对每个train_size训练模型
for train_size, (X_train, X_test, y_train, y_test) in results.items():
    # 创建并训练模型
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # 在训练集和测试集上评估
    train_acc = lr_model.score(X_train, y_train)
    test_acc = lr_model.score(X_test, y_test)
    
    print(f"\ntrain_size={train_size}:")
    print(f"  训练集准确率: {train_acc:.4f}")
    print(f"  测试集准确率: {test_acc:.4f}")
    print(f"  模型系数: {lr_model.coef_[0]}")
```

#### 7.2 选择最优模型

选择测试集准确率最高的train_size作为最终模型，通常为0.8：

```python
best_train_size = 0.8
X_train, X_test, y_train, y_test = results[best_train_size]

# 训练最终模型
final_model = LogisticRegression(random_state=42, max_iter=1000)
final_model.fit(X_train, y_train)

# 详细评估
y_test_pred = final_model.predict(X_test)
y_test_proba = final_model.predict_proba(X_test)  # 预测概率

print("分类报告：")
print(classification_report(y_test, y_test_pred))
```

---

### 第八步：特征重要性分析

#### 8.1 提取模型系数

```python
# 获取模型系数（特征重要性）
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': final_model.coef_[0],
    'Abs_Coefficient': np.abs(final_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print(feature_importance)
```

**系数解读**：
- 正系数：该特征值增大，患糖尿病的概率增大
- 负系数：该特征值增大，患糖尿病的概率降低
- 绝对值越大，特征对预测的影响越大

#### 8.2 可视化特征重要性

```python
# 绘制特征重要性条形图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：相关系数
ax1 = axes[0]
correlations.plot(kind='barh', ax=ax1, color='steelblue')
ax1.set_title('各特征与糖尿病诊断的相关系数')
ax1.set_xlabel('相关系数')
ax1.grid(axis='x', alpha=0.3)

# 右图：模型系数
ax2 = axes[1]
feature_importance_sorted = feature_importance.sort_values('Coefficient')
colors = ['red' if x < 0 else 'green' for x in feature_importance_sorted['Coefficient']]
ax2.barh(feature_importance_sorted['Feature'], 
         feature_importance_sorted['Coefficient'], 
         color=colors)
ax2.set_title('逻辑回归模型系数')
ax2.set_xlabel('系数值')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### 第九步：可视化与总结

#### 9.1 绘制逻辑回归散点图

```python
# 选择两个最重要的特征绘制散点图
# 例如Glucose和Age
fig, ax = plt.subplots(figsize=(10, 6))

# 将数据分为两类绘制
no_diabetes = df_clean[df_clean['Outcome'] == 0]
diabetes = df_clean[df_clean['Outcome'] == 1]

ax.scatter(no_diabetes['Glucose'], no_diabetes['Age'], 
          c='blue', label='无糖尿病', alpha=0.6, s=30)
ax.scatter(diabetes['Glucose'], diabetes['Age'], 
          c='red', label='有糖尿病', alpha=0.6, s=30)

ax.set_xlabel('Glucose')
ax.set_ylabel('Age')
ax.set_title('Glucose vs Age（按糖尿病诊断结果着色）')
ax.legend()
ax.grid(alpha=0.3)
plt.show()
```


```

---

## 常见问题解答

### Q1：为什么要进行特征标准化？
**A**：逻辑回归使用梯度下降法优化，不同特征的量纲差异会导致某些特征的梯度过大或过小，影响收敛速度和模型性能。标准化将所有特征缩放到相同的尺度（均值0，方差1）。

### Q2：如何处理类别不均衡的数据？
**A**：本数据集中两类样本比例接近，但在实际应用中可以使用：
- 重新采样（过采样少数类或欠采样多数类）
- 调整类权重
- 使用F1-score等指标而非准确率

### Q3：模型准确率不高怎么办？
**A**：可以尝试：
- 尝试其他模型（如SVM、决策树、随机森林）
- 调整正则化参数（C参数）
- 进行特征工程或特征选择
- 增加数据量

### Q4：为什么要用train_test_split而不是交叉验证？
**A**：实验要求使用train_test_split。实际应用中，交叉验证更加鲁棒，能更好地估计模型泛化性能。

### Q5：均值填充会引入偏差吗？
**A**：会有一定影响，但对于缺失数据量较少（本数据集中缺失值较多）的情况是可接受的。其他选择包括：
- 使用中位数填充（对异常值鲁棒性更强）
- 使用KNN填充
- 删除缺失值（可能丢失信息）

---

## 实验输出检查清单

- [ ] 数据基本信息输出（形状、前5行、统计信息）
- [ ] 异常值和缺失值处理说明
- [ ] 各特征与Outcome的相关系数
- [ ] 三种train_size下的模型准确率对比
- [ ] 最终模型的详细分类报告
- [ ] 特征重要性排序
- [ ] 相关系数条形图
- [ ] 模型系数条形图
- [ ] 逻辑回归散点图
- [ ] 混淆矩阵热力图

---

## 完整代码文件说明

本指导配套有完整的代码文件 `diabetes_experiment_complete.py`，包含：
1. 数据加载与探索
2. 特征工程（异常值处理、缺失值填充）
3. 特征分析与标准化
4. 数据集划分
5. 模型训练与评估
6. 特征重要性分析
7. 完整可视化

**使用方法**：
```bash
python diabetes_experiment_complete.py
```

---

## 参考资源

- Kaggle数据集：https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- Scikit-learn文档：https://scikit-learn.org/
- Pandas文档：https://pandas.pydata.org/
- 逻辑回归理论：https://en.wikipedia.org/wiki/Logistic_regression

---

**实验作者提示**：按照步骤逐步完成实验，每一步都有具体的代码示例，可以直接复制使用。完成后将代码和输出结果整理成实验报告提交。
