import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import spec_utils as su
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# 1. 定义一个统一的标准波数网格（假设你的光谱大致在 400~4000 cm^-1 之间）
# 这里我们统一取 1000 个点
STANDARD_X = np.linspace(400, 4000, 1000)

# 1. 构建特征矩阵
def build_feature_matrix(data_dir="data"):
    X = []
    y = []
    
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
            
        for file in os.listdir(class_dir):
            if file.endswith('.csv') or file.endswith('.txt'):
                filepath = os.path.join(class_dir, file)
                
                # 读取原始光谱
                x, intensity = su.load_spectrum(filepath)
                px, py = su.preprocess_pipeline(x, intensity)
                
                # 【核心修改点】：数据对齐（插值到统一网格）
                # 使用 np.interp 将原始数据映射到 STANDARD_X 上
                # 注意：np.interp 要求原始 px 是递增的。如果你的光谱是递减的，需要先反转
                if px[0] > px[-1]:
                    px = px[::-1]
                    py = py[::-1]
                
                # 插值对齐
                py_aligned = np.interp(STANDARD_X, px, py)
                
                X.append(py_aligned) 
                y.append(label)
                
    return np.array(X), np.array(y)

# 2. 训练模型并评估
if __name__ == "__main__":
    print("正在加载并预处理数据...")
    X, y = build_feature_matrix()
    
    if len(X) == 0:
        print("错误：未找到数据，请检查 data 文件夹和文件格式！")
        raise SystemExit(1)
        
    print(f"数据加载完毕，样本数: {len(X)}, 类别数: {len(set(y))}")
    
    # 拆分训练集和测试集（按类别等比例拆分）
    # 注意：样本数太少时 stratify 可能会报错，如果报错把 stratify=y 去掉
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # 初始化 KNN 模型，K=5
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Xtr, ytr)
    
    # 预测并打印准确率
    score = knn.score(Xte, yte)
    print(f"KNN 测试集准确率: {score:.4f}")

    # 1. 输出混淆矩阵（看谁被分错了，虽然现在可能全是0）
    y_pred = knn.predict(Xte)
    print("\n===== 混淆矩阵 (KNN) =====")
    print(confusion_matrix(yte, y_pred))
    print("\n===== 分类报告 (KNN) =====")
    print(classification_report(yte, y_pred))

    # 2. 做交叉验证（抵抗单次拆分测试集的偶然性）
    # 注意：样本只有18个，cv=3 比较合适。如果报错，把 cv=3 改成 cv=2 或 cv=5 试试
    print("\n===== 交叉验证 (KNN) =====")
    cv_scores = cross_val_score(knn, X, y, cv=3)
    print(f"3折交叉验证得分: {cv_scores}")
    print(f"平均交叉验证得分: {cv_scores.mean():.4f}")

    # 3. 引入 SVM 对比
    print("\n===== SVM 对比实验 =====")
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(Xtr, ytr)
    print(f"SVM 测试集准确率: {svm.score(Xte, yte):.4f}")


# 找到第一个 PET 和 PC 样本进行对比
plt.figure(figsize=(10, 5))
for i in range(len(y)):
    if y[i] == 'PET':
        plt.plot(X[i], label='PET', alpha=0.7)
        break
for i in range(len(y)):
    if y[i] == 'PC':
        plt.plot(X[i], label='PC', alpha=0.7)
        break

plt.title('PET vs PC Spectrum Comparison')
plt.xlabel('Wavenumber (cm^-1)')
plt.ylabel('Normalized Intensity')
plt.legend()
plt.show() # 运行后会弹出一张图，你可以截图保存，放到简历里！


# 构建一个：标准化 -> PCA降维 -> KNN 的流水线
pca_knn_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.95), # 保留95%的方差
    KNeighborsClassifier(n_neighbors=3)
)

# 再次跑交叉验证
pca_cv_scores = cross_val_score(pca_knn_pipeline, X, y, cv=3)
print(f"\nPCA+KNN 交叉验证得分: {pca_cv_scores}")
print(f"PCA+KNN 平均交叉验证得分: {pca_cv_scores.mean():.4f}")


best_k, best_score = 0, 0
for k in [3, 5, 7, 9, 11]:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=3).mean()
    print(f"K={k}, 平均CV得分={score:.4f}")
    if score > best_score:
        best_score = score
        best_k = k
print(f"最佳K值是 {best_k}，对应得分 {best_score:.4f}")