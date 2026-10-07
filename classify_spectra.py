import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import spec_utils as su
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ================= 核心修改：修改波数范围为拉曼指纹区 400~1500 =================
STANDARD_X = np.linspace(400, 1500, 1000) 

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
                x, intensity = su.load_spectrum(filepath)
                px, py = su.preprocess_pipeline(x, intensity)
                
                if px[0] > px[-1]:
                    px = px[::-1]
                    py = py[::-1]
                
                py_aligned = np.interp(STANDARD_X, px, py)
                X.append(py_aligned) 
                y.append(label)
                
    return np.array(X), np.array(y)


if __name__ == "__main__":
    print("正在加载并预处理数据...")
    X, y = build_feature_matrix()
    
    if len(X) == 0:
        print("错误：未找到数据！")
        raise SystemExit(1)
        
    print(f"数据加载完毕，样本数: {len(X)}, 类别数: {len(set(y))}")
    print(f"当前使用的波数范围: {STANDARD_X[0]} ~ {STANDARD_X[-1]} cm^-1")
    
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # ================= 1. 基础 KNN 模型 =================
    knn_base_pipeline = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )
    knn_base_pipeline.fit(Xtr, ytr)
    
    score = knn_base_pipeline.score(Xte, yte)
    print(f"\n===== KNN 实验结果 =====")
    print(f"KNN 测试集准确率: {score:.4f}")

    y_pred = knn_base_pipeline.predict(Xte)
    print("\n--- 混淆矩阵 (KNN) ---")
    print(confusion_matrix(yte, y_pred))
    print("\n--- 分类报告 (KNN) ---")
    print(classification_report(yte, y_pred))

    print("--- 交叉验证 (KNN) ---")
    cv_scores = cross_val_score(knn_base_pipeline, X, y, cv=3)
    print(f"3折交叉验证得分: {cv_scores}")
    print(f"平均交叉验证得分: {cv_scores.mean():.4f}")


 # ================= 2. PCA + KNN 对比实验 =================
    pca_knn_pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=0.95), 
        KNeighborsClassifier(n_neighbors=3)
    )
    pca_cv_scores = cross_val_score(pca_knn_pipeline, X, y, cv=3)
    print(f"\n===== PCA+KNN 对比实验 =====")
    print(f"PCA+KNN 平均交叉验证得分: {pca_cv_scores.mean():.4f}")


    # ================= 3. 寻找最佳 K 值 =================
    print("\n===== 寻找最佳 K 值 (GridSearchCV) =====")
    param_grid = {'kneighborsclassifier__n_neighbors': [3, 5, 7, 9, 11]}
    knn_tuning_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier())
    grid_search = GridSearchCV(knn_tuning_pipeline, param_grid, cv=3)
    grid_search.fit(X, y)
    
    print(f"最佳K值是 {grid_search.best_params_['kneighborsclassifier__n_neighbors']}，"
          f"对应得分 {grid_search.best_score_:.4f}")


    # ================= 4. SVM 对比实验 (补全结果) =================
    print("\n===== SVM 对比实验 =====")
    svm_pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel='linear', random_state=42)
    )
    svm_pipeline.fit(Xtr, ytr)
    
    svm_score = svm_pipeline.score(Xte, yte)
    print(f"SVM 测试集准确率: {svm_score:.4f}")
    
    # 补全 SVM 的混淆矩阵和分类报告
    y_pred_svm = svm_pipeline.predict(Xte)
    print("\n--- 混淆矩阵 (SVM) ---")
    print(confusion_matrix(yte, y_pred_svm))
    print("\n--- 分类报告 (SVM) ---")
    print(classification_report(yte, y_pred_svm))
    
    # 补全 SVM 的交叉验证
    print("--- 交叉验证 (SVM) ---")
    svm_cv_scores = cross_val_score(svm_pipeline, X, y, cv=3)
    print(f"SVM 3折交叉验证得分: {svm_cv_scores}")
    print(f"SVM 平均交叉验证得分: {svm_cv_scores.mean():.4f}")


   