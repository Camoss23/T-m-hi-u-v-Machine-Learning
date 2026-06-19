import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# 1. Chuyển thư mục làm việc và đọc data
os.chdir('ML/Benh')
data = pd.read_csv('diabetes_data.csv')

# 2. Tách tập thuộc tính và nhãn mục tiêu
target = data['DiabeticClass']
X = data.drop(['DiabeticClass'], axis=1) # axis=1 tự chạy ngon lành, không cần import gì hết!

# 3. Phân loại lại nhóm cột (Đã xóa 'DiabeticClass' ra khỏi đây)
numerical_col = ['Age']
OneHot_col = ['Gender', 'ExcessUrination', 'Polydipsia', 'WeightLossSudden', 'Fatigue',
              'Polyphagia', 'GenitalThrush', 'BlurredVision', 'Itching', 'Irritability',
              'DelayHealing', 'PartialPsoriasis', 'MuscleStiffness', 'Alopecia', 'Obesity']

# 4. Tách dữ liệu Train/Test (Đã sửa từ 'data' thành 'X')
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

# 5. Tạo Pipeline con cho từng nhóm dữ liệu
num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Thêm để không lỗi ma trận thưa
])

# 6. Gộp các bộ xử lý lại bằng ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numerical_col),
    ('cat', cat_pipeline, OneHot_col)
])

# 7. Đóng gói vào Pipeline tổng kèm thuật toán Random Forest
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 8. Tiến hành Huấn luyện và Đánh giá độ chính xác
model.fit(X_train, y_train)

train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Độ chính xác tập Train: {train_accuracy:.2%}")
print(f"Độ chính xác tập Test: {test_accuracy:.2%}")