import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Đọc dữ liệu
data = pd.read_csv('Data.csv')

# Xác định các đặc trưng và nhãn
X = data.drop('Result', axis=1)
y = data['Result']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tiêu chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Tạo mô hình SVM
svm_model = SVC(kernel='linear')

# Huấn luyện mô hình trên tập huấn luyện
svm_model.fit(X_train, y_train)

# Đánh giá mô hình trên tập huấn luyện
y_train_pred = svm_model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
classification_report_train = classification_report(y_train, y_train_pred)

# Đánh giá mô hình trên tập kiểm tra
y_test_pred = svm_model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)
classification_report_test = classification_report(y_test, y_test_pred)

# In kết quả
print("Độ chính xác trên tập huấn luyện:", accuracy_train)
print("Ma trận nhầm lẫn trên tập huấn luyện:")
print(confusion_matrix_train)
print("Báo cáo phân loại trên tập huấn luyện:")
print(classification_report_train)

print("Độ chính xác trên tập kiểm tra:", accuracy_test)
print("Ma trận nhầm lẫn trên tập kiểm tra:")
print(confusion_matrix_test)
print("Báo cáo phân loại trên tập kiểm tra:")
print(classification_report_test)
