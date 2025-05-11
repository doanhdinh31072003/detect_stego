import numpy as np
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data import texts, labels  # 
# Hàm trích xuất đặc trưng
def extract_features(text):
    words = text.split()
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    whitespace_ratio = text.count(' ') / len(text) if len(text) > 0 else 0
    double_spaces = text.count("  ")

    num_letters = sum(c.isalpha() for c in text)
    letter_ratio = num_letters / len(text) if len(text) > 0 else 0

    first_letters = [w[0].lower() for w in words if w]
    first_letter_counts = [first_letters.count(ch) for ch in string.ascii_lowercase]
    most_common_first_letter = max(first_letter_counts) if first_letter_counts else 0

    return [
        avg_word_len,
        whitespace_ratio,
        double_spaces,
        letter_ratio,
        most_common_first_letter
    ]

# Chuẩn bị dữ liệu
X = np.array([extract_features(text) for text in texts])
y = np.array(labels)

# Chia dữ liệu huấn luyện / kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Huấn luyện mô hình
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

print("\nstart")
# Nhập văn bản từ người dùng để dự đoán
print("\nNhập văn bản để kiểm tra (tự nhiên hoặc giấu tin hide):")
user_input = input(">> ")

# Trích xuất đặc trưng và dự đoán
user_features = np.array([extract_features(user_input)])
prediction = clf.predict(user_features)[0]
probs = clf.predict_proba(user_features)[0]

# Kết quả
print(f"Training thành công kết quả lưu tại prediction_result.txt")
label_map = {0: "TỰ NHIÊN", 1: "GIẤU TIN"}
# Ghi xác suất dự đoán ra file
with open("prediction_result.txt", "w", encoding="utf-8") as f:
    f.write("Kết quả dự đoán: {}\n".format(label_map[prediction]))
    f.write("Xác suất:\n")
    f.write(" - Tự nhiên (0): {:.2f}%\n".format(probs[0]*100))
    f.write(" - Giấu tin (1): {:.2f}%\n".format(probs[1]*100))
