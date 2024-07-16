from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk


data = pd.read_csv('milknew.csv')
features = ['pH', 'Temprature', 'Taste', 'Odor', 'Fat ', 'Turbidity', 'Colour']
feature_names = {
    'pH': 'pH',
    'Temprature': 'Temperature (Nhiệt độ)',
    'Taste': 'Taste (Hương vị)',
    'Odor': 'Odor (Mùi)',
    'Fat ': 'Fat (Chất béo)',
    'Turbidity': 'Turbidity (Độ đục)',
    'Colour': 'Colour (Màu sắc)'
}
X = data[features]
X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)
scaler = StandardScaler()
# tính toán giá trị trung bình và độ lệch chuẩn trong X_train và chuẩn hóa
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Nhập giá trị k bằng tay
k = 50

final_model = KMeans(
    n_clusters=k, # Số lượng cụm đã được nhập bằng k
    init='k-means++', # Phương pháp khởi tạo trung tâm cụm vì nó cho kết quả tốt và ổn định hơn random
    n_init=10, # Số lần chạy thuật toán với các trung tâm khác nhau để chọn ra mô hình tốt nhất
    random_state=42 # Đảm bảo rằng các lần chạy sau cùng mỗi khi chạy mô hình đều tạo ra kết quả như nhau, giúp tái tạo được
)

# Huấn luyện mô hình trên dữ liệu huấn luyện (X_train_scaled) và gán nhãn cho mỗi mẫu dữ liệu dựa trên cụm được dự đoán.
final_labels = final_model.fit_predict(X_train_scaled)

class ClusteringApp:
    def __init__(self, master):
        self.master = master
        master.title("Phân cụm các loại sữa")
        # master.configure(bg="#cb9967")  # Đặt màu nền cho form
        self.create_widgets()

    def create_widgets(self):
        # Ảnh
        self.image_label = tk.Label(self.master)
        self.image_label.grid(row=0, column=0, columnspan=2, padx=10, pady=20)

        self.load_image()

        # Tiêu đề
        self.cluster_label = tk.Label(self.master, text="PHÂN CỤM CÁC LOẠI SỮA", font=("Arial", 20, 'bold italic'), fg="red")
        self.cluster_label.grid(row=1, column=0, columnspan=2, pady=(20, 0))

        self.feature_entries = []
        for i, feature in enumerate(features):
            label_text = feature_names.get(feature, feature)
            label = tk.Label(self.master, text=label_text, font=("Arial", 12, 'bold'), anchor='w')
            label.grid(row=i + 2, column=0, padx=50, pady=5, sticky='e')

            entry = tk.Entry(self.master, width=20)
            entry.grid(row=i + 2, column=1, padx=50, pady=5, ipady=4, sticky='w')
            self.feature_entries.append(entry)

        # Tạo frame để chứa nút và ô hiển thị kết quả của nút "Dự đoán nhãn"
        predict_frame = tk.Frame(self.master)
        predict_frame.grid(row=len(features) + 5, column=0, pady=10, padx=(10, 5))

        predict_button = tk.Button(predict_frame, text="Dự đoán nhãn", command=self.predict_cluster, bg="red", fg="white")
        predict_button.pack(side=tk.LEFT)
        predict_button.config(font=("Arial", 14, 'bold'), width=16)

        self.predict_result_text = tk.Text(predict_frame, width=50, height=9, wrap='word')
        self.predict_result_text.pack(side=tk.LEFT, padx=(10, 0))
        self.predict_result_text.config(font=("Arial", 12))

        # Tạo frame để chứa nút và ô hiển thị kết quả của nút "Đánh giá mô hình"
        evaluate_frame = tk.Frame(self.master)
        evaluate_frame.grid(row=len(features) + 5, column=1, pady=10, padx=(5, 10))

        evaluate_button = tk.Button(evaluate_frame, text="Đánh giá mô hình", command=self.evaluate_model, bg="green", fg="white")
        evaluate_button.pack(side=tk.LEFT)
        evaluate_button.config(font=("Arial", 14, 'bold'), width=16)

        self.evaluate_result_text = tk.Text(evaluate_frame, width=50, height=9, wrap='word')
        self.evaluate_result_text.pack(side=tk.LEFT, padx=(10, 0))
        self.evaluate_result_text.config(font=("Arial", 12, 'bold'))

    def load_image(self):
        image_path = "nen.jpg"
        image = Image.open(image_path)
        image = image.resize((700, 200))
        image_tk = ImageTk.PhotoImage(image)
        self.image_label.configure(image=image_tk)
        self.image_label.image = image_tk

    def predict_cluster(self):
    # Lấy giá trị đầu vào từ các ô nhập liệu
        input_values = [entry.get() for entry in self.feature_entries]

        # Kiểm tra xem tất cả các ô nhập liệu có giá trị không rỗng không
        if all(input_values):
            # Chuyển đổi giá trị đầu vào sang kiểu float
            try:
                input_values = [float(value) for value in input_values]
            except ValueError:
                # Hiển thị thông báo nếu giá trị đầu vào không phải là số
                self.predict_result_text.delete(1.0, tk.END)
                self.predict_result_text.insert(tk.END, "Vui lòng nhập giá trị số cho tất cả các đặc trưng.")
                return

            # Chuẩn hóa giá trị đầu vào bằng cách sử dụng scaler đã được huấn luyện trước đó
            input_scaled = scaler.transform([input_values])

            # Dự đoán nhãn cụm cho giá trị đầu vào
            predicted_label = final_model.predict(input_scaled)[0]

            # Lấy trung tâm cụm và giá trị trung bình đặc trưng cho cụm dự đoán
            cluster_center = final_model.cluster_centers_[predicted_label]
            features_mean = X_train_scaled[final_labels == predicted_label].mean(axis=0)

            # Định dạng văn bản để hiển thị
            cluster_center_text = ", ".join(f"{value:.4f}" for value in cluster_center)
            features_mean_text = ", ".join(f"{value:.4f}" for value in features_mean)

            # Hiển thị số cụm bản ghi vừa nhập và tên nhãn cụm trong Text widget có tên predict_result_text
            result_info = f"Tên nhãn cụm: Nhãn {predicted_label}\n"
            self.predict_result_text.delete(1.0, tk.END)
            self.predict_result_text.insert(tk.END, f"{result_info}")

            # Lấy các điểm dữ liệu trong nhóm có nhãn dự đoán
            group_data = X_train_scaled[final_labels == predicted_label]

            # Tính giá trị trung bình của các đặc trưng trong nhóm
            average_features_in_group = group_data.mean(axis=0)

            # Hiển thị tên các đặc trưng và giá trị trung bình tương ứng
            feature_names_and_averages = zip(features, average_features_in_group)

            self.predict_result_text.insert(tk.END, "Các đặc trưng và giá trị trung bình (trung tâm cụm) trong nhóm:")
            for feature, average in feature_names_and_averages:
                self.predict_result_text.insert(tk.END, f"\n{feature}: {average:.4f}")
        else:
            # Hiển thị thông báo nếu không đủ giá trị nhập vào
            self.predict_result_text.delete(1.0, tk.END)
            self.predict_result_text.insert(tk.END, "Vui lòng nhập đủ giá trị cho tất cả các đặc trưng.")


    def evaluate_model(self):
        # Đánh giá mô hình bằng cách sử dụng Silhouette Score và Davies-Bouldin Score
        X_test_scaled = scaler.transform(X_test)
        test_labels = final_model.predict(X_test_scaled)
        silhouette = silhouette_score(X_test_scaled, test_labels)
        davies_bouldin = davies_bouldin_score(X_test_scaled, test_labels)

        # Định dạng văn bản để hiển thị thông tin đánh giá
        evaluation_text = (
            f"Độ đo Silhouette Score: {silhouette:.5f}\n"
            f"Độ đo Davies-Bouldin Score: {davies_bouldin:.5f}"
        )
        
        # Hiển thị thông tin đánh giá trong Text widget có tên evaluate_result_text
        self.evaluate_result_text.delete(1.0, tk.END)
        self.evaluate_result_text.insert(tk.END, evaluation_text)


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1360x765")
    app = ClusteringApp(root)
    root.mainloop()
