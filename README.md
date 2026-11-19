# PHÂN TÍCH VÀ XÂY DỰNG MÔ HÌNH DỰ ĐOÁN MẮC BỆNH TIỂU ĐƯỜNG SỬ DỤNG BỘ DỮ LIỆU DIABETES

Báo cáo Bài tập lớn môn **Phân tích Dữ liệu lớn (Big Data Analytics) - HaUI**.

---

## Thông tin chung
- **Đề tài:** PHÂN TÍCH VÀ XÂY DỰNG MÔ HÌNH DỰ ĐOÁN MẮC BỆNH TIỂU ĐƯỜNG SỬ DỤNG BỘ DỮ LIỆU DIABETES
- **Tập dữ liệu gốc (Kaggle):** [Diabetes Dataset - Ankit Batra](https://www.kaggle.com/datasets/ankitbatra1210/diabetes-dataset) (Quy mô: 70.000 dòng, 34 thuộc tính, 13 thể bệnh tiểu đường khác nhau).
- **Tài liệu Báo cáo chi tiết (Google Docs):** [Báo cáo BTL Nhóm 03](https://docs.google.com/document/d/1ag3rUzlGgeqpg1S8HndxZD-8Jwzw5NgTqOVC_bxllAE/edit?usp=sharing)
- **Tệp mã nguồn chính:** `ChuongTrinh.ipynb` (được tối ưu để chạy trên môi trường Jupyter Notebook cục bộ và hỗ trợ tính toán phân tán với Dask).

---

## Cấu trúc và Nội dung chi tiết của Chương trình (`ChuongTrinh.ipynb`)

Chương trình được tổ chức tuần tự, chặt chẽ gồm 3 phần chính từ khám phá dữ liệu, xây dựng mô hình phân loại đa lớp đến thực nghiệm đo lường hiệu năng tính toán phân tán:

```
ChuongTrinh.ipynb
├── 1. Khám phá, phân tích, trực quan hóa dữ liệu (EDA & Tiền xử lý)
├── 2. Xây dựng mô hình
│   ├── 2.1. Huấn luyện và tối ưu mô hình phân loại đa lớp
│   ├── 2.2. Đánh giá mô hình trên tập kiểm thử
│   └── 2.3. Giải thích mô hình XGBoost đa lớp (SHAP & Feature Importance)
└── 3. So sánh các thuật toán phân loại và đo lường hiệu năng đơn luồng so với phân tán
    ├── 3.1. Chuẩn bị môi trường đo
    ├── 3.2. So sánh 5 thuật toán phân loại
    ├── 3.3. Sinh dữ liệu benchmark có kiểm soát
    ├── 3.4. Thực nghiệm đơn luồng và phân tán
    └── 3.5. Bảng và biểu đồ tổng hợp kết quả
```

---

### 1. Khám phá, phân tích, trực quan hóa dữ liệu (EDA & Tiền xử lý)
Mục tiêu: Đọc, làm sạch, phân tích thống kê đa chiều và chuẩn bị pipeline tiền xử lý chống rò rỉ dữ liệu (*data leakage*).

- **Cài đặt thư viện & Nạp môi trường:** Cài đặt các gói cần thiết (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `kagglehub[pandas-datasets]`, `scikit-learn`, `xgboost`, `shap`).
- **Tải và nạp dữ liệu Kaggle:** Sử dụng `kagglehub` tải trực tiếp tập dữ liệu `ankitbatra1210/diabetes-dataset`, hỗ trợ tự động xử lý đọc file CSV nén và giải nén linh hoạt.
- **Tổng quan dữ liệu:**
  - Kiểm tra kích thước bảng dữ liệu: 70.000 dòng $\times$ 34 cột.
  - Kiểm tra giá trị rỗng/khuyết thiếu (0 ô thiếu) và kiểm tra trùng lặp (0 dòng trùng).
  - Phân loại rõ ràng 12 thuộc tính số học (`num_cols`), 21 thuộc tính nhị phân/định loại (`cat_cols`) và cột mục tiêu `Target`.
- **Thống kê mô tả & Phát hiện ngoại lai:**
  - Tính toán các chỉ số thống kê mô tả (Mean, Std, Min, 25%, 50%, 75%, Max, Skewness) cho các biến số học.
  - Áp dụng hàng rào Tukey ($1.5 \times \text{IQR}$) phát hiện và thống kê các điểm ngoại lai.
- **Phân tích biến mục tiêu (`Target`):**
  - Đếm số lượng mẫu và tỷ lệ phân bố trên 13 thể bệnh tiểu đường (mỗi lớp dao động 5.300 – 5.400 mẫu, phân bố tương đối cân bằng).
  - Vẽ biểu đồ cột phân bố tần suất 13 thể bệnh kèm đường mốc kỳ vọng trung bình.
- **Kiểm định thống kê:**
  - Kiểm định độ cân bằng lớp qua chỉ số Entropy chuẩn hóa ($H_{\text{norm}} \approx 1.0$).
  - Kiểm định phương sai một yếu tố ANOVA và phi tham số Kruskal-Wallis chứng minh sự khác biệt có ý nghĩa thống kê ($p < 0.001$) của các chỉ số sinh hóa giữa các nhóm bệnh.
- **Trực quan hóa nâng cao:**
  - *Boxplot Pulmonary Function:* Nhận diện sự suy giảm rõ rệt chức năng phổi ở bệnh nhân tiểu đường liên quan xơ nang (CFRD).
  - *Heatmap tương quan Pearson:* Phân tích tương quan giữa các biến số học (chỉ ra tương quan cao giữa Tuổi, Huyết áp, Cholesterol và Vòng eo).
  - *Profile Heatmap (Z-score):* Biểu diễn trực quan chân dung giá trị trung bình chuẩn hóa của từng biến số học trên từng thể bệnh.
- **Mã hóa và Phân tách dữ liệu:**
  - Dùng `LabelEncoder` mã hóa 13 thể bệnh sang các mã số $C_0 \dots C_{12}$.
  - Mã hóa các biến nhị phân định tính (Yes/No, Positive/Negative) về $0/1$.
  - Sử dụng `StratifiedShuffleSplit` phân tách dữ liệu thành tập **Train (80% - 56.000 mẫu)** và tập **Test (20% - 14.000 mẫu)**, bảo toàn tỷ lệ 13 lớp.
- **Xây dựng bộ tiền xử lý tùy biến (`DiabetesPreprocessor`):**
  - Kế thừa `BaseEstimator` và `TransformerMixin` của Scikit-Learn: thực hiện `StandardScaler` cho biến số và One-Hot Encoding cho các biến phân loại.
  - Áp dụng quy tắc nghiêm ngặt: chỉ `fit_transform` trên `X_train` và `transform` trên `X_test`, xác nhận kiểm tra $\text{mean} \approx 0$ và $\text{std} \approx 1$ để loại trừ hoàn toàn nguy cơ rò rỉ dữ liệu.

---

### 2. Xây dựng mô hình
Mục tiêu: Huấn luyện, tinh chỉnh tham số, đánh giá kiểm thử độc lập và bóc tách giải thích quyết định của mô hình.

#### 2.1. Huấn luyện và tối ưu mô hình phân loại đa lớp
- **Mô hình Baseline:** Huấn luyện mô hình Logistic Regression đa lớp với 3-fold Stratified Cross-Validation trên tập Train (Macro F1 đạt $\approx 0.7574$).
- **Mô hình XGBoost ban đầu:** Huấn luyện XGBoost với siêu tham số mặc định trên cùng 3 fold phân tầng (Macro F1 đạt $\approx 0.9041$).
- **Tối ưu siêu tham số XGBoost (RandomizedSearchCV):**
  - Khảo sát không gian tham số: `learning_rate` ($[0.03, 0.2]$), `max_depth` ($[3, 8]$), `n_estimators` ($[100, 400]$), `subsample` ($[0.7, 1.0]$).
  - Đánh giá 20 cấu hình $\times$ 3 fold = 60 lần thử nghiệm theo thang đo `f1_macro`.
  - Tự động fit cấu hình tối ưu trên toàn bộ tập Train (Macro F1 đạt $\approx 0.9049$, giảm phương sai giữa các fold).

#### 2.2. Đánh giá mô hình trên tập kiểm thử
- **Đánh giá trên tập kiểm thử độc lập (Test Set - 14.000 mẫu):**
  - So sánh chi tiết giữa Logistic Regression, XGBoost ban đầu và XGBoost đã tối ưu qua các độ đo: Accuracy, Macro/Weighted Precision, Recall, F1-Score, Macro ROC-AUC (One-vs-Rest).
  - XGBoost tối ưu đạt Accuracy $\approx 90.66\%$, Macro F1 $\approx 0.9054$, Macro ROC-AUC $\approx 0.9965$.
- **Đánh giá hiệu năng chi tiết theo từng lớp (Per-Class Performance):**
  - Lớp nhận diện hoàn hảo (F1 = 1.0000): $C_4$ (Neonatal Diabetes Mellitus - NDM) nhờ phân lập tuyệt đối từ đặc trưng cân nặng lúc sinh (`Birth Weight`).
  - Các lớp đạt độ chính xác rất cao (Recall = 1.0000): $C_5$ (Prediabetic), $C_8$ (Type 1), $C_{10}$ (Type 3c), $C_{12}$ (Wolfram).
  - Các lớp có độ nhầm lẫn cao: $C_9$ (Type 2 Diabetes, Recall đạt 66.17%), $C_6$ (Secondary Diabetes), $C_7$ (Steroid-Induced Diabetes) do đặc điểm lâm sàng tương đồng.
- **Trực quan hóa đánh giá:**
  - Biểu diễn Ma trận nhầm lẫn 13 $\times$ 13 (Confusion Matrix) dạng số đếm thực tế và dạng chuẩn hóa tỷ lệ phần trăm (Normalized).
  - Vẽ đồ thị các đường cong ROC One-vs-Rest cho toàn bộ 13 lớp bệnh.

#### 2.3. Giải thích mô hình XGBoost đa lớp (Model Explainability)
- **Tầm quan trọng của đặc trưng theo Gain:**
  - Xác định các đặc trưng đóng góp lớn nhất vào việc giảm hàm mất mát: Huyết áp (`Blood Pressure`: 15.94%), Tăng cân thai kỳ (`Weight Gain During Pregnancy`: 13.81%), Cân nặng sơ sinh (`Birth Weight`: 11.59%), Đường huyết (`Blood Glucose Levels`: 10.45%), Tuổi (`Age`: 9.28%).
- **Phân tích giá trị SHAP (TreeExplainer):**
  - Khởi tạo `shap.TreeExplainer` tính toán trên 650 mẫu kiểm thử đại diện (sai số bảo toàn cộng additivity tối đa chỉ $2.86 \times 10^{-6}$).
  - *SHAP Summary Bar:* Xếp hạng Top 5 thuộc tính chi phối chính cho từng thể bệnh cụ thể.
  - *SHAP Beeswarm Plot:* Phân tích chiều hướng tác động (dương/âm) của giá trị đặc trưng tới điểm margin log-odds của các lớp.
  - *Phân tích ca lỗi thực tế (Case Studies & Waterfall Plots):* Đối chiếu chi tiết đóng góp từng thuộc tính trong các ca đoán sai điển hình (ví dụ: Type 2 đoán nhầm sang Secondary Diabetes hoặc Steroid-Induced Diabetes).

---

### 3. So sánh các thuật toán phân loại và đo lường hiệu năng đơn luồng so với phân tán
Mục tiêu: Đánh giá độc lập 5 thuật toán học máy phổ biến và thực nghiệm đo lường khả năng mở rộng dữ liệu (Scalability) với tính toán phân tán.

#### 3.1. Chuẩn bị môi trường đo
- Cài đặt và cấu hình thư viện đo lường hiệu năng: `psutil`, `pyarrow`, `dask[distributed,dataframe]`.
- Xây dựng lớp đo lường tài nguyên phần cứng `ResourceMonitor`: Chạy trên luồng riêng để lấy mẫu mức tiêu thụ RAM liên tục, ghi nhận mức RAM đỉnh (Peak RAM) và thời gian thực thi chính xác.
- Khởi tạo cụm phân tán Dask cục bộ (`LocalCluster` và `Client`) với cơ chế dọn dẹp và giới hạn bộ nhớ theo số worker phù hợp với tài nguyên máy.

#### 3.2. So sánh 5 thuật toán phân loại
- Thực nghiệm huấn luyện và đánh giá độc lập 5 mô hình trên cùng tập dữ liệu:
  1. **Logistic Regression** (Mô hình tuyến tính)
  2. **Random Forest** (Mô hình quần thể cây bagging)
  3. **Decision Tree** (Cây quyết định đơn lẻ)
  4. **Gaussian Naive Bayes** (Phân loại xác suất Bayes)
  5. **XGBoost** (Mô hình tăng cường độ dốc tối ưu)
- Đánh giá toàn diện trên 2 nhóm chỉ số:
  - *Chất lượng phân loại:* Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, Macro ROC-AUC.
  - *Hiệu năng tài nguyên:* Thời gian huấn luyện (Training time - giây), Thời gian dự đoán (Inference time - giây), Lượng RAM tiêu thụ thêm (Memory usage - MB).
- Vẽ biểu đồ cột so sánh trực quan cả 3 khía cạnh hiệu năng của 5 mô hình.

#### 3.3. Sinh dữ liệu benchmark có kiểm soát
- Xây dựng hàm sinh dữ liệu bán nhân tạo (`generate_benchmark_split`) dựa trên phân phối xác suất và thống kê thực của tập dữ liệu gốc.
- Cho phép mở rộng quy mô dữ liệu theo các hệ số nhân có kiểm soát (Small, Medium, Large) mà vẫn giữ nguyên phân phối lớp và tính chất sinh học của các thuộc tính.
- Kiểm thử đối sánh tính đúng đắn: dữ liệu sinh trên máy đơn và sinh phân tán đảm bảo đồng nhất tuyệt đối.

#### 3.4. Thực nghiệm đơn luồng và phân tán
- Thiết lập kịch bản thực nghiệm huấn luyện XGBoost qua các cấp độ quy mô dữ liệu:
  - Chế độ đơn luồng / tuần tự (Single-thread baseline).
  - Chế độ phân tán với cụm Dask (Distributed execution).
- Ghi nhận chi tiết thời gian huấn luyện và mức RAM đỉnh qua từng mốc dữ liệu.

#### 3.5. Bảng và biểu đồ tổng hợp kết quả
- Tổng hợp bảng so sánh chi tiết giữa tính toán đơn luồng và tính toán phân tán.
- Trực quan hóa đồ thị so sánh thời gian huấn luyện và độ gia tăng bộ nhớ theo quy mô dữ liệu.
- Phân tích ngưỡng hiệu quả (Speedup threshold), chi phí truyền thông/điều phối cụm (Communication overhead) và tự động giải phóng tài nguyên cụm Dask sau khi hoàn thành.

---

## Hướng dẫn cài đặt và Chạy chương trình

### 1. Yêu cầu hệ thống
- Python $\ge$ 3.10
- Môi trường chạy: Jupyter Notebook / JupyterLab / VS Code / Antigravity IDE

### 2. Cài đặt các thư viện phụ thuộc
Mở terminal và cài đặt các thư viện cần thiết:
```bash
pip install pandas numpy matplotlib seaborn scipy kagglehub scikit-learn xgboost joblib shap psutil pyarrow "dask[distributed,dataframe]"
```

### 3. Chạy chương trình
1. Mở tệp `ChuongTrinh.ipynb`.
2. Chọn Kernel Python tương ứng.
3. Bấm **Run All** để chạy toàn bộ chương trình từ Phần 1 đến Phần 3.
*(Lưu ý: Dataset từ Kaggle sẽ được tải và lưu trong thư mục cache tự động, không yêu cầu can thiệp thủ công).*
