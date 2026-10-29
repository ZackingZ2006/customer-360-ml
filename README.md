# Customer 360 ML — Dự đoán mua lại trong 90 ngày

Một thử nghiệm mở rộng [Customer 360](https://github.com/ZackingZ2006/customer-360), dự đoán khách đã từng mua có phát sinh đơn hoàn tất trong 90 ngày tiếp theo. Dữ liệu là giả lập năm 2025; mục tiêu của bài là đánh giá quy trình mô hình hóa và so sánh với quy tắc đơn giản.

![Đánh giá mô hình trên tập test](evaluation.png)

## Kết quả chính

Mô hình Logistic Regression được chọn bằng tập validation. Trên một mốc test gồm 489 khách, Average Precision đạt **0,729**; quy tắc dựa trên lần mua gần nhất cũng đạt **0,729** và có Precision@20% cao hơn (**78,6% so với 74,5%**). Kết quả chưa cho thấy lợi thế rõ ràng để thay quy tắc đơn giản bằng ML.

## Phương pháp

- Tạo đặc trưng từ 90 ngày lịch sử tại từng ngày chốt; nhãn là có mua lại trong 90 ngày tiếp theo.
- Chia train, validation và test theo thời gian; nhãn của mốc trước hoàn tất trước mốc sau.
- So sánh Logistic Regression, Random Forest, quy tắc recency/frequency và xác suất nền. Chọn mô hình trên validation, sau đó đánh giá trên test.
- Kiểm tra rò rỉ dữ liệu theo thời gian, biên cửa sổ, đơn hủy và nhãn chưa đủ thời gian quan sát.

## Tệp và cách chạy

- [Notebook](Customer360_ML.ipynb): phân tích và kết quả đã lưu.
- [Mã Python](customer360_ml.py): chuẩn bị dữ liệu, huấn luyện và đánh giá.
- [Customer360_ML.zip](Customer360_ML.zip): toàn bộ mã, hai bảng dữ liệu giả lập, kết quả và hướng dẫn. Giải nén và làm theo README trong thư mục để chạy lại.
- [Dự án Customer 360 gốc](https://github.com/ZackingZ2006/customer-360).

## Giới hạn

Dữ liệu gồm 600 khách và một năm giả lập, với một mốc test; không thể kết luận mô hình sẽ hoạt động tốt trên dữ liệu thực hoặc qua các mùa. Nguồn thiếu thời điểm hoàn tất và lịch sử thay đổi đơn nên bài giả định trạng thái cuối đã biết tại ngày đặt. Điểm mua lại không đo được tác động của ưu đãi; cần dữ liệu thực, thêm các mốc kiểm tra và thử nghiệm có đối chứng trước khi ứng dụng.
