# Kiểm chứng thực thi

Ngày kiểm chứng: 16/10/2026 (Asia/Bangkok).

- Dữ liệu sao chép khớp SHA-256 với hai CSV nguồn của dự án SQL; không sửa file nguồn.
- Đối soát: 600 khách; 3.200 đơn; 2.998 Completed; doanh thu Completed 53.357.307.150 VND.
- Tập Train: 283 khách, 201 có mua lại. Validation: 397 khách, 267 có mua lại. Test: 489 khách, 315 có mua lại.
- Mỗi cặp khách–ngày chốt có đúng một dòng. Nhãn của mỗi bước trước hoàn tất trước bước sau.
- Notebook đã chạy đủ 8 ô code, không có error output, có kết quả và biểu đồ được lưu.
- 5 kiểm tra temporal integrity đã qua: sửa đơn tương lai không đổi đặc trưng; đúng biên cửa sổ lịch sử/nhãn; từ chối nhãn khi chưa đủ 90 ngày; không tính đơn hủy; một dòng mỗi khách ở mỗi mốc.
- Xác suất Test nằm trong [0,1]. File minh họa cuối năm có 575 khách, cửa sổ kết quả đến 31/03/2026, không có nhãn được bịa cho năm 2026.
- Biểu đồ evaluation.png đã được kiểm tra trực quan. Báo cáo HTML dùng biểu đồ nhúng, không phụ thuộc máy chủ ngoài.

Kết quả số học không xác minh giả định Completed đã biết vào order_date. Nguồn không có completed_at để kiểm tra điều này. Xem README và NOTION_ML.md để hiểu phạm vi đánh giá.
