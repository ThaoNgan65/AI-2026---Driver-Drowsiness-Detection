# Driver Drowsiness Detection

Hệ thống **phát hiện dấu hiệu buồn ngủ của tài xế theo thời gian thực** bằng webcam, phục vụ dự án tham gia **Cuộc thi Sáng tạo trẻ Quốc gia 2026 – Bảng C (Trí tuệ nhân tạo)**.

Ứng dụng dùng camera để theo dõi khuôn mặt, phân tích các dấu hiệu liên quan đến buồn ngủ (nhắm mắt kéo dài, ngáp, gật gù, cúi đầu sâu), cảnh báo bằng âm thanh và giao diện desktop.

---

## Tính năng chính

| Kênh phát hiện | Mô tả ngắn |
|----------------|------------|
| **Mắt nhắm (EAR)** | Eye Aspect Ratio + timer (~1.8s), có làm mượt và chống reset khi camera rung |
| **Ngáp (MAR)** | Mouth Aspect Ratio duy trì đủ thời gian |
| **Gật gù (Nod)** | Chu kỳ cúi → ngẩng, cần đủ 2 lần trong cửa sổ thời gian |
| **Cúi sâu (Deep)** | Giữ đầu cúi quá ngưỡng đủ lâu |

**Thích ứng môi trường**

- Kính râm: bỏ qua EAR khi vùng mắt không đáng tin
- Thiếu sáng: tăng cường ảnh nhẹ + nới ngưỡng EAR
- Chói sáng (glare): nén highlight; mắt cháy sáng → không báo nhắm mắt sai
- Camera rung: khóa Nod; EAR vẫn theo dõi với tolerance ngắn
- Camera bị che: trạng thái `CAMERA OBSTRUCTED`, không báo buồn ngủ

**An toàn hệ thống**

- Watchdog / heartbeat: detection dừng bất thường → `SYSTEM ERROR`
- Ghi log cảnh báo (`alerts_log.csv`) và lỗi hệ thống (`system_errors.log`)

---

## Yêu cầu hệ thống

- **Hệ điều hành:** Windows 10/11 (khuyến nghị; `winsound` dùng cho beep)
- **Python:** 3.9 – 3.11 (khuyến nghị môi trường ảo / conda)
- **Phần cứng:** Webcam; chạy CPU, không bắt buộc GPU mạnh
- **Model:** file `face_landmarker.task` (MediaPipe Face Landmarker)

---

## Cài đặt

### 1. Tạo môi trường (ví dụ conda)

```bash
conda create -n drowsy python=3.10 -y
conda activate drowsy
```

### 2. Cài thư viện

```bash
pip install -r requirements.txt
```

### 3. Tải model MediaPipe

Tải file model và đặt **cùng thư mục** với script:

- URL:  
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task  
- Tên file: `face_landmarker.task`

### 4. Chạy ứng dụng

```bash
python drowsiness_gui_v3.py
```

1. Màn hình giới thiệu → **Bắt đầu / Vào ứng dụng**  
2. **Start Camera** để bắt đầu detection  
3. **Stop Camera** / **Exit** khi kết thúc  

---

## Cấu trúc thư mục gợi ý

```text
.
├── drowsiness_gui_v3.py   # Ứng dụng chính (GUI + detection)
├── face_landmarker.task   # Model MediaPipe (tự tải)
├── requirements.txt
├── README.md
├── alerts_log.csv         # Tự tạo khi có cảnh báo
└── system_errors.log      # Tự tạo khi có lỗi hệ thống
```

---

## Cách hoạt động (tóm tắt kỹ thuật)

1. **MediaPipe Face Landmarker** (Tasks API, chế độ VIDEO) lấy 468 landmark khuôn mặt realtime.  
2. Tính **EAR / MAR** và ước lượng **pitch / yaw** từ landmark.  
3. Các kênh độc lập (mắt, ngáp, gật, cúi sâu) kết hợp theo logic **OR**.  
4. Cảnh báo khi điều kiện đủ thời gian / số lần; beep định kỳ khi đang WARNING.  
5. GUI **Tkinter** + thread detection để giao diện không bị treo.

Không dùng mô hình deep learning nặng thêm ngoài Face Landmarker; phù hợp máy sinh viên và dễ giải thích với giám khảo.

---

## Phím / nút điều khiển

| Điều khiển | Chức năng |
|------------|-----------|
| Bắt đầu / Vào ứng dụng | Vào màn hình chính |
| Start Camera | Bật camera + detection |
| Stop Camera | Tắt detection |
| Exit | Thoát ứng dụng |
| ← Giới thiệu | Về màn hình giới thiệu |

---

## Log

- **`alerts_log.csv`:** thời gian bắt đầu/kết thúc, thời lượng, loại cảnh báo (Mắt nhắm / Ngáp / Gật gù / Cúi sâu).  
- **`system_errors.log`:** lỗi init, camera, detection, watchdog.

---

## Lưu ý khi demo / kiểm thử

- Ngồi đủ sáng, mặt trong khung hình; thử nhắm mắt ~2s, há miệng (ngáp), gật đầu 2 lần.  
- Tránh rung camera mạnh khi test Nod.  
- Che camera vài giây để kiểm tra `CAMERA OBSTRUCTED`.  
- Nếu Start lỗi: kiểm tra webcam, file `face_landmarker.task`, và log trong `system_errors.log`.

---

## Giới hạn

- Chỉ dựa trên **dấu hiệu quan sát được từ khuôn mặt**, không khẳng định trạng thái buồn ngủ bên trong.  
- Beep (`winsound`) chủ yếu trên Windows.  
- Hiệu năng phụ thuộc webcam và CPU; độ phân giải đang dùng khoảng 640×480.

---

## Tác giả / mục đích

Dự án sinh viên – **Cuộc thi Sáng tạo trẻ Quốc gia 2026**, lĩnh vực Trí tuệ nhân tạo (Bảng C).