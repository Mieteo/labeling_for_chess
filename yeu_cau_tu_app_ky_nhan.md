# Yêu cầu bổ sung từ app "Xiangqi Scanner & AI Analyzer" (Ky Nhan) — hỗ trợ gán nhãn ảnh "Digital"

Tài liệu này **bổ sung**, không thay thế, `labeling_tool_requirements.md`. Đọc
tài liệu gốc trước — mọi ràng buộc ở đó (định dạng YOLO `.txt`, `classes.txt`
15 lớp cố định thứ tự, không rewrite file ảnh không đổi, v.v.) vẫn giữ nguyên
100%, áp dụng cho cả dữ liệu Digital mô tả dưới đây.

Ghi ngày 28/08/2026. Tự chứa (self-contained) — agent đọc tài liệu này không
cần quyền truy cập repo app gốc (`Ky Nhan`), các trích dẫn đường dẫn/số liệu
dưới đây đã được xác nhận trực tiếp trong mã nguồn app tại thời điểm viết.

## 0. Bối cảnh

App "Ky Nhan" đang lên kế hoạch retrain lại model phát hiện quân cờ trên ảnh
chụp màn hình phần mềm cờ tướng (gọi là **"Digital"**, phân biệt với
**"Physical"** = ảnh chụp bàn cờ gỗ thật mà `labeling_for_chess` đang phục vụ
hiện nay) — xem file kế hoạch phía app,
`docs/phase5_digital_model_retrain_plan.md`. Lý do: model Digital hiện tại
(`xiangqi_piece_detector_v1.onnx`) dựa trên kiến trúc Ultralytics YOLO (ràng
buộc license thương mại), và dữ liệu training gốc (2022) đã mất — buộc phải
thu thập ảnh mới + train lại từ đầu trên kiến trúc Apache-2.0 khác (YOLOX-Nano
hoặc NanoDet, chưa chốt).

Nhiệm vụ gán nhãn cho Digital **giống hệt về bản chất** với Physical đang làm:
khoanh box quanh từng quân cờ trên ảnh gốc (không cần góc bàn, không cần crop
90 giao điểm — pipeline Digital giữ nguyên "detect toàn ảnh", không chuyển
sang paradigm góc+crop của Physical). Vì vậy quyết định của chủ dự án là
**tiếp tục dùng `labeling_for_chess`** cho Digital thay vì xây tool riêng —
tài liệu này liệt kê các điều chỉnh cần thiết để tool phục vụ tốt cả 2 loại
dữ liệu.

## 1. Thư mục ảnh Digital là thư mục riêng biệt, không trộn với `chessImg`

- Ảnh Digital sẽ nằm trong 1 thư mục phẳng riêng (ví dụ `digitalImg`), tách
  biệt hoàn toàn khỏi thư mục ảnh Physical hiện có — đúng nguyên tắc "1 thư
  mục phẳng" ở mục 1.1 tài liệu gốc, chỉ là 2 thư mục độc lập, mỗi thư mục có
  `classes.txt` + các `<tên>.txt` riêng của nó.
- **Không cần đổi danh sách 15 lớp** — dùng nguyên `classes.txt` 15 dòng hiện
  có (mục 1.2 tài liệu gốc: `red_king` … `black_pawn`, `hand`). Lý do giữ
  nguyên thay vì rút gọn: đơn giản hoá (1 schema class duy nhất cho toàn bộ
  tool, script `tool/unify_scanner_labels.py` phía app gộp nhãn theo *tên*
  lớp nên không phát sinh vấn đề nếu 1 lớp không dùng tới). Lớp `hand` gần
  như sẽ luôn rỗng với ảnh Digital (không có tay người che ảnh chụp màn
  hình) — đó là kết quả bình thường, không phải lỗi, không cần xử lý gì
  thêm.

## 2. Circle-detect (mục 4 tài liệu gốc) không áp dụng cho Digital — cần ẩn, không cần xoá

Tài liệu gốc đã tự nêu rõ: tính năng phát hiện hình tròn "chỉ áp dụng cho ảnh
chụp bàn cờ thật... không áp dụng cho ảnh chụp màn hình" (mục 4, đoạn đầu).
Hiện IO là người dùng tự biết không bấm nút đó khi làm Digital. Yêu cầu thêm:

- Thêm khái niệm **"chế độ ảnh" theo thư mục**: `physical` hoặc `digital`.
  Suy luận mặc định đơn giản nhất — theo tên thư mục đang mở (không cần
  heuristic phức tạp), có thể lưu tường minh trong file trạng thái phiên
  `.labeling_session.json` đã có sẵn (mục 5 tài liệu gốc) để không phải đoán
  lại mỗi lần mở, và cho override tay qua 1 menu/toggle nếu suy luận sai.
- Khi ở chế độ `digital`: **ẩn** (không cần xoá code) toàn bộ UI liên quan
  circle-detect (nút auto-scan, radius-guided, slider dung sai). Tránh người
  dùng bấm nhầm rồi thắc mắc vì sao không ra gợi ý nào trên ảnh vuông vắn của
  screenshot.
- Auto-detect bằng model AI ở mục 3 dưới đây thay thế vai trò "tăng tốc gán
  tay" cho Digital, đúng vị trí mà circle-detect đang giữ cho Physical.

## 3. Tính năng mới — Auto-detect bằng model AI có sẵn của app (`xiangqi_piece_detector_v1.onnx`)

**Bối cảnh/lý do:** đánh giá thực tế của chủ dự án về model Digital hiện tại
(chính model sắp bị thay thế): **vị trí quân và màu quân đã tốt**, điểm yếu
chỉ nằm ở **độ chính xác phân loại vai trò quân** (xem
`docs/phase5_digital_model_retrain_plan.md` mục "Vì sao cần làm ngay"). Vì
vậy dùng chính model cũ này để **pre-label** (sinh box + lớp gợi ý) cho ảnh
Digital mới thu thập là hợp lý — giảm phần lớn công vẽ box tay, người dùng
chỉ cần **review và sửa lớp sai** (thường chỉ sai vai trò, ít khi sai vị
trí/màu), thay vì vẽ từ đầu. Đây **không phải** train/fine-tune gì trong tool
— chỉ chạy inference (suy luận) model đã có sẵn, dùng đúng tinh thần "gợi ý
cần người xác nhận" y hệt circle-detect.

Điều này **nới rộng** mục "ngoài phạm vi" cũ của tài liệu gốc (mục 7: "Không
cần train hay chạy model AI trong tool này") — câu đó viết cho bối cảnh chỉ
có Physical/circle-detect; nay bổ sung đúng 1 ngoại lệ có kiểm soát này cho
Digital, các giới hạn khác của mục 7 gốc vẫn giữ nguyên.

### 3.1. Spec kỹ thuật model (đã xác nhận từ mã nguồn app `Ky Nhan`, KHÔNG suy đoán)

- **File model:** `assets/models/xiangqi_piece_detector_v1.onnx` trong repo
  app. Model không có sẵn trong repo `labeling_for_chess` — người dùng copy
  thủ công 1 lần vào máy dev, tool chỉ cần cho phép **trỏ đường dẫn file
  .onnx** qua UI/config (không vendor lại file model trong repo
  `labeling_for_chess`, tránh 2 nơi lưu cùng 1 binary phải đồng bộ tay).
- **Input tensor:** tên `images`, `float32`, shape `[1, 3, 640, 640]` (NCHW),
  kênh màu RGB, giá trị chuẩn hoá `[0, 1]` (chia 255). Tiền xử lý: resize
  giữ tỉ lệ khung hình kiểu **letterbox** vào canvas vuông 640×640, phần đệm
  tô màu xám `(114, 114, 114)`.
- **Output tensor:** tên `output0`, `float32`, shape `[1, 19, 8400]` = 4 kênh
  box (`cx, cy, w, h`, đơn vị pixel theo ảnh **đã letterbox** 640×640) + 15
  kênh điểm số lớp. Đây là format xuất chuẩn **Ultralytics YOLOv8
  single-tensor** — **không có kênh objectness riêng**, điểm số lớp đã là
  xác suất trực tiếp (không cần áp sigmoid thêm khi decode).
- **Ngưỡng mặc định (đúng giá trị app đang dùng):** confidence threshold
  `0.25`, NMS IoU threshold `0.45`, NMS chạy **theo từng lớp riêng**
  (per-label, không gộp NMS toàn cục). Nên giữ các giá trị này làm mặc định
  trong tool (để so sánh nhất quán với app), nhưng **cho phép chỉnh qua UI**
  (ảnh Digital tự thu thập có thể cần ngưỡng khác ảnh benchmark cũ).
- **Un-letterbox bắt buộc:** box giải mã ra ở toạ độ ảnh 640×640 đã letterbox
  — phải biến đổi ngược về toạ độ pixel tuyệt đối trên **ảnh gốc** (kích
  thước thật của file ảnh Digital) trước khi hiển thị hoặc ghi ra `.txt`
  chuẩn hoá `[0,1]`. Không được ghi thẳng toạ độ letterbox.

### 3.2. Bảng ánh xạ lớp — BẮT BUỘC đọc kỹ, rủi ro "sai lặng lẽ" cao nhất của tính năng này

Thứ tự 15 lớp của **model** (dùng ký hiệu FEN: chữ thường = quân đen, chữ hoa
= quân đỏ) **khác hoàn toàn** thứ tự 15 lớp của **`classes.txt`** trong
`labeling_for_chess` (mục 1.2 tài liệu gốc). Nếu map theo index thay vì theo
tên, mọi gợi ý sinh ra sẽ **sai lớp một cách âm thầm** — đúng loại lỗi mà
tài liệu gốc mục 1.4 đã cảnh báo cho trường hợp classes.txt, giờ áp dụng y
hệt cho bảng ánh xạ model→tool này:

| Index model | Ký tự | Ý nghĩa | Tên lớp trong `classes.txt` của tool |
| --- | --- | --- | --- |
| 0 | `n` | mã đen | `black_horse` |
| 1 | `b` | tượng đen | `black_elephant` |
| 2 | `a` | sĩ đen | `black_advisor` |
| 3 | `k` | tướng đen | `black_king` |
| 4 | `r` | xe đen | `black_rook` |
| 5 | `c` | pháo đen | `black_cannon` |
| 6 | `p` | tốt đen | `black_pawn` |
| 7 | `R` | xe đỏ | `red_rook` |
| 8 | `N` | mã đỏ | `red_horse` |
| 9 | `A` | sĩ đỏ | `red_advisor` |
| 10 | `K` | tướng đỏ | `red_king` |
| 11 | `B` | tượng đỏ | `red_elephant` |
| 12 | `C` | pháo đỏ | `red_cannon` |
| 13 | `P` | tốt đỏ | `red_pawn` |
| 14 | `0` | không phải quân cờ (vùng bàn cờ, nội bộ model dùng để định vị bàn) | **Không map sang lớp nào cả — loại bỏ hoàn toàn khỏi danh sách gợi ý**, không phải `hand`, không phải lớp nào khác. |

Danh sách gốc (tham chiếu để đối chiếu khi cần, lấy từ
`lib/src/core/scanner/xiangqi_pwa_baseline.dart` phía app,
hằng số `digitalBoardModelLabels`):
`['n','b','a','k','r','c','p','R','N','A','K','B','C','P','0']`

### 3.3. Tham chiếu implementation có sẵn — nên adapt, không viết lại từ đầu

Repo app `Ky Nhan` đã có sẵn 1 bản Python độc lập (dùng `onnxruntime`) tái
hiện đúng pipeline preprocess/decode/NMS/letterbox ở trên, dùng để benchmark
offline:

```
benchmark_runs/phase5_digital_detector_real_probe_20260827/digital_detector_probe.py
```

File này là điểm khởi đầu tốt nhất để adapt logic inference cho tool —
tránh viết lại decode/NMS từ đầu và tránh lệch kết quả so với app thật.

**Lưu ý quan trọng — tránh nhầm:** `tool/xiangqi_pwa_benchmark.py` trong cùng
repo app là script cho **1 model khác, cũ hơn**
(`Entity_chess_recognition_model.onnx`), **không phải** model Digital hiện
tại. Không dùng file đó làm tham chiếu cho tính năng này.

Dependency cần thêm cho `labeling_for_chess`: `onnxruntime` (Python, MIT
license — không phát sinh vấn đề license so với tinh thần permissive đã chốt
ở mục 6 tài liệu gốc). Máy dev đã cài `onnxruntime` từ trước cho các việc
Phase 5 khác của app, theo `docs/phase5_scanner_model_independence.md`.

### 3.4. Hành vi UI bắt buộc — tái dùng đúng tinh thần "gợi ý chưa xác nhận" của circle-detect

1. Auto-detect là hành động người dùng chủ động bấm (nút/phím tắt), có 2 chế
   độ: **ảnh hiện tại** và **chạy hàng loạt** cho toàn bộ ảnh chưa gán nhãn
   trong thư mục (mục tiêu chính: giảm công cho toàn bộ ~200 ảnh sắp thu
   thập, không chỉ từng ảnh lẻ).
2. Mỗi box+lớp model đề xuất hiển thị ở trạng thái **"gợi ý — chưa xác
   nhận"** — tái dùng đúng cơ chế trực quan (nét đứt vs nét liền) đã có sẵn
   từ circle-detect (mục 4.3 tài liệu gốc), không cần thiết kế mới.
3. Người dùng xử lý từng gợi ý bằng đúng thao tác đã có: **xác nhận** (giữ
   nguyên lớp model gán), **sửa lớp** (gõ phím tắt 1-phím sẵn có ở mục 3 tài
   liệu gốc để đổi vai trò/màu — dự kiến thao tác hay dùng nhất, vì model
   yếu nhất ở phân loại vai trò), **sửa vị trí/kích thước box** (kéo cạnh/tâm
   như box thường vẽ tay), hoặc **xoá** (false positive).
4. **TUYỆT ĐỐI không tự ghi gợi ý chưa xác nhận vào `.txt` khi lưu** — giữ
   đúng bất biến bắt buộc ở mục 4.5 tài liệu gốc, áp dụng y hệt cho auto-detect.
5. Nên có action "chạy lại auto-detect cho ảnh này" (ví dụ sau khi đổi
   ngưỡng confidence) và "xoá toàn bộ gợi ý chưa xác nhận của ảnh này" —
   song song 2 action tương tự đã có ở circle-detect (mục 4.6 tài liệu gốc).
6. **Hiệu năng:** 1 ảnh nên chạy dưới vài giây trên CPU thường (máy dev
   không có GPU rời bắt buộc). Chạy batch nhiều ảnh phải có progress bar,
   không block UI (chạy nền/thread riêng), cho phép huỷ giữa chừng.

## 4. Ngoài phạm vi (bổ sung, không thay thế mục 7 tài liệu gốc)

- Auto-detect chỉ là gợi ý tăng tốc gán nhãn — **không** phải bước đánh giá
  chất lượng model. Tool **không cần** tự tính/hiển thị precision-recall của
  auto-detect so với nhãn tay cuối cùng; việc so sánh model cũ vs model mới
  huấn luyện lại là việc của quy trình benchmark phía app
  (`docs/phase5_digital_model_retrain_plan.md` mục "Các bước triển khai"),
  không phải của tool gán nhãn.
- Không train/fine-tune bất kỳ model nào bên trong `labeling_for_chess` —
  chỉ chạy inference model đã export sẵn (`.onnx`).
- Không cần map ngược nhãn Digital về 90 giao điểm bàn cờ — Digital giữ
  nguyên paradigm detect toàn ảnh, không có bước rectify/crop như Physical.

## 5. Tài liệu liên quan

- [`labeling_tool_requirements.md`](labeling_tool_requirements.md) — tài
  liệu yêu cầu gốc, mọi ràng buộc định dạng/tương thích dữ liệu vẫn áp dụng.
- (phía repo app `Ky Nhan`) `docs/phase5_digital_model_retrain_plan.md` —
  kế hoạch retrain Digital, lý do và bối cảnh đầy đủ cho tài liệu này.
- (phía repo app `Ky Nhan`) `docs/phase2_model_provenance.md` — provenance
  và benchmark tham chiếu của model `xiangqi_piece_detector_v1.onnx` hiện
  đang dùng làm auto-labeler ở mục 3.
