# Yêu cầu: Công cụ đánh nhãn nhẹ cho Xiangqi Scanner (thay thế labelImg)

Tài liệu này mô tả yêu cầu cho một AI coding agent khác xây dựng một công cụ
đánh nhãn (labeling tool) độc lập, dùng để gán nhãn tay cho ảnh bàn cờ tướng
(Xiangqi) chụp thực tế, phục vụ training model scanner mới của app
"Xiangqi Scanner & AI Analyzer". Tài liệu tự chứa (self-contained) — agent
đọc tài liệu này không cần quyền truy cập vào repo gốc của app.

## 0. Bối cảnh (vì sao cần tool mới)

- Model scanner hiện tại của app bị chặn phát hành thương mại vì license
  AGPL-3.0 (Ultralytics YOLO) và thiếu provenance dữ liệu train rõ ràng.
  Hướng khắc phục: train model thay thế bằng code tự viết + dữ liệu tự thu
  thập + framework permissive (PyTorch/torchvision, BSD-3-Clause).
- Dữ liệu tự thu thập: **4458 ảnh** chụp bàn cờ tướng thật (đa số `.jpg`, vài
  `.png`), đang được gán nhãn tay bằng **labelImg 1.8.6 (MIT license)**, chạy
  hoàn toàn local, không upload ảnh lên đâu cả (đã cân nhắc và huỷ dùng
  Roboflow vì gói free bắt buộc project Public).
- labelImg là tool đóng gói sẵn (`pip install labelImg`), không sửa được mã
  nguồn, thiếu tính năng gán nhãn nhanh bằng phím tắt theo lớp và không có
  hỗ trợ khoanh box tự động. Tool mới cần thay thế nó, **giữ nguyên khả năng
  đọc/ghi đúng định dạng dữ liệu đã có** (một phần ảnh đã được gán nhãn dở
  bằng labelImg từ trước, không được làm hỏng hoặc phải di chuyển dữ liệu
  này).
- Model downstream không phải object detector YOLO — nó là **classifier
  từng giao điểm** (per-intersection CNN). Việc phát hiện 4 góc bàn cờ và
  ánh xạ box về 90 giao điểm (9 cột x 10 hàng) đã có sẵn, chạy **hoàn toàn
  tự động** (Hough-based corner detector) ở bước xử lý sau, KHÔNG cần người
  dùng đánh dấu 4 góc bàn cờ bằng tay. Nhiệm vụ duy nhất của người gán nhãn
  là khoanh box quanh từng quân cờ (và tay che, nếu có) trên ảnh gốc, gán
  đúng lớp.

## 1. Bắt buộc: tương thích dữ liệu 100% với labelImg (không thương lượng)

Đây là ràng buộc quan trọng nhất — sai một trong các điểm dưới đây sẽ làm
hỏng lặng lẽ (silent corruption) toàn bộ tiến độ đã gán nhãn từ trước.

### 1.1. Cấu trúc thư mục

- Một thư mục ảnh **phẳng** (flat, không train/valid/test split). Ví dụ:
  `D:\Documents\Photoshop\chessImg\0001.jpg`, `0001.txt`, `0002.jpg`,
  `0002.png`, `classes.txt`, v.v — ảnh và nhãn nằm chung một thư mục.
- Với mỗi ảnh `<tên>.jpg` (hoặc `.jpeg`/`.png`) **đã được gán nhãn**, có một
  file `<tên>.txt` cùng tên (khác đuôi) trong **cùng thư mục**.
- **Ảnh chưa có `<tên>.txt` = "chưa được xem/gán nhãn"**, khác hẳn với "đã
  xem, xác nhận không có quân cờ nào" (trường hợp này `<tên>.txt` tồn tại
  nhưng rỗng — 0 dòng). Tool phải giữ được sự phân biệt này: cho phép người
  dùng lưu một `.txt` rỗng một cách tường minh (ví dụ hành động "Lưu ảnh này
  – không có đối tượng nào") thay vì im lặng bỏ qua không tạo file.
- Một file `classes.txt` ở gốc thư mục ảnh, một dòng = một tên lớp, **thứ
  tự dòng chính là class index (0-based)** dùng trong mọi file `.txt`.

### 1.2. Danh sách lớp (15 lớp, thứ tự CỐ ĐỊNH)

File `classes.txt` hiện tại (do labelImg tự sinh từ file
`tool/labelimg_predefined_classes.txt` được truyền vào lúc khởi động) có nội
dung theo đúng thứ tự sau — **tool mới PHẢI dùng đúng thứ tự này khi mở một
thư mục đã có `classes.txt` sẵn**, không được sắp xếp lại, không được đổi
tên lớp, không được chèn lớp mới vào giữa danh sách:

```
0  red_king
1  red_advisor
2  red_elephant
3  red_horse
4  red_cannon
5  red_rook
6  red_pawn
7  black_king
8  black_advisor
9  black_elephant
10 black_horse
11 black_cannon
12 black_rook
13 black_pawn
14 hand
```

`hand` (index 14) là lớp đặc biệt: đánh dấu vùng ảnh bị tay người che (occlusion)
khi đang xếp/chỉnh quân cờ để chụp ảnh — không phải một quân cờ.

Nếu tool mới mở một thư mục **chưa có `classes.txt`** (thư mục hoàn toàn
mới, chưa từng gán nhãn), nó có thể tự sinh `classes.txt` theo đúng danh
sách 15 lớp trên (đây cũng là input tương đương
`tool/labelimg_predefined_classes.txt`, có thể coi là cấu hình mặc định của
tool mới).

### 1.3. Định dạng một dòng nhãn (YOLO bbox, không phải PascalVOC)

```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: số nguyên, index vào `classes.txt` (0-based).
- 4 giá trị còn lại: số thực, **chuẩn hoá [0, 1]** theo chiều rộng/cao ảnh
  gốc (không phải pixel tuyệt đối). `x_center`/`y_center` là tâm box,
  `width`/`height` là kích thước box — đúng công thức YOLOv5/v8, KHÔNG phải
  góc trái-trên/phải-dưới.
- Mỗi box một dòng, phân cách bằng khoảng trắng, không có header/footer.
  Dòng trống được bỏ qua khi đọc.
- Không ghi trường "verified" hay "difficult" nào khác vào file `.txt` này
  — pipeline downstream không đọc các trường đó, thêm vào không sai nhưng
  vô ích, và nếu thêm sai định dạng có thể làm hỏng parser hiện có.

### 1.4. Vì sao mức tương thích này bắt buộc

Script gộp nhãn `tool/unify_scanner_labels.py` (hàm `load_flat_yolo_dataset`)
đọc chính xác cấu trúc trên: với mỗi ảnh, nếu không có `<tên>.txt` thì bỏ
qua hẳn (không tính là "đã xác nhận rỗng"); nếu có, đọc từng dòng, tách
`class_id` rồi tra `classes.txt[class_id]` ra tên lớp. Nếu tool mới ghi sai
thứ tự lớp, đổi định dạng toạ độ, hoặc gộp/tách khác đi, **toàn bộ nhãn đã
gán từ trước qua labelImg (và nhãn mới gán qua tool mới) sẽ bị đọc sai lớp
một cách âm thầm, không có cảnh báo lỗi** — vì file vẫn đúng cú pháp, chỉ
sai ngữ nghĩa. Đây là rủi ro nghiêm trọng nhất của việc thay tool giữa
chừng, nên **acceptance test bắt buộc** (xem mục 8) là: mở lại một thư mục
đã gán nhãn dở bằng labelImg thật, kiểm tra tool mới hiển thị đúng y hệt số
box/lớp/vị trí, và sau khi lưu, file `.txt` phải giữ nguyên với các ảnh
không chỉnh sửa trong phiên đó (không rewrite lại toàn bộ thư mục mỗi lần
lưu — chỉ ghi đè file `.txt` của ảnh đang thao tác).

## 2. Toàn bộ tính năng cần có (parity với labelImg)

Ghi chú: danh sách dưới đây dựa trên hành vi phổ biến của labelImg, đã được
chủ dự án xác nhận là đúng chuẩn — coi đây là baseline tính năng cơ bản bắt
buộc của tool mới.

- **Duyệt thư mục ảnh**: mở một thư mục, liệt kê toàn bộ ảnh `.jpg/.jpeg/.png`
  theo thứ tự tên file. Danh sách phải load nhanh và mượt với 4458 ảnh (xem
  mục 6 — yêu cầu hiệu năng).
- **Danh sách file bên cạnh**: hiện danh sách toàn bộ ảnh trong thư mục,
  phân biệt trực quan ảnh đã gán nhãn (có `.txt`) và chưa gán nhãn, click
  để nhảy nhanh tới ảnh bất kỳ.
- **Vẽ bounding box**: công cụ vẽ hình chữ nhật bằng chuột (kéo-thả), có
  phím tắt bật chế độ vẽ.
- **Chọn / sửa / xoá box**: click chọn 1 box đang có trên ảnh, kéo cạnh/góc
  để resize, kéo giữa để di chuyển, phím Delete để xoá box đang chọn.
- **Gán lớp cho box**: sau khi vẽ, chọn 1 trong 15 lớp (qua danh sách lớp
  định sẵn, không gõ tay tên lớp để tránh lỗi chính tả sinh lớp lạ).
- **Danh sách box của ảnh hiện tại**: panel liệt kê toàn bộ box đã vẽ trên
  ảnh đang mở kèm tên lớp, click một dòng để highlight box tương ứng trên
  ảnh.
- **Điều hướng ảnh**: phím tắt ảnh kế tiếp / ảnh trước, giữ nguyên toàn bộ
  box đang vẽ dở khi chuyển ảnh (hoặc cảnh báo nếu có thay đổi chưa lưu).
- **Lưu**: phím tắt lưu ảnh hiện tại (ghi `<tên>.txt`); nên có tuỳ chọn
  auto-save khi chuyển ảnh để tránh mất công nếu quên bấm lưu.
- **Zoom / fit**: phóng to/thu nhỏ ảnh, fit-to-window, fit-to-width — quan
  trọng vì quân cờ nhỏ so với khung ảnh toàn bàn, cần zoom để vẽ box chính
  xác.
- **Undo/redo** thao tác vẽ/xoá/sửa box trong phiên hiện tại.
- **Copy box**: nhân bản box đang chọn (hữu ích khi nhiều quân cùng lớp có
  kích thước gần giống nhau).
- **Thư mục lưu = thư mục ảnh, không cấu hình khác** (đơn giản hoá có chủ
  đích so với labelImg — labelImg cho đổi "save dir" riêng biệt khỏi thư
  mục ảnh; tool mới **không cần** tính năng đó, luôn lưu `.txt` cạnh ảnh
  gốc, đúng với ràng buộc mục 1).

## 3. Tính năng mới #1 — Gán nhãn nhanh bằng phím tắt 1 phím

> Cập nhật 28/07/2026 (xem mục 9): thiết kế "chord 2 bước tuần tự" mô tả ở
> phiên bản đầu của mục này đã bị **thay thế** bởi thiết kế 1-phím dưới đây
> sau khi dùng thử thực tế cho thấy 2 bước vẫn chậm hơn cần thiết. Bảng
> `Ctrl,<màu>,<vai trò>` cũ không còn hiệu lực.

**Mục tiêu**: gán lớp cho box đang chọn (hoặc box vừa vẽ xong, tự động được
chọn) mà không cần rời tay khỏi bàn phím để mở dropdown/click danh sách lớp,
và nhanh hơn nữa so với chord 2 bước: **chỉ 1 phím** cho mỗi lớp quân cờ.

**Cơ chế**: mỗi vai trò quân là **1 chữ cái**, gõ trực tiếp, không cần giữ
Ctrl:

| Phím | Vai trò |
|---|---|
| `p` | pawn (tốt) |
| `c` | cannon (pháo) |
| `r` | rook (xe) |
| `h` | horse (mã) |
| `e` | elephant (tượng) |
| `a` | advisor (sĩ) |
| `k` | king (tướng) |

**Màu quân do hoa/thường của chữ cái quyết định**:

- Gõ **chữ thường** (không bật Caps Lock) → quân **đen** (`black_<vai trò>`).
- Gõ **chữ HOA** (bật Caps Lock trước, hoặc giữ Shift) → quân **đỏ**
  (`red_<vai trò>`).
- Cơ chế đọc `event.text()` của phím vừa gõ (giá trị đã được hệ điều hành
  dịch theo trạng thái Caps Lock/Shift thực tế), không dùng modifier
  Ctrl/Shift để suy màu — vì Caps Lock không phải là một Qt keyboard
  modifier, chỉ ảnh hưởng tới ký tự sinh ra.
- Sau khi gõ, gán lớp `<màu>_<vai trò>` tương ứng cho box đang chọn, tự động
  lưu thay đổi vào state của ảnh (chưa cần ghi file ngay, theo cơ chế
  save/auto-save chung).
- Lớp `hand` không có màu và không nằm trong 7 chữ cái vai trò ở trên (chữ
  `h` đã dùng cho horse) — giữ nguyên phím tắt riêng **`Ctrl+H`**, bấm ngay
  lập tức, không phân biệt hoa/thường.

**Xung đột phím tắt cần tránh** (đã rà soát khi đổi từ chord sang 1-phím):

- Trước đây `A` = "ảnh trước", `D` = "ảnh sau" (phím tắt điều hướng, không
  giữ modifier). Chữ `a` nay dùng cho advisor (sĩ) nên xung đột trực tiếp
  với phím `A` cũ (Qt/OS không phân biệt được "A do gõ thường" khác "A do
  gõ hoa" ở tầng shortcut nếu vẫn dùng QAction shortcut đơn giản). Do đó
  **điều hướng ảnh trước/sau đã đổi sang phím mũi tên `←`/`→`**, không dùng
  chữ cái nữa, để giải phóng toàn bộ 26 chữ cái cho việc gán lớp và tránh
  mọi xung đột tương lai.
- `W` (bật chế độ vẽ box), `Ctrl+D` (nhân bản), `Ctrl+S/O/Z/Shift+Z/+/-/=/0/9`
  (lưu, mở, undo/redo, zoom...) đều không trùng với 7 chữ cái vai trò hay
  `Ctrl+H`, giữ nguyên không đổi.

**Yêu cầu UX đi kèm**:

- Sau mỗi lần gán lớp thành công, hiện thông báo ngắn ở status bar (ví dụ
  "Đã gán lớp: red_cannon") để người dùng xác nhận ngay đã gõ đúng màu/vai
  trò mong muốn — quan trọng vì lỗi hay gặp nhất là quên bật/tắt Caps Lock.
- Nếu không có box nào đang chọn khi gõ phím, không làm gì (báo nhẹ ở status
  bar), không được tạo box mới từ hư không.
- Sau khi vẽ xong 1 box mới (thả chuột), box đó nên **tự động ở trạng thái
  đang chọn**, để luồng thao tác là: vẽ box → gõ 1 phím gán lớp → vẽ box
  tiếp theo, hoàn toàn không cần chạm chuột vào danh sách lớp.

## 4. Tính năng mới #2 — Tự động khoanh box bằng phát hiện hình tròn (circle detection)

**Bối cảnh**: quân cờ tướng thật là các đĩa hình tròn (khắc/in chữ Hán trên
mặt tròn), nên có thể dùng xử lý ảnh cổ điển (không cần ML) để phát hiện vị
trí quân cờ trên ảnh, tiết kiệm thời gian vẽ tay 4458 ảnh x nhiều quân/ảnh.
Tính năng này **chỉ áp dụng cho ảnh chụp bàn cờ thật** (thư mục
`chessImg` hiện tại) — không áp dụng cho ảnh chụp màn hình bàn cờ digital
(hình vuông/không tròn, không cần tính năng này).

**Yêu cầu — cả 2 chế độ đều phải có, người dùng chọn dùng chế độ nào**:

1. **Chế độ tự động toàn dải (auto-scan)**: chạy Hough Circle Transform
   (hoặc thuật toán tương đương) trên ảnh gốc vừa load, quét một dải bán
   kính hợp lý (có thể suy từ kích thước ảnh, ví dụ 1.5%–6% chiều rộng ảnh),
   trả về danh sách tâm + bán kính các hình tròn tìm được.
2. **Chế độ đo bán kính tham chiếu (radius-guided, khuyến nghị mặc định)**:
   người dùng dùng một công cụ đo nhanh (click tâm rồi kéo ra mép quân cờ
   thật rõ nét trên ảnh, hoặc vẽ 1 box mẫu quanh 1 quân) → tool tính ra bán
   kính pixel `r0` → chạy lại Hough circle chỉ trong khoảng `r0 × (1 ± dung sai)`.
   **Dung sai mặc định ±15%, và bắt buộc có control trên UI (slider hoặc ô
   nhập số, đơn vị %) để người dùng tự chỉnh giá trị này** — thay đổi giá
   trị xong có nút/phím tắt "chạy lại phát hiện" ngay trên dung sai mới,
   không cần đo lại bán kính từ đầu. Vì các ảnh tự chụp khoảng cách/góc máy
   khác nhau giữa các ảnh (không đồng nhất), bán kính tham chiếu nên đo lại
   nhanh mỗi ảnh (thao tác 1 click-kéo), nhưng tool nên nhớ giá trị lần đo
   gần nhất (và dung sai gần nhất) làm gợi ý mặc định để không phải đo lại
   từ đầu nếu ảnh kế tiếp chụp cùng khoảng cách.
3. Với mỗi hình tròn tìm được, tool tự sinh 1 box hình vuông bao quanh (cạnh
   ≈ 2×bán kính, tâm trùng tâm tròn) ở trạng thái **"gợi ý — chưa xác nhận"**
   (khác biệt rõ về màu/kiểu nét với box đã người dùng xác nhận, ví dụ viền
   nét đứt vs nét liền).
4. Người dùng xử lý từng gợi ý: **xác nhận** (chuyển thành box thật, sau đó
   gán lớp bằng chord ở mục 3), **xoá** (false positive — vòng tròn không
   phải quân cờ, ví dụ giao điểm bàn cờ, nút áo, đồ vật tròn khác), hoặc để
   nguyên rồi tự vẽ tay các quân bị bỏ sót (quân bị che khuất một phần, méo
   hình do góc chụp, sát viền ảnh…).
5. **Không được tự động ghi các box gợi ý chưa xác nhận vào file `.txt`
   khi lưu** — chỉ box đã người dùng xác nhận (và gán lớp) mới được ghi ra.
   Đây là yêu cầu bắt buộc để tránh rác dữ liệu train từ false positive
   không ai kiểm tra.
6. Nên có action "xoá toàn bộ gợi ý chưa xác nhận của ảnh này" (reset
   nhanh nếu kết quả phát hiện quá tệ trên 1 ảnh cụ thể) và action "chạy lại
   phát hiện" sau khi đổi bán kính tham chiếu.

Không bắt buộc thư viện cụ thể — agent xây dựng tự chọn (OpenCV
`HoughCircles`, scikit-image, hoặc cài đặt riêng), miễn nhẹ, chạy local,
không phụ thuộc mạng/cloud, và đủ nhanh cho thao tác tương tác (khuyến nghị
< 1–2 giây/ảnh trên CPU máy dev thường).

## 5. Tính năng mới #3 — Load/lưu tiến độ trong cùng thư mục, resume không mất dữ liệu

Phần này **phần lớn đã có sẵn miễn phí** nhờ tuân thủ đúng mục 1 (mỗi ảnh
đã gán nhãn có `<tên>.txt` cạnh nó) — copy nguyên thư mục sang máy khác là
copy được cả ảnh và tiến độ, không cần đồng bộ gì thêm, đúng như quy trình
hiện tại đã ghi trong tài liệu dự án.

Yêu cầu **thêm** so với labelImg (để chủ động hơn, không phải tự nhớ đang
dừng ở ảnh nào):

- **Xác định điểm dừng tự động khi mở lại thư mục**: khi mở một thư mục đã
  có nhãn dở, tool tự tìm ảnh **đầu tiên theo thứ tự tên file chưa có
  `<tên>.txt`** và nhảy thẳng tới đó (thay vì luôn mở ảnh đầu danh sách như
  labelImg mặc định, bắt người dùng tự kéo tới chỗ đang dở).
- **File trạng thái phiên làm việc lưu NGAY TRONG thư mục ảnh** (ví dụ
  `.labeling_session.json`), không lưu ở thư mục cấu hình riêng của máy
  (kiểu `AppData`/home dir) — vì quy trình hiện tại là copy nguyên thư mục
  `chessImg` sang máy khác để tiếp tục làm; nếu trạng thái phiên nằm ngoài
  thư mục đó, nó sẽ không đi theo khi chuyển máy, mất lại đúng vấn đề tool
  này được yêu cầu giải quyết. File trạng thái này chỉ nên ghi thông tin
  tiện ích (ảnh đang mở gần nhất, bán kính tham chiếu gần nhất dùng cho mục
  4, v.v.) — **không phải nguồn sự thật cho tiến độ gán nhãn**; nguồn sự
  thật vẫn luôn là việc có/không có file `<tên>.txt` cạnh mỗi ảnh, để nếu
  file trạng thái này bị mất/hỏng/xoá, tool vẫn suy lại đúng điểm dừng từ
  chính danh sách `.txt` đang có.
- Tool phải mở được một thư mục đã gán nhãn dở bằng **labelImg thật**
  (không phải chỉ dữ liệu tool mới tự tạo) và tiếp tục đúng mạch, vì đây là
  tình huống thực tế: một phần đã làm bằng labelImg, phần còn lại chuyển
  sang tool mới.

## 6. Yêu cầu phi chức năng

- **Nhẹ, chạy local, offline** — không gọi API/cloud nào, không upload
  ảnh (giữ đúng lý do ban đầu bỏ Roboflow).
- **Hiệu năng với thư mục lớn**: 4458 ảnh, ~388 MiB. Danh sách file/thumbnail
  phải load lười (lazy) hoặc ảo hoá (virtualized), không được đọc/giải mã
  toàn bộ 4458 ảnh cùng lúc lúc mở thư mục.
- **Không phá hoại dữ liệu**: không bao giờ ghi đè `classes.txt` theo thứ
  tự khác thứ tự đã có sẵn khi mở một thư mục đã tồn tại `classes.txt`; khi
  lưu một ảnh, chỉ ghi đè đúng file `.txt` của ảnh đó, không đụng file khác.
- **Stack đã chốt: Python 3.12 + PySide6 (GUI) + OpenCV-Python (xử lý ảnh /
  circle detection) + NumPy/Pillow**. Lý do chọn: máy dev đã có sẵn
  Python 3.12.10 và pip từ trước, dùng để cài `labelImg`, `torch`,
  `onnxruntime` cho các bước khác của Phase 5 (xem
  `docs/phase5_scanner_model_independence.md` mục "Môi trường cần thiết") —
  thêm `pyside6`/`opencv-python` qua pip không tốn công dựng toolchain mới,
  và giữ đúng tinh thần dùng công cụ nội bộ ngắn hạn, không đáng đầu tư một
  toolchain biên dịch C++/Qt riêng. **PySide6** là binding Qt chính thức
  của Qt Company, license **LGPL-3.0** (khác PyQt5 hiện là GPL/thương mại
  dual-license) — dynamic-link đúng cách theo yêu cầu LGPL thì không kéo
  theo nghĩa vụ copyleft cho code ứng dụng, phù hợp tinh thần né GPL/AGPL
  của Phase 5. `opencv-python` license Apache-2.0.
- **License thư viện phụ trợ khác**: nếu cần thêm thư viện ngoài danh sách
  trên, giữ nguyên tinh thần né GPL/AGPL — ưu tiên MIT/BSD/Apache/LGPL
  (dynamic-link đúng cách).
- **Windows là nền tảng chính** (máy dev dùng Windows) — đảm bảo chạy tốt
  trên Windows trước tiên, đa nền tảng là cộng thêm chứ không bắt buộc.

## 7. Ngoài phạm vi (out of scope)

- Không cần tự động phát hiện 4 góc bàn cờ hay ánh xạ giao điểm — việc này
  đã có sẵn, chạy tự động ở bước xử lý sau (`HoughBoardCornerDetector` +
  `BoardPerspectiveRectifier`, ngoài phạm vi tool đánh nhãn này).
  Tool đánh nhãn chỉ sinh ra box quân cờ trên ảnh gốc (chưa rectify).
- Không cần hỗ trợ nhiều định dạng export khác (PascalVOC XML, COCO JSON,
  CreateML…) — chỉ cần đúng 1 định dạng YOLO `.txt` phẳng như mục 1.
- Không cần tính năng "đổi thư mục lưu khác thư mục ảnh".
- Không cần upload/sync cloud dưới bất kỳ hình thức nào.
- Không cần train hay chạy model AI trong tool này — phát hiện hình tròn ở
  mục 4 là xử lý ảnh cổ điển (Hough), không phải suy luận model.

## 8. Tiêu chí nghiệm thu (acceptance criteria)

1. **Test tương thích ngược (bắt buộc, quan trọng nhất)**: chuẩn bị một thư
   mục con nhỏ (5–10 ảnh) đã gán nhãn thật bằng labelImg (có `classes.txt`
   + vài `.txt` theo đúng định dạng mục 1). Mở bằng tool mới → phải hiển thị
   đúng y hệt số lượng box, lớp, vị trí (sai số làm tròn chấp nhận được ở
   mức chuyển đổi normalize/pixel qua lại, không được lệch lớp). Sửa 1 ảnh,
   lưu lại → các ảnh không đụng tới phải giữ nguyên byte-for-byte (hoặc ít
   nhất giữ nguyên nội dung logic: cùng box, cùng lớp, cùng thứ tự
   class index).
2. **Test round-trip tool mới**: gán nhãn một ảnh mới hoàn toàn bằng tool
   mới, lưu lại, rồi chạy thử `tool/unify_scanner_labels.py --labelimg
   NAME=<thư mục test> --skip-chessai-data` (script thật của repo) — phải
   chạy thành công, không rơi vào nhóm `unmapped_labels` trong
   `summary.json` sinh ra.
3. **Test resume**: gán nhãn dở một thư mục (dừng ở ảnh thứ N theo alphabet),
   đóng tool, mở lại đúng thư mục đó → tool phải tự nhảy tới ảnh N+1 (ảnh
   đầu tiên chưa có `.txt`), không yêu cầu người dùng tự kéo tìm.
4. **Test phím tắt gán lớp**: vẽ 1 box, gõ `c` (chữ thường) → box phải được
   gán đúng lớp `black_cannon`; bật Caps Lock rồi gõ `C` (chữ hoa) → phải
   đổi thành `red_cannon`; bấm `Ctrl+H` → phải gán `hand`.
5. **Test circle-assist**: trên 1 ảnh có quân cờ tròn rõ nét, chạy chế độ
   radius-guided sau khi đo 1 quân mẫu → phải sinh ra ít nhất một vài gợi ý
   box trùng vị trí quân thật (không cần chính xác 100%, đây là công cụ hỗ
   trợ); xác nhận 1 gợi ý và lưu → box đó phải xuất hiện trong `.txt` với
   toạ độ hợp lệ trong `[0,1]`; các gợi ý không xác nhận không được xuất
   hiện trong `.txt`.
6. **Test hiệu năng**: mở thư mục đầy đủ (hoặc thư mục giả lập ~4000+ ảnh)
   không bị treo/lag nghiêm trọng khi load danh sách file.

## 9. Quyết định đã chốt (xác nhận bởi chủ dự án, 28/07/2026)

Không còn câu hỏi mở — mọi điểm dưới đây đã được chủ dự án xác nhận trực
tiếp, agent xây dựng có thể triển khai thẳng theo tài liệu này:

- **Mục 2** (danh sách tính năng kế thừa labelImg): xác nhận đúng chuẩn
  hành vi phổ biến của người dùng, coi là baseline bắt buộc, không thu hẹp.
- **Stack xây dựng**: Python 3.12 + PySide6 + OpenCV-Python (xem mục 6) —
  chủ dự án giao toàn quyền chọn giữa Python hoặc C++/Qt, đã chọn Python vì
  môi trường pip đã sẵn sàng trên máy dev từ các bước Phase 5 khác.
- **Phím tắt gán lớp ở mục 3**: ban đầu chốt chord 2 bước `Ctrl+<màu>+<vai
  trò>`; **trong cùng ngày, sau khi dùng thử**, chủ dự án yêu cầu đổi sang
  1 phím duy nhất mỗi vai trò (`p/c/r/h/e/a/k`), màu suy ra từ hoa/thường
  (Caps Lock/Shift) thay vì bước riêng — xem bản mới nhất của mục 3. Đây là
  quyết định **thay thế**, không phải bổ sung; bảng chord cũ không còn áp
  dụng. Kèm theo đó, phím điều hướng ảnh trước/sau đổi từ chữ `A`/`D` sang
  mũi tên `←`/`→` để tránh xung đột với chữ `a` (advisor).
- **Dung sai bán kính** ở chế độ radius-guided (mục 4): mặc định **±15%**,
  bắt buộc có control trên UI để người dùng tự tinh chỉnh theo từng
  ảnh/lô ảnh.
- **Danh sách 15 lớp** ở mục 1.2: xác nhận là ràng buộc cứng, không thêm
  lớp nào khác (không tách quân lật/nghiêng, không thêm lớp viền bàn cờ...).
