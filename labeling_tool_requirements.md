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
  ảnh. (Cập nhật 29/07/2026: panel tách thành **2 cột cạnh nhau** — cột trái
  "Đen / chưa gán" (quân đen + box chưa gán lớp), cột phải "Đỏ / hand" (quân
  đỏ + nhãn `hand`); mỗi cột tự sắp theo thứ tự cố định `p, r, c, h, e, a, k`
  (trái) / `P, R, C, H, E, A, K` (phải), phần "leftover" của mỗi cột
  (chưa-gán-lớp / hand) luôn ở dưới cùng cột đó — xem
  `panels._BLACK_COLUMN_ORDER` / `_RED_COLUMN_ORDER`. Chỉ 1 trong 2 cột được
  highlight tại một thời điểm, khớp với box đang chọn trên canvas.

  Trên ảnh, mỗi box cũng có một nhãn chữ nhỏ in đậm vẽ **bên trong box, góc
  trên-trái** — không có nền, 1 chữ cái vai trò viết thường (hoặc chữ
  "hand"), màu chữ đen hoặc đỏ theo màu quân; box chưa gán lớp không vẽ gì cả
  (xem lại 29/07/2026, 2 lần: bản đầu đặt nhãn ở tâm box kèm nền trắng, che
  mất quân cờ; bản 2 chuyển ra ngoài góc box nhưng vẫn còn nền trắng và to;
  bản hiện tại bỏ hẳn nền, thu nhỏ, đưa vào trong góc box). Vị trí luôn tính
  theo góc box (không kẹp theo biên ảnh), nên quân sát mép bàn/mép ảnh vẫn an
  toàn — không bao giờ crash (có test phủ trường hợp box dán sát góc (0,0)
  và box 1x1 pixel).
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

**Cập nhật 29/07/2026 — nhớ thư mục đang làm việc giữa các lần chạy tool**:
khác với file trạng thái phiên ở trên (theo từng thư mục ảnh, đi theo thư
mục khi copy sang máy khác), đây là một preference **theo máy** (đường dẫn
thư mục chỉ có ý nghĩa trên chính máy đó) — lưu ở `QSettings` (registry
Windows, khoá `ChessLabeler/XiangqiLabeler`), không phải trong thư mục
ảnh. Mỗi lần mở thư mục thành công (dù qua dialog hay tự động resume), tool
ghi lại đường dẫn; mỗi lần khởi động, tool đọc lại và tự mở đúng thư mục đó
nếu vẫn còn tồn tại (bỏ qua lặng lẽ nếu thư mục đã bị xoá/di chuyển).

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

## 10. Cập nhật bắt buộc — metadata gold-set, 4 góc bàn cờ, FEN người xác minh và điều kiện chụp

> **Quyết định mới:** mục này thay thế phần “không cần người dùng đánh dấu 4
> góc” ở mục 0 và phần “không cần tự động phát hiện 4 góc” ở mục 7, nhưng chỉ
> theo nghĩa **tool phải cho phép người gán nhãn ghi ground truth bằng tay**.
> Tool không phải tự suy luận góc. Ground truth này là cần thiết để benchmark
> độc lập thuật toán OpenCV/Hough, phép rectification và full-board FEN của
> scanner Phase 5.

### 10.1. Quyết định định dạng: không gộp metadata vào `<stem>.txt`

Tool mới hoàn toàn có thể tự đọc một file `.txt` custom, nhưng **không được
gộp metadata vào file YOLO `<stem>.txt`**. File này phải tiếp tục chỉ chứa các
dòng YOLO năm trường:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Lý do là dữ liệu cũ từ labelImg, tool hiện tại và các script downstream đều
đọc nó như YOLO thuần. Thêm JSON, header, comment hoặc một cột thứ sáu có thể
làm hỏng parser, bị ghi đè khi mở/lưu lại, hoặc tệ hơn là gây silent corruption.
Tính tương thích ngược ở mục 1 vẫn là ràng buộc cứng.

Mỗi ảnh được phép có thêm một **sidecar JSON** cùng thư mục:

```text
0105.jpg          # ảnh gốc, không bị tool sửa
0105.txt          # YOLO bbox thuần cho 15 lớp hiện có
0105.meta.json    # metadata board-level mô tả trong mục này
```

Quy tắc bắt buộc:

- Tên sidecar phải đúng `<stem>.meta.json`, với `<stem>` đúng tên ảnh, phân
  biệt ảnh theo quy tắc hệ điều hành nhưng không phụ thuộc phần mở rộng ảnh.
- Tool vẫn phải mở/lưu bình thường mọi ảnh cũ chỉ có `.txt`; sidecar là tùy
  chọn cho đến khi người dùng lưu metadata lần đầu.
- Danh sách ảnh phải bỏ qua `.meta.json`, `.labeling_session.json`,
  `classes.txt` và mọi file không phải ảnh.
- `.txt` là source of truth cho YOLO bbox; `.meta.json` là source of truth cho
  góc, FEN, điều kiện ảnh và trạng thái review. Không nhân bản bbox sang JSON.
- JSON dùng UTF-8, JSON chuẩn không có comment, field name tiếng Anh ổn định,
  không ghi đường dẫn tuyệt đối của máy vào file.
- Khi Save, ghi ra file tạm cùng thư mục rồi rename/replace nguyên tử. Không
  được để một lần crash tạo JSON nửa chừng hoặc phá file `.txt` cũ.
- Nếu JSON lỗi cú pháp/schema, tool phải cảnh báo rõ, mở ảnh/bbox ở chế độ an
  toàn và **không tự ghi đè** metadata lỗi cho tới khi người dùng chọn sửa/lưu.

### 10.2. Schema `meta.json` phiên bản 1

Đây là schema chuẩn phải được tool đọc/ghi. Field chưa biết dùng `null` hoặc
`"unknown"`, không suy đoán để làm đẹp dữ liệu.

```json
{
  "schema_version": 1,
  "image": {
    "filename": "0105.jpg",
    "width_px": 1920,
    "height_px": 1080
  },
  "board": {
    "corners_px": {
      "top_left": { "x": 233.0, "y": 91.5 },
      "top_right": { "x": 1680.5, "y": 110.0 },
      "bottom_right": { "x": 1748.0, "y": 997.5 },
      "bottom_left": { "x": 176.0, "y": 971.0 }
    },
    "corners_status": "human_verified",
    "image_orientation": "red_at_bottom",
    "position_complete": true,
    "board_fen": "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR",
    "side_to_move": null,
    "full_fen": null,
    "fen_status": "human_verified"
  },
  "capture": {
    "lighting": "even",
    "shadow": "mild",
    "glare": "none",
    "perspective": "mild",
    "board_material": "wood",
    "board_fill": "large",
    "distance": "medium",
    "blur": "none",
    "occlusion": "none",
    "occlusion_severity": "none",
    "environment": "indoor",
    "device_model": "Redmi Note 11",
    "capture_group": null
  },
  "review": {
    "status": "self_checked",
    "fen_verified": true,
    "corners_verified": true,
    "exclude_from_gold": false,
    "exclusion_reason": null,
    "notes": ""
  }
}
```

Quy ước dữ liệu:

- `image.filename`, `image.width_px`, `image.height_px` là fingerprint tối
  thiểu để tool phát hiện sidecar bị ghép nhầm sang ảnh khác hoặc ảnh đã bị
  resize. Không lưu đường dẫn tuyệt đối.
- `corners_px` lưu tọa độ pixel của **ảnh gốc**, với `x` tăng sang phải và `y`
  tăng xuống dưới. Đây là hệ tọa độ canonical duy nhất vì benchmark geometry
  cần đo trực tiếp sai số pixel/reprojection error. Không lưu đồng thời pixel
  và normalized trong JSON để tránh hai bản bị lệch; script export có thể suy
  ra `x / image.width_px`, `y / image.height_px` khi thực sự cần normalized.
- Bốn điểm là **bốn giao điểm lưới ngoài cùng của bàn 9×10**, không phải mép
  gỗ/nhựa, khung trang trí hoặc vùng ảnh bao quanh bàn.
- Thứ tự không đổi: `top_left`, `top_right`, `bottom_right`, `bottom_left`,
  theo chiều kim đồng hồ trong ảnh gốc.
- `corners_status` là một trong: `unmarked`, `partial`, `auto_suggested`,
  `human_marked`, `human_verified`, `not_applicable`. Phiên bản đầu không cần
  tự đề xuất góc, nhưng schema giữ `auto_suggested` cho tương lai.
- `image_orientation` là hướng của bàn trong ảnh: `red_at_bottom`,
  `red_at_top`, `red_at_left`, `red_at_right`, `unknown`. Nó mô tả ảnh, không
  phải trạng thái lượt đi.
- `board_fen` là **piece-placement field** của Xiangqi FEN, từ rank trên ảnh
  logic xuống rank dưới, dùng bộ ký hiệu `K/A/B/R/C/N/P` cho Đỏ và chữ thường
  cho Đen. Đây là field chính để đo exact-match vị trí bàn cờ.
- `side_to_move` chỉ nhận `red`, `black` hoặc `null`. Ảnh tĩnh thường không
  cho biết lượt đi, vì vậy mặc định phải là `null`, không được tự đặt Đỏ.
- `full_fen` chỉ được điền khi người dùng thực sự biết lượt đi; khi có giá trị,
  nó phải khớp `board_fen` và `side_to_move`. Nếu chưa biết lượt đi, giữ
  `full_fen: null`.
- `fen_status` nhận: `not_started`, `human_marked`, `human_verified`,
  `not_applicable`; chỉ trạng thái `human_verified` mới được coi là ground
  truth FEN cho benchmark.
- `position_complete: false` nếu bàn bị cắt mất, quân che quá nhiều hoặc người
  gán nhãn không thể xác minh chính xác toàn thế cờ. Ảnh đó có thể vẫn dùng để
  train detector/classifier nhưng không được dùng tính full-board FEN
  exact-match.
- `review.status` nhận: `unreviewed`, `annotated`, `self_checked`,
  `gold_verified`, `needs_review`. Chỉ `gold_verified` mới được đưa vào gold
  holdout chính thức.
- Không lưu định danh cá nhân, GPS, serial thiết bị hoặc EXIF thô. `device_model`
  chỉ là tên model/family thiết bị phục vụ phân tích lỗi theo thiết bị.

### 10.3. Nhập và hiển thị 4 góc bằng bàn phím

Tool phải thêm chế độ **Board corners** trên canvas ảnh gốc. Đây là thao tác
nhanh, không thay thế công cụ vẽ bbox.

#### Phím tắt và hành vi

Khi canvas ảnh đang có focus, không có text field/dropdown/popup đang nhận
phím:

| Phím | Hành vi |
|---|---|
| `1` | Lấy vị trí chuột hiện tại làm `top_left` |
| `2` | Lấy vị trí chuột hiện tại làm `top_right` |
| `3` | Lấy vị trí chuột hiện tại làm `bottom_right` |
| `4` | Lấy vị trí chuột hiện tại làm `bottom_left` |
| `0` | Xóa cả bốn góc sau hộp xác nhận ngắn |

- Mỗi phím chỉ thay thế đúng góc tương ứng; cho phép bấm lại để chỉnh từng góc
  mà không bắt buộc nhập lại bốn góc.
- Hành động đặt/xóa/thay thế góc phải đi vào undo/redo stack riêng của ảnh.
- Sau thao tác, status bar báo tên góc và tọa độ pixel; ví dụ
  `Đã đặt top_left: (128, 91)`.
- Nếu canvas không có focus hoặc người dùng đang nhập FEN/chọn dropdown, các
  phím `1`–`4` phải đi theo control đang focus, không được vô tình ghi góc.

#### Overlay trực quan

- Mỗi góc đã đặt được vẽ thành **một hình tròn fill nhỏ bán kính 3 px màn
  hình** quanh đúng điểm. Bán kính hiển thị giữ 2–3 px khi zoom để không che
  lưới, nhưng tâm phải bám chính xác vào tọa độ ảnh khi pan/zoom/fit.
- Dùng bốn màu cố định dễ phân biệt: top-left xanh lá, top-right xanh dương,
  bottom-right cam, bottom-left tím. Tooltip hoặc panel liệt kê tên/giá trị;
  không cần chữ lớn che ảnh.
- Khi rê chuột lên marker, highlight nhẹ marker tương ứng và hiển thị tên góc.
- Marker phải nằm trên bbox overlay nhưng không được cản thao tác vẽ/chọn box.

#### Kiểm tra trước khi đánh dấu `human_verified`

Tool phải kiểm tra và hiển thị lỗi/cảnh báo, không âm thầm nhận dữ liệu xấu:

- đủ bốn điểm, trong biên ảnh;
- không trùng nhau;
- polygon lồi, không tự cắt kiểu bow-tie;
- thứ tự theo chiều kim đồng hồ như schema;
- diện tích tối thiểu hợp lý, mặc định ít nhất 1% diện tích ảnh.

`human_marked` có thể lưu khi còn cảnh báo để người dùng quay lại xử lý;
`human_verified` chỉ có thể chọn khi các kiểm tra hình học trên pass. Kiểm tra
này không chứng minh góc đúng với bàn thật, nhưng chặn lỗi nhập liệu rõ ràng.

### 10.4. Board editor số hóa và nguồn FEN do con người xác minh

#### Bố cục

- Màn hình chính vẫn đặt ảnh bàn cờ thật làm vùng làm việc trung tâm/trái.
- Thêm một **dock/panel ở bên phải** tên `Bàn cờ số hóa & FEN`, có thể cuộn khi
  màn hình thấp nhưng không được che ảnh gốc.
- Panel hiển thị bàn Xiangqi 9 cột × 10 hàng. Mặc định hiển thị Đen ở phía trên,
  Đỏ ở phía dưới; có nút `Lật hiển thị` chỉ đổi hướng nhìn, không đổi state hay
  FEN logic.
- Khi mở ảnh chưa có metadata, board hiển thị thế xuất phát đủ 32 quân như một
  scaffold để người dùng chỉnh nhanh. Tuy nhiên `board_fen` và `fen_status`
  phải vẫn là `null`/`not_started` cho đến khi người dùng chủ động lưu/xác nhận
  board; tool tuyệt đối không được tự ghi FEN thế xuất phát cho ảnh mới.
- Khi metadata đã có `board_fen`, khởi tạo từ chính FEN đó; không reset lại thế
  xuất phát trừ khi người dùng chủ động chọn.

#### Tương tác bắt buộc

- Người dùng drag một quân từ giao điểm nguồn sang giao điểm đích. Quân phải
  snap đúng một trong 90 giao điểm.
- Nếu đích có quân, thao tác là bắt quân: quân đích bị xóa, quân nguồn chuyển
  tới đích.
- Click quân rồi click giao điểm đích là phương án tương đương cho touchpad.
- Chọn quân và bấm `Delete`/`Backspace` xóa quân; cần thiết để mô tả thế cờ có
  quân đã bị bắt.
- Có palette 14 loại quân (`red_*`, `black_*`, không có `hand`) để thêm quân:
  chọn loại trong palette rồi click một giao điểm trống. Điều này phục vụ thế
  cờ dựng tay/nhập sai trước đó; không giới hạn theo lịch sử nước đi.
- Có nút `Thế xuất phát`, `Bàn trống`, `Undo`, `Redo`, `Lật hiển thị` và `Sao
  chép board FEN`. Các nút reset phải hỏi xác nhận nếu state đã thay đổi.
- Board editor là công cụ annotation, **không bắt buộc luật đi hợp lệ từng
  nước**. Người dùng được phép di chuyển/thêm/xóa trực tiếp để khớp ảnh.

#### FEN, validation và lưu

- `board_fen` được sinh lại tức thời từ state board sau mọi chỉnh sửa; không
  phải trường người dùng gõ tự do trong luồng chính.
- UI hiển thị field read-only, nút Copy, và trạng thái validation.
- Có thể bổ sung action phụ `Dán board FEN` sau này, nhưng phải validate và
  cập nhật toàn bộ board trước khi áp dụng; không phải yêu cầu tối thiểu.
- Áp dụng Xiangqi board validator: số tướng, cung tướng/sĩ, tượng qua sông,
  hai tướng đối mặt, số quân tối đa. Hiển thị từng lỗi dễ hiểu.
- Validator lỗi **không được tự sửa board**. Người dùng có thể lưu state
  `position_complete: false`/`needs_review` để không chặn công việc, nhưng
  không được đánh dấu `fen_verified: true` hay `gold_verified` khi còn lỗi.
- Bàn hợp lệ không tự động chứng minh FEN đúng với ảnh. Người dùng phải đối
  chiếu ảnh–board rồi mới bật `fen_verified`.
- Khi người dùng biết lượt đi, có dropdown `side_to_move`: `Không biết`,
  `Đỏ`, `Đen`. Chỉ khi chọn Đỏ/Đen mới sinh `full_fen`; mặc định là `null`.

### 10.5. Dropdown metadata điều kiện ảnh — tối ưu tốc độ gán nhãn

Thêm panel `Điều kiện chụp` dưới hoặc cạnh board editor. Mọi field là
dropdown/combobox có keyboard navigation; không yêu cầu gõ tự do trừ
`device_model` và `notes`.

| Field | Giá trị bắt buộc |
|---|---|
| `lighting` | `unknown`, `very_dark`, `dim`, `even`, `bright`, `mixed` |
| `shadow` | `unknown`, `none`, `mild`, `strong` |
| `glare` | `unknown`, `none`, `mild`, `strong` |
| `perspective` | `unknown`, `frontal`, `mild`, `strong`, `extreme` |
| `board_material` | `unknown`, `wood`, `plastic`, `paper`, `stone`, `other` |
| `board_fill` | `unknown`, `tiny`, `small`, `medium`, `large`, `very_large` |
| `distance` | `unknown`, `near`, `medium`, `far` |
| `blur` | `unknown`, `none`, `mild`, `strong` |
| `occlusion` | `unknown`, `none`, `hand`, `piece`, `object`, `multiple` |
| `occlusion_severity` | `unknown`, `none`, `mild`, `strong` |
| `environment` | `unknown`, `indoor`, `outdoor`, `mixed` |
| `device_model` | combobox editable, `unknown` mặc định, nhớ các giá trị gần đây |
| `capture_group` | combobox editable, `null` mặc định; dùng để nhóm ảnh cùng phiên/bối cảnh |

Yêu cầu UX:

- Mặc định mọi tag là `unknown`; không tự điền “normal” hoặc “none”.
- `occlusion = none` tự đề xuất `occlusion_severity = none`; nếu đổi
  `occlusion` sang giá trị khác, severity trở lại `unknown` để người dùng xác
  nhận.
- `device_model` và `capture_group` phải có danh sách recent values từ
  `.labeling_session.json`, cho phép chọn bằng vài phím và không lưu ID cá nhân.
- Có nút `Áp dụng điều kiện hiện tại cho ảnh tiếp theo` và tùy chọn áp dụng cho
  dải ảnh người dùng chọn. Hành động batch phải hiện số ảnh bị đổi và có xác
  nhận; không được ghi đè field đã khác `unknown` nếu người dùng không chọn
  `Ghi đè giá trị đã có`.
- Panel phải hiện một badge `Metadata chưa đủ` khi chưa đủ góc/FEN/tags để
  người dùng biết ảnh này chưa thể là gold sample, nhưng vẫn cho phép lưu YOLO
  bbox độc lập.

### 10.6. Luồng thao tác đề xuất cho một ảnh

1. Mở ảnh; tool đọc `.txt` YOLO và `.meta.json` nếu có.
2. Kiểm tra/sửa bbox quân cờ như workflow hiện có.
3. Đặt bốn góc bằng `1` → `2` → `3` → `4`; kiểm tra overlay trên lưới thật.
4. Điều chỉnh board số hóa cho khớp ảnh thật, bao gồm các quân đã bị bắt.
5. Kiểm tra board FEN, chọn `position_complete`, và chỉ xác minh FEN sau khi
   đối chiếu ảnh–board.
6. Chọn nhanh điều kiện chụp bằng dropdown; dùng `unknown` nếu không chắc.
7. Chọn `review.status`, lưu. Tool ghi `.txt` khi bbox thay đổi và ghi
   `.meta.json` khi metadata thay đổi; không rewrite file không thay đổi.

### 10.7. Tương thích, migration và phạm vi downstream

- Các ảnh cũ đã gán bằng labelImg không có `.meta.json` phải mở bình thường.
  Tool hiển thị metadata ở trạng thái chưa hoàn thành, không tự sinh hay tự
  coi góc/FEN từ thuật toán hiện tại là ground truth.
- Với 74 ảnh hiện có, ưu tiên bổ sung bốn góc, orientation và FEN do người
  kiểm tra trước; đây là gold-set ban đầu có giá trị hơn việc tăng nhanh số
  lượng ảnh không có ground truth board-level.
- `tool/unify_scanner_labels.py` hiện chỉ đọc YOLO và sẽ không tự đọc sidecar.
  Đây là hành vi đúng ở giai đoạn này: sidecar không được làm thay đổi train
  pipeline cũ. Một tool benchmark/ingest metadata riêng sẽ join `<stem>.txt`
  với `<stem>.meta.json` ở phase tiếp theo.
- Khi metadata thiếu, sai schema hoặc không khớp stem ảnh, tool phải báo trong
  sidebar/log; không bỏ qua im lặng khi người dùng đang cố tạo gold sample.

### 10.8. Acceptance criteria bổ sung

1. **YOLO bất biến:** mở/lưu metadata cho ảnh chỉ có `.txt` không được chèn
   một byte metadata nào vào `.txt`; `unify_scanner_labels.py` vẫn đọc ảnh đó
   thành công như trước.
2. **Round-trip metadata:** lưu rồi đóng/mở lại ảnh phải khôi phục chính xác 4
   góc, FEN board, side-to-move, tags, review status và notes.
3. **Canvas mapping:** ở các mức zoom fit, 100%, 300% và sau pan, bấm `1`–`4`
   phải lưu cùng tọa độ ảnh trong sai số tối đa 1 pixel ảnh; marker phải bám
   đúng điểm khi đổi zoom.
4. **Corner validation:** 4 điểm hợp lệ được chấp nhận; thiếu điểm, trùng điểm,
   bow-tie và polygon quá nhỏ bị cảnh báo; không thể gắn `human_verified` khi
   kiểm tra hình học fail.
5. **Board editor/FEN:** thế xuất phát sinh đúng `board_fen` chuẩn; drag, bắt,
   thêm, xóa, undo/redo và reset cập nhật FEN đúng; lật hiển thị không đổi FEN
   logic; side-to-move `null` không sinh `full_fen`.
6. **Validation/review:** board bất hợp lệ có thông báo cụ thể và không thể
   được đánh dấu `fen_verified` hoặc `gold_verified`; vẫn có thể lưu
   `needs_review` để không mất công annotation.
7. **Tag UX:** tất cả dropdown có đúng taxonomy, `unknown` là default, recent
   device/capture group hoạt động, và batch apply không vô tình ghi đè metadata
   có ý nghĩa.
8. **Không phá resume:** `.labeling_session.json` chỉ giữ preference/session;
   khi copy nguyên thư mục ảnh sang máy khác, `.txt` và `.meta.json` đi cùng,
   tool vẫn nhận diện đúng tiến độ và metadata.

### 10.9. Không triển khai trong yêu cầu này

Tài liệu này chỉ bổ sung yêu cầu sản phẩm/UX/schema cho tool. Không yêu cầu ở
lần này sửa scanner app, sửa `unify_scanner_labels.py`, train lại model, hoặc
tự động suy luận góc/FEN. Những bước downstream chỉ được làm sau khi sidecar
schema và dữ liệu gold-set đã được tạo, review và benchmark riêng.

### 10.10. Trạng thái triển khai hiện tại (tool custom)

Các yêu cầu của mục 10 đã được triển khai trong tool tại thời điểm cập nhật tài
liệu này:

- `chess_labeler/metadata.py`: schema sidecar v1, kiểm tra fingerprint ảnh,
  kiểm tra hình học bốn góc, đọc an toàn và ghi nguyên tử `<stem>.meta.json`.
  Module này không đọc/ghi `.txt` YOLO.
- `chess_labeler/canvas.py`: hotkey `1`–`4`/`0`, marker overlay bám tọa độ pixel
  ảnh, tooltip, và undo/redo góc riêng ở cửa sổ chính.
- `chess_labeler/board_editor.py`: board Xiangqi 9×10, FEN board, drag/click,
  bắt quân, palette, delete, lật hiển thị, undo/redo, copy FEN và validator
  cấu trúc.
- `chess_labeler/metadata_panel.py`: một form không tab con/không cuộn, hướng
  ảnh, trạng thái có/không của 4 góc và FEN, taxonomy điều kiện chụp tiếng Việt
  (vẫn lưu mã schema tiếng Anh) và batch-apply có xác nhận.
- `chess_labeler/main_window.py`: chỉ ghi `.txt` khi bbox thay đổi, chỉ ghi
  `.meta.json` khi metadata thay đổi; metadata lỗi không bị ghi đè tự động;
  có thao tác xóa ảnh an toàn bằng `Shift+Delete`.

Lệnh kiểm thử đầy đủ:

```powershell
python -m compileall -q chess_labeler
python -m pytest -q
```

### 10.11. Cập nhật UX tối giản cho annotator (29/07/2026)

Phần này **thay thế mọi yêu cầu mâu thuẫn** trước đó trong mục 10 về các tab
metadata, dropdown trạng thái góc/FEN, lượt đi, review, thiết bị, nhóm chụp
và loại khỏi gold set.

#### Một form duy nhất, không cuộn

- Khu vực metadata chỉ có **một form gọn**, không còn ba tab `Cần hoàn tất`,
  `Điều kiện chụp`, `Bổ sung & review` và không dùng thanh cuộn nội bộ.
- Form hiển thị đủ các trường ngay trong dock phải ở kích thước cửa sổ làm
  việc chuẩn. Các điều kiện chụp nằm trong lưới hai cột.
- Giữ `Ghi chú (tùy chọn)` ngắn, không gọi là review.

#### 4 góc bàn cờ và FEN: chỉ có/không có

- Hướng dẫn luôn hiển thị trong form: đưa chuột lên **bốn giao điểm lưới ngoài
  cùng** của ảnh (không phải mép khung bàn), bấm `1` = trên-trái, `2` =
  trên-phải, `3` = dưới-phải, `4` = dưới-trái; `0` xóa bốn góc sau xác nhận.
  Canvas phải có focus; click lại vào ảnh nếu đang chọn dropdown hoặc nhập FEN.
- Toolbar cũng có nút `Đánh dấu góc (1–4)` để đưa focus về canvas và nhắc lại
  thứ tự phím, nên thao tác này không còn bị ẩn trong phím tắt.
- Không còn dropdown `corners_status`. UI chỉ hiển thị `Đã đánh dấu (4/4)` hoặc
  `Chưa đủ (n/4)` dựa trực tiếp trên các tọa độ góc. Lỗi hình học vẫn được báo
  riêng để chặn dữ liệu sai rõ ràng.
- Không còn dropdown `fen_status`, checkbox xác minh hay `position_complete`.
  UI chỉ hiển thị `Đã có FEN` khi `board_fen` có giá trị, ngược lại là `Chưa có
  FEN`; validator bàn cờ vẫn báo lỗi riêng.
- Tab ngoài được đặt tên `Bàn cờ & FEN`. Nút `Mở FEN` dẫn trực tiếp tới tab;
  `Xác nhận FEN` dùng cả thế cờ đầu đang hiển thị làm ground truth; `Bỏ FEN`
  đưa ảnh về trạng thái chưa có FEN sau xác nhận.
- Không nhập `side_to_move`. Với FEN mới/sửa/xác nhận, tool ghi
  `side_to_move: null` và `full_fen: null`.

#### Điều kiện chụp và tương thích sidecar

- Mọi dropdown điều kiện chụp hiển thị tiếng Việt dễ hiểu. JSON vẫn lưu các mã
  tiếng Anh ổn định (`even`, `indoor`, …) của schema v1 để không làm hỏng các
  script downstream.
- Không còn thu thập `device_model` hoặc `capture_group` trong UI. Giá trị v1
  cũ được giữ khi mở/lưu sidecar để tương thích ngược; mẫu mới dùng mặc định
  `unknown` và `null`.
- Không còn UI `review`, `Loại khỏi gold set` hay lý do loại. Các trường review
  v1 cũ được giữ để sidecar cũ vẫn mở được; trạng thái kỹ thuật về góc/FEN được
  tool tự suy diễn nội bộ từ dữ liệu và validation, không yêu cầu annotator chọn.
- Benchmark Phase 5 phải dùng 4 góc đủ/hợp lệ, `board_fen` có giá trị và board
  không có lỗi. Nếu cần một quy trình QA hai người hoặc gold-holdout nghiêm
  ngặt hơn, quy trình đó sẽ được bổ sung riêng thay vì làm nặng giao diện gán
  nhãn ban đầu.

#### Xóa ảnh khỏi tập dữ liệu

- `Shift+Delete` (hoặc nút `Xóa ảnh`) hỏi xác nhận rồi chuyển ảnh hiện tại,
  `<stem>.txt` và `<stem>.meta.json` đang tồn tại vào Thùng rác. Thao tác không
  auto-save thay đổi chưa lưu trước khi xóa.
- Nếu có ảnh khác cùng stem và sidecar tồn tại, tool chặn thao tác để tránh xóa
  nhãn dùng chung. Sau khi xóa thành công, tool chọn ảnh kế tiếp; ở ảnh cuối,
  tool chọn ảnh trước đó. Nếu không còn ảnh, canvas và trạng thái hiện tại được
  reset.
