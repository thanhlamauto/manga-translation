## Hướng dẫn chạy infer (Mac & Windows)

File này hướng dẫn cách cài đặt và chạy suy luận (infer) cho việc dịch truyện / PDF bằng script `translate_pdf.py`.

---

## 1. Yêu cầu hệ thống

- **Python**: 3.10–3.12 (khuyến nghị dùng bản giống trong repo nếu có ghi rõ).
- **GPU (tùy chọn)**:
  - Windows + NVIDIA: có thể tận dụng GPU cho một số model (nếu bạn cài đủ CUDA/PyTorch).
  - macOS (Apple Silicon / Intel): hiện tại pipeline **tự fallback về CPU**, ngay cả khi bạn bật `--gpu`, để tránh lỗi CoreML.
- **Dung lượng đĩa**:
  - Các model ONNX/HF (detection, OCR, inpainting) có thể chiếm vài GB.

---

## 2. Cài đặt phụ thuộc (dependencies)

### 2.1. Cài đặt hệ thống cho `pdf2image` (Poppler)

`translate_pdf.py` dùng `pdf2image`, cần **Poppler** để chuyển PDF → ảnh.

- **macOS (Homebrew)**:

```bash
brew install poppler
```

- **Windows**:
  1. Tải bản Poppler cho Windows (ví dụ: tìm “poppler for windows” và tải bản `.zip`) và giải nén, ví dụ:
     - `C:\tools\poppler-23.08.0\`
  2. Thêm đường dẫn `bin` vào `PATH`, ví dụ:
     - `C:\tools\poppler-23.08.0\Library\bin`
  3. Mở lại terminal / PowerShell sau khi chỉnh `PATH`.

Kiểm tra nhanh:

```bash
pdftoppm -h
```

Nếu lệnh tồn tại mà không báo “command not found” là OK.

---

### 2.2. Cài đặt bằng `uv` (khuyến nghị)

Repo đã có `pyproject.toml` và `uv.lock`, nên dùng `uv` là tiện nhất.

1. Cài `uv` (nếu chưa có):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Hoặc trên Windows (PowerShell):

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

2. Vào thư mục project:

```bash
cd /path/to/comic-translate
```

3. Tạo và dùng virtualenv + cài deps:

```bash
uv sync
```

Lệnh này sẽ:
- Tạo `.venv/`
- Cài mọi dependency theo `pyproject.toml` + `uv.lock`

4. (Tùy chọn) Kích hoạt `.venv` thủ công:

- macOS / Linux:

```bash
source .venv/bin/activate
```

- Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

> Khi dùng `uv run ...` thì `uv` sẽ tự dùng môi trường đã sync, không nhất thiết phải `activate`.

---

### 2.3. Cài đặt bằng `pip` (nếu không dùng `uv`)

Không khuyến khích (vì repo đã chuẩn cho `uv`), nhưng nếu bạn muốn `pip`:

1. Tạo virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.\.venv\Scripts\activate    # Windows
```

2. Cài phụ thuộc:

```bash
pip install -r requirements.txt
```

---

## 3. Chạy infer trên PDF

### 3.1. Lệnh cơ bản

Từ thư mục project (`comic-translate/`), với `uv`:

```bash
uv run translate_pdf.py \
  --pdf input.pdf \
  --out output_translated.pdf \
  --source English \
  --target Vietnamese \
  --translator google \
  --inpainter lama_manga \
  --gpu
```

Giải thích tham số:
- **`--pdf`**: đường dẫn file PDF gốc.
- **`--out`**: đường dẫn file PDF đầu ra sau khi dịch + inpaint.
- **`--source`**: ngôn ngữ nguồn (tên tiếng Anh, ví dụ `English`, `Japanese`, `Korean`).
- **`--target`**: ngôn ngữ đích (ví dụ `English`, `Vietnamese`, ...).
- **`--translator`**: key translator:
  - `google` → **Google Translate** (miễn phí, không cần API key).
  - Các giá trị khác (DeepL, GPT, …) cần cấu hình credential trong UI, **headless script hiện mặc định an toàn hơn với `google`**.
**Ghi chú về mô hình dịch có ngữ cảnh (PDF headless)**:

- Khi chạy `translate_pdf.py`, các block thoại sẽ được gom thành transcript, sau đó đi qua pipeline trong `processing.py` (local context, character profiles, enrich transcript, translate) để dịch sang tiếng Việt (hiện tối ưu cho English → Vietnamese).
- Để pipeline này hoạt động, bạn cần đặt biến môi trường `OPENAI_API_KEY` hợp lệ (OpenAI API).

- **`--inpainter`**:
  - `lama_manga` → ánh xạ nội bộ sang `LaMa` (model inpaint dành cho manga/anime).
  - Bạn có thể dùng `aot` nếu muốn AOT-GAN (nếu đã được map đúng).
- **`--gpu`**:
  - Windows + NVIDIA: có thể dùng GPU cho một số phần (tùy môi trường CUDA/PyTorch).
  - macOS: script đã cấu hình để tránh dùng CoreML cho ONNX, nên phần lớn sẽ chạy trên CPU kể cả khi bạn bật `--gpu`.

Nếu muốn **ép CPU hoàn toàn**, chỉ cần bỏ `--gpu`:

```bash
uv run translate_pdf.py \
  --pdf input.pdf \
  --out output_translated.pdf \
  --source English \
  --target Vietnamese \
  --translator google \
  --inpainter lama_manga
```

---

## 4. Ghi chú riêng cho macOS

- Không cần cài CUDA.
- ONNX Runtime sẽ **bị chặn CoreML provider** trong code, để tránh crash với một số model (RT-DETR, LaMa ONNX).
- Hiệu năng sẽ **chủ yếu là CPU**; tốc độ phụ thuộc vào:
  - Độ phân giải PDF (`--dpi`, mặc định 300),
  - Số trang,
  - Model detection/OCR/inpainting.

Nếu thấy quá chậm:
- Giảm `--dpi` (ví dụ `--dpi 200`),
- Dùng PDF ít trang hơn để test.

---

## 5. Ghi chú riêng cho Windows

- Nếu bạn có **GPU NVIDIA + CUDA + PyTorch**:
  - Một số phần của pipeline (đặc biệt là khi dùng backend Torch) có thể tận dụng GPU.
  - Tuy nhiên script headless hiện đang ưu tiên backend ONNX cho inpainting/detection, nên lợi ích GPU phụ thuộc cấu hình thực tế.
- Quan trọng nhất là:
  - Cài **Poppler** (mục 2.1),
  - Chạy lệnh trong **PowerShell / CMD** với quyền user bình thường (không cần admin, trừ khi chỉnh PATH).

Ví dụ lệnh trong PowerShell:

```powershell
cd C:\path\to\comic-translate
uv run translate_pdf.py `
  --pdf input.pdf `
  --out output_translated.pdf `
  --source English `
  --target Vietnamese `
  --translator google `
  --inpainter lama_manga `
  --gpu
```

---

## 6. Kiểm tra kết quả

Sau khi lệnh chạy xong, bạn sẽ thấy:

- Log dạng:

```text
Translating pages: 100%|██████████| 19/19 [..]
[INFO] Rendering text with font: /System/Library/Fonts/Supplemental/Arial Bold.ttf
[INFO] Text rendering successful
Done: output_translated.pdf
```

- File `output_translated.pdf` sẽ:
  - Được inpaint để xóa text gốc,
  - Vẽ lại text đã dịch (tiếng Việt) lên các bubble / block tương ứng.

Bạn có thể mở `output_translated.pdf` bằng bất kỳ trình đọc PDF nào để kiểm tra.

Nếu cần tinh chỉnh thêm (font, kích thước, outline, ngôn ngữ, model…), có thể mở file `translate_pdf.py` và điều chỉnh phần:

- `HeadlessSettings` (font/uppercase/context),
- `render_translations` (font, màu, size),
- Tham số CLI trong hàm `main()`.


