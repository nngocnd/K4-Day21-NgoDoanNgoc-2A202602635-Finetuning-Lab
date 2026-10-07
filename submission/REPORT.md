# Lab 21 — Evaluation Report

**Họ tên**: Ngo Doan Ngoc  **MSSV**: 2A202602635  **Ngày**: 2026-10-07
**Adapter (HF Hub)**: https://huggingface.co/DNgoc/lab21-qwen35-triage-vi
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4 (Colab Free, 14,6 GB khả dụng, sm_75 → fp16)

> Mọi con số dưới đây lấy trực tiếp từ các file trong `results/` (lần chạy đầy đủ, `eval_limit = null`,
> `smoke_mode = false`). Lần chạy đầu tiên bị mất vì VM Colab bị thu hồi trước khi tải kết quả về,
> nên toàn bộ pipeline NB1 → NB5 đã được chạy lại từ đầu; báo cáo này chỉ dùng số của lần chạy thứ hai.

---

## 0. Lựa chọn thí nghiệm và lý do

| | Lựa chọn | Lý do |
|---|---|---|
| Base model | `unsloth/Qwen3.5-4B` (mặc định của tier T4) | Vừa T4 ở bf16/fp16 LoRA (peak 8,78 GB / 14,6 GB). Giữ model mặc định để mask đã được kiểm chứng sẵn bằng `check_mask_agreement.py` và mốc NB2 so được với cùng một base. |
| Dataset | Corpus mặc định: 250 ticket CSKH tiếng Việt → JSON 4 trường (`intent`, `urgency`, `product`, `sentiment`) | Mọi nhóm điểm có thang khách quan (so trường với nhãn, parse JSON), không cần LLM-judge. Đây là lần đầu chạy pipeline nên dùng corpus mặc định trước, đúng gợi ý của README. |
| Report | Theo khung mẫu, tự viết nội dung | Khung mẫu khớp đủ 6 mục rubric 4.1. |

Mốc NB2 được đóng băng **trước** khi train, và cùng một base model `unsloth/Qwen3.5-4B` được dùng cho cả ba baseline lẫn bốn adapter.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (`data/train_seed.jsonl`); eval: 50 ticket target + 15 câu regression |
| Train / val | 225 / 25 (seed 42, `train_frac = 0.9`) |
| `max_length` | 1024 (giá trị của tier T4) — p95 đo được là **98** token, max 101, gợi ý 256 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch → **30** optimizer step (batch 1 × grad_accum 16 = batch hiệu dụng 16 < 32) |
| Precision | fp16 + gradient scaling (T4 không có bf16) |

**Về `max_length`:** NB1 đo p95 = 98 token, p99 = 100, dài nhất 101 trên 250 mẫu, và gợi ý `max_length = 256`.
Tôi giữ giá trị 1024 của tier. Vì mẫu dài nhất chỉ có 101 token, cả 256 lẫn 1024 đều **không cắt bỏ
mẫu nào**, nên trên corpus này lựa chọn đó không ảnh hưởng đến kết quả. Giữ nguyên giá trị tier cũng giúp
cấu hình train khớp với cấu hình mà `make verify` và các run NB4 dùng. Nếu đổi sang dataset có câu trả lời
dài (ví dụ có reasoning trace) thì phải đo lại p95 và đặt lại `max_length` theo nó.

**Template có giữ khối `<think>` không?** **Có** — `template_check.json`: `ok: true`, `open_tag_present: true`,
`body_present: true`, verdict *"reasoning preserved — safe to train on traces"*. Chuỗi render:
`<|im_start|>assistant\n<think>\nbuoc 1: kiem tra. buoc 2: tra loi.\n</think>\n\n4<|im_end|>`.
Trên corpus mặc định, câu trả lời huấn luyện là JSON trần không có trace, nên `valid_trace_rate = 0.0` trong
`verdict.json` là đúng kỳ vọng (không có trace nào để giữ hay làm mất), không phải lỗi.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39 / 94 token) — so với 1.00 (94/94) ở mode `everything` |
| Câu trả lời nằm trong loss | `true` (`answer_is_supervised`) |
| Câu hỏi KHÔNG nằm trong loss | `true` (`question_is_masked`) |

Đoạn được tính loss (giải mã ngược các vị trí `labels != -100`, `results/mask_proof.json`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn bị mask (không tính loss):

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>

```

Nhận xét: ranh giới mask nằm ngay sau `<think>\n\n` mà template tự chèn vào generation prompt; phần được
học là `</think>` + JSON + `<|im_end|>`. Model học cả token EOS, đó là lý do `format = 1.0` sau fine-tune.
`supervised_fraction = 0.41` thấp xa ngưỡng 0.95, nên loss không bị tính trên prompt.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

Nguồn: `results/baselines_frozen.json` (a, b) và `results/verdict.json` (c). n = 50 ticket target, 15 câu regression.

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3253.0 |
| (b) base + optimized prompt + few-shot | 0.765 | 0.7911 | 1.000 | 1030.3 |
| (c) LoRA fine-tune (`correct`) | **0.970** | **0.5444** | 1.000 | 1556.2 |

**(b) có thật sự mạnh hơn (a) không?** **Có**, rất rõ: target 0.000 → 0.765, format 0.000 → 1.000. Với prompt
ngây thơ, base model không trả về JSON parse được ở bất kỳ mẫu nào (format = 0), nên target cũng bằng 0, và
nó còn chậm gấp ~3 lần vì sinh văn bản dài. Prompt tối ưu + few-shot sửa được hoàn toàn vấn đề định dạng.

**Có sửa `OPTIMIZED_PROMPT` không?** **Không.** `make verify` xác nhận *"baseline (b) prompt unmodified"*,
SHA `719e74d3b6232053`. Không làm mạnh, không làm yếu.

---

## 4. Giải phẫu cấu hình sai (NB4)

Nguồn: `results/runs.csv` (train loss, tham số, VRAM, thời gian) và `results/autopsy.json` (target/format NB5 §4).
Cả bốn run dùng **cùng 30 step** (`make verify`: *"all runs share ONE step budget"*).

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | format | train s | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear (12 module) | 16 | 32,464,896 | 1e-4 | 0.6268 | **0.97** | 1.00 | 413.0 | 8.78 |
| `attn_only` | q,v (2 module) | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5368 | **0.97** | 1.00 | 271.7 | 8.79 |
| `wrong_lr` | text-linear (12 module) | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.00** | 0.00 | 416.9 | 8.78 |
| `qlora` | text-linear (12 module), 4-bit | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.94** | 1.00 | 514.4 | 3.86 |

Mỗi run đổi **đúng một biến** so với `correct`:
- `attn_only`: **vị trí** gắn adapter (q,v thay vì toàn bộ linear), rank nâng lên 283 để khớp ngân sách tham số (sai lệch 8.192 / 32.464.896 = 0,025% < 5%).
- `wrong_lr`: **learning rate** (1e-5, thang full-FT, thay vì 1e-4 = 10×).
- `qlora`: **độ chính xác của trọng số base** (4-bit thay vì 16-bit).

**Xếp hạng theo target (NB5):** `correct` = `attn_only` (0.97) > `qlora` (0.94) > `wrong_lr` (0.00).
**Xếp hạng theo train loss (NB4):** `attn_only` (0.537) < `correct` (0.627) < `qlora` (0.706) < `wrong_lr` (1.570).
Hai thứ tự **khác nhau ở ngôi đầu**: theo loss thì `attn_only` "thắng", theo tác vụ thì nó chỉ hoà.

**4.1 — `attn_only` vs `correct`.** Với ngân sách tham số gần như bằng nhau (32,46 M vs 32,46 M), `attn_only`
**hoà** `correct` trên tập target (0.97 = 0.97, format đều 1.00). Nhưng train loss của nó thấp hơn rõ (0.537 vs
0.627). Nếu xếp hạng bằng loss, tôi sẽ kết luận sai rằng gắn adapter vào attention với rank lớn tốt hơn. Thực tế
loss thấp hơn chỉ cho thấy LoRA rank 283 trên 2 module ghi nhớ 225 mẫu huấn luyện tốt hơn, không phải là phân loại
ticket mới tốt hơn. Về *rank vs vị trí*: khi ngân sách tham số được giữ cố định, đổi vị trí không làm thay đổi kết quả
trên tác vụ hẹp này, nên **ở đây cả vị trí lẫn rank đều không phải đòn bẩy**. Tác vụ triage 4 trường đủ đơn giản để
32 M tham số đặt ở đâu cũng đủ. So sánh công bằng này chỉ ra rằng nếu so `q,v @ r=16` với `all-linear @ r=16`, khoảng
cách thấy được sẽ là do *ngân sách*, không phải do *vị trí*. Một điểm phụ đáng ghi: `attn_only` train nhanh hơn
(271,7 s vs 413,0 s) và sinh nhanh hơn (993,5 ms vs 1556,2 ms) vì chỉ chèn LoRA vào 2 loại module.

**4.2 — `wrong_lr`.** Chỉ đổi LR từ 1e-4 xuống 1e-5, nhưng train loss dừng ở 1,570 sau 30 step (so với 0,627), và
quan trọng hơn: target = 0.00, format = 0.00. Adapter **không học được cả định dạng JSON**, nên hành vi gần như
giống base model với prompt ngây thơ (baseline (a) cũng có target 0, format 0). Latency 5672,9 ms vì model sinh
văn bản tự do cho tới giới hạn token thay vì dừng sau JSON. Nếu chỉ nhìn đường loss mà không biết LR, tôi dễ kết
luận sai rằng *dữ liệu quá khó* hoặc *LoRA không đủ sức với 4B* và đi tăng rank. Thực ra chỉ cần nhân LR lên 10
lần, đúng như khuyến nghị LoRA ≈ 10× LR full-FT (§11.3).

**4.3 — `qlora`.** QLoRA 4-bit giảm peak VRAM từ 8,78 GB xuống **3,86 GB (−56 %)**. Đổi lại: target giảm
0.97 → **0.94** (−0.03, tức thêm 6 trường sai trên 200 trường), train chậm hơn 25 % (514,4 s vs 413,0 s, vì phải
dequantize mỗi bước), sinh chậm hơn (1962,3 ms vs 1556,2 ms), và train loss cao hơn (0,706 vs 0,627). Số đo của tôi
**ủng hộ khuyến nghị không dùng QLoRA cho dòng model này khi bf16/fp16 LoRA đã vừa GPU**: trên T4, 16-bit LoRA vừa
(8,78 / 14,6 GB) nên trả giá bằng chất lượng và tốc độ để tiết kiệm VRAM là không đáng. QLoRA chỉ hợp lý khi
16-bit không vừa (ví dụ 9B trên T4). Mức tụt 0.03 không lớn, nhưng đó là cái giá đo được, không phải miễn phí.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **`FAILED`**
`target Δ = +0.205` · `regression Δ = −0.247` · `valid_trace_rate = 0.0` *(results/verdict.json)*

Lý do do cổng ghi: *"general capability regressed by 0.247 (tolerance 0.020). See deck §6.3 — add 1-5% replay data."*

**Diễn giải.** Bản fine-tune **thắng rõ trên tác vụ đích**: target 0.765 → 0.970 (+0.205) so với đối thủ thật
là base + prompt tối ưu + few-shot, và đạt điều đó với prompt ngắn hơn (không cần few-shot trong context). Nhưng nó
**trượt cổng hồi quy**: điểm regression trên 15 câu hỏi phổ thông giảm từ 0.7911 xuống 0.5444 (−0.247), gấp hơn 12
lần mức dung sai 0.02. Đây là **quên thảm hoạ** (catastrophic forgetting) điển hình. Toàn bộ 225 mẫu train cùng một
dạng (ticket → JSON 4 khoá), và với LR 1e-4 trên toàn bộ linear layer trong 30 step, model bị kéo mạnh về hành vi
"luôn trả về JSON triage". Train loss xuống 0,02–0,03 ở các bước cuối (log NB3) cho thấy model đã gần như thuộc lòng
một định dạng đầu ra duy nhất. Khi gặp câu hỏi phổ thông, adapter vẫn đẩy model về phía phong cách đó và làm mất khả
năng trả lời tự do.

Theo thứ tự chẩn đoán của NB5: format = 1.0 nên template/mask không có vấn đề (NB1 đã chứng minh); target tăng mạnh
nên LR không có vấn đề. Lỗi nằm ở bước 2: **regression tụt → cần replay**. FAILED ở đây không có nghĩa là
*bài toán này không cần fine-tune*, vì fine-tune đúng là thắng prompt engineering trên tác vụ đích. Nó có nghĩa là
**bản fine-tune này chưa được phép ship** nếu model còn phải phục vụ câu hỏi chung. Nếu triển khai như một endpoint
chuyên dụng chỉ nhận ticket CSKH thì regression không còn là tiêu chí, nhưng cổng của lab được đặt cho một model đa
dụng và tôi không nới ngưỡng.

---

## 6. Định tính — bắt buộc có cả ca THUA

Nguồn: `results/qualitative.json` (dự đoán của bản fine-tune + điểm theo trường) và nhãn trong `data/eval_target.jsonl`.
Lưu ý trung thực: pipeline **không lưu dự đoán từng mẫu của baseline (b)**, chỉ lưu điểm tổng hợp, nên cột (b) dưới
đây không có dữ liệu từng mẫu. Các ca "FT thua" là ca bản fine-tune sai so với nhãn, cộng với ca thua trên nhóm
regression, nơi (b) đo được tốt hơn hẳn (0.7911 vs 0.5444).

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 (i=0) | "…chuột không dây… Cho tôi trả lại. **Gấp**. Shop hỗ trợ tốt." | doi_tra · cao · chuột không dây · tich_cuc | không lưu | doi_tra · cao · chuột không dây · tich_cuc (1.00) | ✅ FT đúng cả 4 trường, kể cả urgency "Gấp" → cao |
| 2 (i=47) | "…ốp lưng điện thoại… Shipper không gọi. Hỏi cho biết thôi. Shop hỗ trợ tốt." | van_chuyen · thap · ốp lưng điện thoại · tich_cuc | không lưu | van_chuyen · thap · ốp lưng điện thoại · … (1.00) | ✅ FT hiểu "Shipper không gọi" là vận chuyển, dù không có chữ "giao hàng" |
| 3 (i=4) | "…đèn bàn LED… Vỡ khi nhận. **Gấp**. Shop xem giúp." | san_pham_loi · cao · đèn bàn LED · trung_tinh | không lưu | san_pham_loi · cao · đèn bàn LED · trung_tinh (1.00) | ✅ FT đúng |
| 4 (i=3) | "…bình giữ nhiệt… Chưa thấy tiền. **Khi nào tiện**. Cảm ơn shop nhiều." | hoan_tien · **thap** · bình giữ nhiệt · tich_cuc | không lưu | hoan_tien · **trung_binh** · bình giữ nhiệt · … (0.75) | ❌ **FT thua**: sai urgency |
| 5 (i=39) | "…nồi chiên không dầu… Hoàn tiền. **Khi nào tiện**. Quá tệ." | hoan_tien · **thap** · nồi chiên không dầu · tieu_cuc | không lưu | hoan_tien · **trung_binh** · nồi chiên không dầu · … (0.75) | ❌ **FT thua**: sai urgency; có lẽ "Quá tệ" kéo model nghĩ là không thể chờ |
| 6 (i=12) | "…áo khoác gió… Bị lỗi. **Khi nào tiện**. Cảm ơn shop nhiều." | san_pham_loi · **thap** · áo khoác gió · tich_cuc | không lưu | san_pham_loi · **trung_binh** · áo khoác gió · … (0.75) | ❌ **FT thua**: sai urgency |
| 7 | 15 câu hỏi phổ thông (vd. "Thủ đô của Việt Nam là thành phố nào?", "Dịch sang tiếng Anh: 'Tôi thích đọc sách'") | từ khoá đúng | **0.7911** | **0.5444** | ❌ **FT thua** (b) trên toàn nhóm regression: quên thảm hoạ |

**Có mẫu chung nào ở các ca FT thua không?** **Có, và rất rõ.** Cả **6/6** ticket sai trên tập target (i = 3, 5, 12,
39, 41, 46) đều sai **cùng một trường (urgency) theo cùng một kiểu**: nhãn `thap`, model đoán `trung_binh`, và cả 6
đều chứa cụm **"Khi nào tiện"**. Ngược lại, tập eval có đúng 6 ticket chứa cụm này, nên model sai **0/6** trên nhóm này
và đúng 100 % ở mọi ticket còn lại. Điều đáng chú ý: trong `train_seed.jsonl` có 35 ticket chứa "Khi nào tiện" và
**cả 35 đều được gán `thap`**, nên dữ liệu không mâu thuẫn. Giả thuyết của tôi: cụm này đứng cạnh các cụm "trung tính"
khác ("Cho tôi hỏi", "Cảm ơn shop") vốn hay đi với `trung_binh` trong train, và với chỉ 30 step model chưa học được
rằng "Khi nào tiện" phải lấn át chúng. Prior "không gấp nhưng vẫn cần trả lời" của base model cũng có thể kéo về mức
giữa. Đây là lỗi hệ thống của *một tín hiệu cụ thể*, không phải nhiễu ngẫu nhiên. Nó gợi ý cách sửa qua dữ liệu
(thêm ví dụ tương phản) chứ không phải qua rank hay LR.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi **không** deploy bản fine-tune này như một model đa dụng. Trên tác vụ đích, nó thắng thật và thắng
đậm: 0.970 so với 0.765 của đối thủ mạnh nhất (base + prompt tối ưu + few-shot), format 100 %, và lỗi còn lại tập
trung vào đúng một tín hiệu ("Khi nào tiện" → urgency). Nhưng cổng hồi quy FAILED vì điểm câu hỏi phổ thông giảm từ
0.791 xuống 0.544. Nguyên nhân là dữ liệu huấn luyện thuần một dạng: 225 mẫu cùng định dạng JSON, LR 10× trên toàn bộ
linear layer, nên adapter kéo cả phân phối đầu ra về một hành vi. Tôi sẽ chỉ triển khai nếu (1) endpoint chỉ nhận
ticket CSKH (adapter được hot-swap riêng cho tác vụ này, model gốc vẫn phục vụ câu hỏi chung), hoặc (2) sau khi train
lại với 1–5 % dữ liệu replay câu hỏi phổ thông và đo lại để regression nằm trong dung sai 0.02.

Về câu hỏi *đâu là đòn bẩy thật*, thí nghiệm có kiểm soát trả lời khá rõ. **Learning rate** là đòn bẩy lớn nhất: chỉ
đổi LR 10× là đi từ 0.00 lên 0.97 target. **Mask đúng** là điều kiện tiên quyết: chính nó cho model học được EOS và
format 1.0. **Vị trí adapter không phải đòn bẩy** trên tác vụ này: cùng ngân sách tham số thì attention-only hoà
all-linear. **QLoRA** đổi 56 % VRAM lấy −0.03 target và chậm hơn 25 %, không đáng khi 16-bit đã vừa. Cuối cùng,
**thành phần dữ liệu** mới là thứ quyết định pass/fail: thiếu replay gây ra hồi quy, và một cụm từ khó cần thêm ví dụ
tương phản. Điều này nhất quán với tinh thần của lab: điểm số thay thế (train loss) xếp sai thứ tự `attn_only`, và
chỉ cổng bốn nhóm mới phát hiện ra chi phí hồi quy mà riêng điểm target sẽ che mất.

**Ba điều tôi học được** (cụ thể):
1. **Train loss thấp hơn không có nghĩa là model tốt hơn.** `attn_only` có loss 0.537 < 0.627 của `correct` nhưng
   target bằng nhau (0.97). Nếu chọn adapter theo `final_loss`, tôi sẽ chọn sai lý do. Từ nay tôi xếp hạng run bằng
   điểm trên tập eval đã đóng băng, không bằng loss.
2. **Một prompt tốt có thể sửa phần lớn vấn đề trước khi cần fine-tune.** Riêng prompt tối ưu + few-shot đã đưa target
   từ 0.000 lên 0.765 và format từ 0 lên 1.0 mà không train gì. Phần thắng thật của fine-tune chỉ là khoảng +0.205 còn
   lại, và phần đó phải trả bằng −0.247 regression. Nếu không đo (b) trước khi train, tôi sẽ tưởng fine-tune mang lại
   +0.97.
3. **Lỗi của model thường có cấu trúc, và phân tích lỗi chỉ ra cách sửa.** 6/6 lỗi là cùng một trường, cùng một cụm
   từ, cùng một hướng sai. Đọc từng dự đoán thua cho tôi một giả thuyết kiểm chứng được (thêm ví dụ tương phản cho
   "Khi nào tiện"), thứ mà con số 0.97 tổng hợp không bao giờ cho thấy.

Ngoài ra, về vận hành: VM Colab bị thu hồi gần như ngay sau khi pipeline chạy xong và tôi mất toàn bộ `results/`
lần đầu. Lần hai tôi thêm một ô tự nén và tải `results/` + `adapters/correct/` về ngay sau khi verify. Bài học: kết
quả chưa được lưu ra ngoài VM thì coi như chưa có.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
- Trộn 1–5 % dữ liệu replay câu hỏi phổ thông (deck §6.3) vào tập train, train lại `correct` cùng 30 step, và đo xem
  regression có quay về trong dung sai 0.02 mà vẫn giữ target ≥ 0.765 hay không. Đây là thí nghiệm trực tiếp nhất để
  đổi FAILED thành PASSED một cách trung thực.
- Thêm ví dụ tương phản cho cụm "Khi nào tiện" (cùng cụm nhưng đi với các tín hiệu khác) để kiểm tra giả thuyết lỗi urgency.
- Giảm `max_length` xuống 256 theo p95 để xác nhận rằng nó không đổi kết quả nhưng giảm bộ nhớ.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — link: https://huggingface.co/DNgoc/lab21-qwen35-triage-vi (adapter `correct`, public)

**Trạng thái `make verify` (Colab, trước khi điền report):** 25 passed · 1 warning · 1 failure. Failure duy nhất là
*"REPORT.md filled in"* (vì lúc đó report vẫn là mẫu), và chính báo cáo này sửa lỗi đó. Các kiểm tra liêm chính đều xanh:
mask proof, full eval set (50 item), prompt (b) không sửa, (b) > (a) (0.000 → 0.765), eval set không sửa, cùng 30 step
cho bốn run, `attn_only` khớp ngân sách (32,456,704 vs 32,464,896).
