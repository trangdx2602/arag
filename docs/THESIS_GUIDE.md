# Hướng Dẫn Thực Nghiệm Luận Văn
## Entity-Aware A-RAG with Evidence Verification

---

## Mục Lục

1. [Tổng quan](#1-tổng-quan)
2. [Cài đặt môi trường](#2-cài-đặt-môi-trường)
3. [Cấu hình API Key](#3-cấu-hình-api-key)
4. [Build Index](#4-build-index)
5. [Chạy thực nghiệm (Local)](#5-chạy-thực-nghiệm-local)
6. [Đánh giá kết quả](#6-đánh-giá-kết-quả)
7. [Chạy trên Google Colab](#7-chạy-trên-google-colab)
8. [Cấu trúc kết quả](#8-cấu-trúc-kết-quả)
9. [Thứ tự thực nghiệm đề xuất](#9-thứ-tự-thực-nghiệm-đề-xuất)

---

## 1. Tổng Quan

**Branch:** `thesis-entity-evidence-arag`

### 5 Variants so sánh

| Variant | Mô tả |
|---------|-------|
| `naive_rag` | Baseline yếu nhất: 1 semantic_search + 1 LLM call |
| `arag_baseline` | A-RAG gốc (không thay đổi) |
| `arag_entity_tracker` | A-RAG + Entity-aware Context Tracker |
| `arag_evidence_checker` | A-RAG + Evidence Sufficiency Checker |
| `arag_entity_evidence_full` | A-RAG + cả 2 module |

### Datasets

| Dataset | Số câu | Số chunks | Loại |
|---------|--------|-----------|------|
| MuSiQue | 1,000 | 1,354 | Multi-hop 2–4 bước |
| HotpotQA | 1,000 | 1,311 | Multi-hop 2 bước |

---

## 2. Cài Đặt Môi Trường

### Yêu cầu
- Python 3.10+
- Git

### Clone và checkout branch

```bash
git clone https://github.com/trangdx2602/arag.git
cd arag
git checkout thesis-entity-evidence-arag
```

### Cài package

```bash
# Cách 1: pip (khuyến nghị cho Colab)
pip install -e ".[full]"

# Cách 2: uv (nhanh hơn, cho local)
uv sync --extra full
```

### Kiểm tra cài đặt thành công

```bash
python -c "from arag.agent.entity_agent import EntityAwareAgent; print('OK')"
python -m pytest tests/ -v
```

---

## 3. Cấu Hình API Key

### Tạo file `.env` từ template

```bash
cp .env.example .env
```

Mở `.env` và điền thông tin:

```env
ARAG_API_KEY=sk-...           # OpenAI / Compatible API key
ARAG_BASE_URL=https://api.openai.com/v1
ARAG_MODEL=gpt-4o-mini
```

### Load `.env` trong terminal

**Windows PowerShell:**
```powershell
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^#][^=]*)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim())
    }
}
```

**Linux / macOS / Git Bash:**
```bash
export $(grep -v '^#' .env | xargs)
```

### Kiểm tra key đã set

```bash
python -c "import os; print('API key set:', bool(os.getenv('ARAG_API_KEY')))"
```

### Dùng provider khác (OpenAI-compatible)

```env
# Ví dụ: Groq
ARAG_API_KEY=gsk_...
ARAG_BASE_URL=https://api.groq.com/openai/v1
ARAG_MODEL=llama-3.3-70b-versatile

# Ví dụ: OpenRouter
ARAG_API_KEY=sk-or-...
ARAG_BASE_URL=https://openrouter.ai/api/v1
ARAG_MODEL=openai/gpt-4o-mini
```

---

## 4. Build Index

Cần build index một lần trước khi chạy thực nghiệm (semantic search).

### MuSiQue

```bash
# CPU (chậm, ~10–15 phút)
python scripts/build_index.py \
    --chunks data/musique/chunks.json \
    --output data/musique/index \
    --model Qwen/Qwen3-Embedding-0.6B \
    --device cpu

# GPU
python scripts/build_index.py \
    --chunks data/musique/chunks.json \
    --output data/musique/index \
    --model Qwen/Qwen3-Embedding-0.6B \
    --device cuda:0
```

### HotpotQA

```bash
python scripts/build_index.py \
    --chunks data/hotpotqa/chunks.json \
    --output data/hotpotqa/index \
    --model Qwen/Qwen3-Embedding-0.6B \
    --device cuda:0
```

### Kiểm tra index đã build

```bash
python -c "
from pathlib import Path
for ds in ['musique', 'hotpotqa']:
    f = Path(f'data/{ds}/index/sentence_index.pkl')
    status = f'{f.stat().st_size:,} bytes' if f.exists() else 'NOT FOUND'
    print(f'{ds}: {status}')
"
```

---

## 5. Chạy Thực Nghiệm (Local)

Tất cả lệnh chạy từ thư mục gốc `arag/`.

### Script chính

```
python scripts/thesis/run_variants.py \
    --config <config_file> \
    --variant <variant_name> \
    --questions <questions_file> \
    --output <output_dir> \
    [--limit N] [--workers N] [--verbose]
```

| Tham số | Mô tả | Mặc định |
|---------|-------|---------|
| `--config` | File YAML config | bắt buộc |
| `--variant` | Tên variant | bắt buộc |
| `--questions` | File câu hỏi | bắt buộc |
| `--output` | Thư mục lưu kết quả | bắt buộc |
| `--limit N` | Giới hạn số câu | không giới hạn |
| `--workers N` | Số luồng song song | 5 |
| `--verbose` | In chi tiết từng câu | tắt |

---

### Bước 1 — Smoke test (3 câu, verbose)

Dùng để kiểm tra pipeline trước khi chạy nhiều:

```bash
python scripts/thesis/run_variants.py \
    --config configs/thesis/musique_base.yaml \
    --variant arag_baseline \
    --questions data/musique/questions.json \
    --output results/thesis/smoke_test \
    --limit 3 --workers 1 --verbose
```

---

### Bước 2 — MVP: 20 câu MuSiQue × 5 variants

```bash
# Naive RAG
python scripts/thesis/run_variants.py \
    --config configs/thesis/musique_base.yaml \
    --variant naive_rag \
    --questions data/musique/questions.json \
    --output results/thesis/naive_rag_musique \
    --limit 20 --workers 3

# A-RAG Baseline
python scripts/thesis/run_variants.py \
    --config configs/thesis/musique_base.yaml \
    --variant arag_baseline \
    --questions data/musique/questions.json \
    --output results/thesis/arag_baseline_musique \
    --limit 20 --workers 3

# A-RAG + Entity Tracker
python scripts/thesis/run_variants.py \
    --config configs/thesis/musique_entity.yaml \
    --variant arag_entity_tracker \
    --questions data/musique/questions.json \
    --output results/thesis/arag_entity_tracker_musique \
    --limit 20 --workers 3

# A-RAG + Evidence Checker
python scripts/thesis/run_variants.py \
    --config configs/thesis/musique_evidence.yaml \
    --variant arag_evidence_checker \
    --questions data/musique/questions.json \
    --output results/thesis/arag_evidence_checker_musique \
    --limit 20 --workers 3

# A-RAG Full (ET + EV)
python scripts/thesis/run_variants.py \
    --config configs/thesis/musique_full.yaml \
    --variant arag_entity_evidence_full \
    --questions data/musique/questions.json \
    --output results/thesis/arag_entity_evidence_full_musique \
    --limit 20 --workers 3
```

---

### Bước 3 — Mở rộng: 100 câu MuSiQue

Đổi `--limit 20` thành `--limit 100` và tăng workers:

```bash
# Ví dụ cho A-RAG baseline, 100 câu
python scripts/thesis/run_variants.py \
    --config configs/thesis/musique_base.yaml \
    --variant arag_baseline \
    --questions data/musique/questions.json \
    --output results/thesis/arag_baseline_musique_100 \
    --limit 100 --workers 5
```

> **Checkpoint resume**: Nếu bị ngắt giữa chừng, chạy lại cùng lệnh — script tự skip câu đã xong.

---

### Bước 4 — HotpotQA (sau khi MuSiQue ổn định)

```bash
# Thay musique → hotpotqa trong --config, --questions, --output
python scripts/thesis/run_variants.py \
    --config configs/thesis/hotpotqa_base.yaml \
    --variant arag_baseline \
    --questions data/hotpotqa/questions.json \
    --output results/thesis/arag_baseline_hotpotqa \
    --limit 100 --workers 5
```

---

## 6. Đánh Giá Kết Quả

### So sánh nhanh (Contain-Match, không tốn tiền)

```bash
python scripts/thesis/compare_results.py \
    --results results/thesis/ \
    --dataset musique
```

Output mẫu:

```
=== Thesis Results Comparison — MUSIQUE ===

--------------------------------------------------------------------------------------------
Variant                         N       Contain-Acc  Avg Loops   Avg Tokens   Avg Cost($)
--------------------------------------------------------------------------------------------
naive_rag                       20      15.0%        1.000       450          0.00012
arag_baseline                   20      40.0%        4.200       1820         0.00089
arag_entity_tracker             20      45.0%        3.800       1650         0.00095
arag_evidence_checker           20      43.0%        4.500       1900         0.00102
arag_entity_evidence_full       20      48.0%        4.100       1700         0.00108
--------------------------------------------------------------------------------------------
```

### LLM-Accuracy (tốn thêm API call, chính xác hơn)

```bash
# Chạy cho từng variant
python scripts/eval.py \
    --predictions results/thesis/arag_baseline_musique/predictions.jsonl \
    --workers 5

# Output lưu tại: results/thesis/arag_baseline_musique/predictions_eval_summary.json
```

### Xem kết quả eval summary

```bash
python -c "
import json
from pathlib import Path
for p in sorted(Path('results/thesis').glob('*/predictions_eval_summary.json')):
    d = json.load(open(p))
    print(f'{p.parent.name}:')
    print(f'  LLM-Acc: {d[\"llm_accuracy\"]:.1%}  |  Contain-Acc: {d[\"contain_accuracy\"]:.1%}  |  Loops: {d[\"avg_loops\"]:.1f}  |  Cost: \${d[\"avg_cost_per_question\"]:.5f}')
"
```

---

## 7. Chạy Trên Google Colab

### Mở notebook

```
notebooks/thesis_colab.ipynb
```

Upload lên Google Colab hoặc mở trực tiếp từ GitHub.

### Set API key trong Colab Secrets

1. Click biểu tượng 🔑 (Secrets) ở thanh bên trái
2. Thêm secret: `ARAG_API_KEY` = `sk-...`
3. Thêm secret: `ARAG_MODEL` = `gpt-4o-mini`

### Thứ tự chạy các cell

| Cell | Nội dung | Lưu ý |
|------|---------|-------|
| Cell 1 | Clone repo + checkout branch | Sửa `REPO_URL` nếu dùng fork |
| Cell 2 | Cài môi trường + load API key | Phải set Colab Secrets trước |
| Cell 3 | Mount Google Drive | Cần xác nhận quyền |
| Cell 4 | Build index MuSiQue + HotpotQA (GPU) | ~2–3 phút/dataset |
| Cell 5 | Chạy Naive RAG | Sửa `LIMIT` theo nhu cầu |
| Cell 6 | Chạy A-RAG Baseline | — |
| Cell 7 | Chạy A-RAG + Entity Tracker | — |
| Cell 8 | Chạy A-RAG + Evidence Checker | — |
| Cell 9 | Chạy A-RAG + Full | — |
| Cell 10 | Evaluate + bảng so sánh | Xem kết quả tại đây |
| Cell 11 | Copy results về Google Drive | — |

### Điều chỉnh số câu trong notebook

Trong các cell 5–9, sửa biến `LIMIT`:

```python
LIMIT = 20    # MVP: kiểm tra pipeline
LIMIT = 100   # Kết quả sơ bộ
LIMIT = 500   # Kết quả luận văn chính thức
```

---

## 8. Cấu Trúc Kết Quả

### File đầu ra

```
results/thesis/
├── arag_baseline_musique/
│   ├── predictions.jsonl          # Kết quả raw (1 dòng = 1 câu)
│   └── predictions_eval_summary.json  # Tổng hợp sau khi chạy eval.py
├── arag_entity_tracker_musique/
│   └── predictions.jsonl
├── ...
└── comparison_musique.json        # Bảng so sánh từ compare_results.py
```

### Cấu trúc 1 dòng trong `predictions.jsonl`

```json
{
  "qid": "musique_2hop__123",
  "question": "Where was the director of Inception born?",
  "gold_answer": "London",
  "pred_answer": "Christopher Nolan was born in London, England.",
  "loops": 4,
  "total_retrieved_tokens": 1820,
  "total_cost": 0.00089,
  "chunks_read_count": 3,
  "chunks_read_ids": ["42", "87", "103"],
  "entity_count": 5,
  "verification": {
    "is_sufficient": true,
    "coverage_score": 0.85,
    "missing_gaps": []
  },
  "trajectory": [
    {
      "loop": 1,
      "tool_name": "semantic_search",
      "arguments": {"query": "director of Inception"},
      "tool_result": "...",
      "retrieved_tokens": 320
    }
  ]
}
```

| Trường | Có ở variant |
|--------|-------------|
| `entity_count` | ET, Full |
| `verification` | EV, Full |
| `trajectory` | Tất cả (trừ naive_rag chỉ có 1 entry) |

---

## 9. Thứ Tự Thực Nghiệm Đề Xuất

```
Giai đoạn 1 — Kiểm tra pipeline
    └── 3 câu MuSiQue, variant arag_baseline, --verbose
        → Xác nhận API key + index + output đúng format

Giai đoạn 2 — MVP (20 câu MuSiQue)
    └── 5 variants × 20 câu
        → compare_results.py → kiểm tra số liệu có chiều hướng đúng

Giai đoạn 3 — Sơ bộ (100 câu MuSiQue)
    └── 5 variants × 100 câu
        → eval.py (LLM-Accuracy) → kết quả báo cáo tiến độ

Giai đoạn 4 — Đầy đủ (500 câu MuSiQue + 500 câu HotpotQA)
    └── 5 variants × 500 câu × 2 datasets
        → eval.py trên toàn bộ → bảng kết quả luận văn chính thức

Giai đoạn 5 — Phân tích
    └── Error analysis, case study, viết luận văn
```

---

## Ghi Chú Nhanh

```bash
# Kiểm tra tiến độ chạy (số câu đã xong)
python -c "
from pathlib import Path
for p in sorted(Path('results/thesis').glob('*/predictions.jsonl')):
    lines = sum(1 for _ in open(p, encoding='utf-8'))
    print(f'{p.parent.name}: {lines} câu')
"

# Xem 1 câu kết quả cụ thể
python -c "
import json
with open('results/thesis/arag_baseline_musique/predictions.jsonl', encoding='utf-8') as f:
    row = json.loads(f.readline())
print('Q:', row['question'])
print('Gold:', row['gold_answer'])
print('Pred:', row['pred_answer'])
print('Loops:', row['loops'], '| Cost: \$', row['total_cost'])
"
```
