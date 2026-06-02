# Thesis: Entity-Aware A-RAG with Evidence Verification

## Mục Tiêu

Nghiên cứu cải tiến mô hình A-RAG dựa trên theo dõi thực thể và kiểm chứng bằng chứng trong hỏi đáp đa bước.

Baseline: Naive RAG + A-RAG gốc (không thay đổi).  
Cải tiến: 2 module mới — **Entity-aware Context Tracker** và **Evidence Sufficiency Checker**.

## Các Variant

| Variant | Mô tả | Config |
|---------|-------|--------|
| `naive_rag` | One-shot: 1 semantic_search + 1 LLM call | (no config, built-in) |
| `arag_baseline` | A-RAG gốc, BaseAgent + default.txt | `musique_base.yaml` |
| `arag_entity_tracker` | EntityAwareAgent + entity tracking ON | `musique_entity.yaml` |
| `arag_evidence_checker` | EntityAwareAgent + evidence gate ON | `musique_evidence.yaml` |
| `arag_entity_evidence_full` | Cả 2 module | `musique_full.yaml` |

## Cấu Trúc Code Mới

```
src/arag/
├── core/
│   ├── entity_state.py         # EntityNode, Relation, EntityState
│   └── context.py              # Thêm entity_state, verification_result (backward-compat)
├── agent/
│   ├── entity_agent.py         # EntityAwareAgent (subclass BaseAgent)
│   └── naive_rag.py            # NaiveRAGAgent
├── tools/
│   ├── entity_lookup.py        # Tra EntityState không cần retrieval
│   └── check_evidence.py       # Tool kiểm chứng bằng chứng
└── verification/
    ├── extractor.py             # EntityExtractor (LLM-based)
    └── checker.py               # EvidenceSufficiencyChecker (LLM-based)

configs/thesis/                  # 8 config files (musique + hotpotqa × 4 variants)
scripts/thesis/
    run_variants.py              # CLI runner cho 1 variant + 1 dataset
    compare_results.py           # In bảng so sánh results
notebooks/
    thesis_colab.ipynb           # Notebook Colab chính (11 cells)
```

## Cách Chạy Local

### Cài môi trường
```bash
git clone <repo-url>
cd arag
git checkout thesis-entity-evidence-arag
pip install -e ".[full]"
```

### Set API key
```bash
export ARAG_API_KEY="your-openai-key"
export ARAG_MODEL="gpt-4o-mini"
# Hoặc dùng ARAG_BASE_URL để dùng provider khác
```

### Build index MuSiQue
```bash
python scripts/build_index.py \
    --chunks data/musique/chunks.json \
    --output data/musique/index \
    --model Qwen/Qwen3-Embedding-0.6B \
    --device cpu  # hoặc cuda:0
```

### Chạy MVP 20 câu (smoke test)
```bash
# Step 1: 3 câu để kiểm tra pipeline
python scripts/thesis/run_variants.py \
    --config configs/thesis/musique_base.yaml \
    --variant arag_baseline \
    --questions data/musique/questions.json \
    --output results/thesis/arag_baseline_musique \
    --limit 3 --workers 1 --verbose

# Step 2: 20 câu tất cả variants
for variant in naive_rag arag_baseline arag_entity_tracker arag_evidence_checker arag_entity_evidence_full; do
    python scripts/thesis/run_variants.py \
        --config configs/thesis/musique_base.yaml \
        --variant $variant \
        --questions data/musique/questions.json \
        --output results/thesis/${variant}_musique \
        --limit 20 --workers 3
done

# Step 3: So sánh kết quả
python scripts/thesis/compare_results.py \
    --results results/thesis/ \
    --dataset musique
```

### Chạy evaluation (LLM-Accuracy)
```bash
python scripts/eval.py \
    --predictions results/thesis/arag_baseline_musique/predictions.jsonl \
    --config configs/thesis/musique_base.yaml \
    --workers 5
```

## Cách Chạy Colab

1. Mở `notebooks/thesis_colab.ipynb` trên Google Colab
2. Chạy Cell 1: Clone repo + checkout branch
3. Chạy Cell 2: Cài môi trường
4. Chạy Cell 3: Mount Google Drive
5. **Set API key trong Cell 2** (nhập vào Colab Secret hoặc set ENV)
6. Chạy Cell 4: Build index trên GPU (nhanh hơn CPU ~5x)
7. Chạy Cells 5–9: Các variants (mỗi cell 20 câu)
8. Chạy Cell 10: In bảng so sánh
9. Chạy Cell 11: Copy results về Drive

## Cách Đọc Kết Quả

File `predictions.jsonl`: mỗi dòng là 1 câu hỏi với các trường:
- `pred_answer`: câu trả lời của model
- `gold_answer`: đáp án đúng
- `loops`: số vòng lặp ReAct
- `total_retrieved_tokens`: tổng tokens đã truy xuất
- `total_cost`: chi phí USD
- `entity_count`: số thực thể đã track (chỉ có ở ET variants)
- `verification`: kết quả kiểm chứng (chỉ có ở EV variants)

File `comparison_musique.json`: bảng tổng hợp tất cả metrics sau khi chạy `compare_results.py`.

### Metrics quan trọng
| Metric | Mô tả |
|--------|-------|
| `contain_acc` | Contain-Match Accuracy (không cần LLM) |
| `avg_loops` | Số vòng lặp trung bình (thấp hơn = hiệu quả hơn) |
| `avg_cost_usd` | Chi phí trung bình mỗi câu |
| `avg_entity_count` | Số thực thể track trung bình (ET variants) |
| `avg_coverage_score` | Coverage bằng chứng (EV variants, 0.0–1.0) |

Để tính LLM-Accuracy chính xác hơn, chạy thêm `scripts/eval.py`.

## Datasets

| Dataset | Questions | Chunks | Multi-hop |
|---------|-----------|--------|-----------|
| MuSiQue | 1,000 | 1,354 | 2–4 hop |
| HotpotQA | 1,000 | 1,311 | 2 hop |

## Thứ Tự Chạy Thực Nghiệm

1. **MVP**: 20 câu MuSiQue × 5 variants → kiểm tra pipeline
2. **Mở rộng**: 100 câu MuSiQue × 5 variants → kết quả sơ bộ
3. **Full**: 500 câu MuSiQue + 500 câu HotpotQA × 5 variants → kết quả luận văn
4. **LLM-Accuracy**: Chạy `eval.py` trên full results → bảng so sánh chính thức
