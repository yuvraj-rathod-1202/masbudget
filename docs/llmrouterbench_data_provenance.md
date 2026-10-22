# Data Provenance and Audit Report: LLMRouterBench for MASBudget

This document records the provenance, conversion methodology, scientific validation, cost estimation, and exhaustive audit results for the real-benchmark datasets generated for the MASBudget offline multi-agent model selection simulator.

---

## 1. Primary Source Identification

- **Benchmark Name:** LLMRouterBench (Lihao et al., 2025)
- **Official Repository:** [https://github.com/ynulihao/LLMRouterBench](https://github.com/ynulihao/LLMRouterBench)
- **Hugging Face Dataset Card:** [https://huggingface.co/datasets/NPULH/LLMRouterBench](https://huggingface.co/datasets/NPULH/LLMRouterBench)
- **Direct Archive URL:** `https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/main/bench-release.tar.gz?download=true`
- **Archive Size:** 1,283,503,080 bytes (1.28 GB)
- **Archive SHA-256 Digest:**
  `b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1`
- **Local Cache Location:** `~/.cache/huggingface/hub/datasets--NPULH--LLMRouterBench/blobs/b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1`

---

## 2. Benchmark and Split Selection

- **Selected Benchmark:** `livecodebench`
- **Selected Split:** `test`
- **Rationale:**
  The MASBudget problem statement specifically focuses on software engineering and code generation tasks under multi-agent budget constraints (cf. `project_proposal/problem_statement.md` and `project_proposal/experiments_plan.md`). LiveCodeBench evaluates generated code against execution unit-test suites, providing a rigorous, verifiable code benchmark. Furthermore, the LLMRouterBench archive contains evaluation files for 38 distinct models evaluated on the exact same 1,055 instances of `livecodebench/test`.

---

## 3. Model Pool Selection & Consistency

### Official Model Pool Context
The official LLMRouterBench documentation distinguishes two experimental routing regimes:
- **Performance Setting:** Evaluates routing quality irrespective of token cost (often includes local open-weight models evaluated without commercial API pricing metadata).
- **Performance-Cost Setting:** Evaluates cost-performance trade-offs for commercial/flagship models where per-request token pricing is actively tracked.

### Replacement of `Llama-3.1-8B-Instruct`
In earlier iterations, `Llama-3.1-8B-Instruct` was evaluated as a low-cost baseline. However, in LLMRouterBench it belongs to the Performance pool and had incomplete token billing representation relative to flagship models. Following Section 1 requirements, it was replaced by **`deepseek-v3.1-terminus`**, an eligible flagship model from the Performance-Cost regime with complete evaluation coverage (1,055 tasks) and realistic token-based API cost tracking.

### Selected 4-Model Flagship Pool
We selected four flagship models representing distinct capability and cost tiers across the Performance-Cost spectrum:

1. **`deepseek-v3.1-terminus`** (Cheapest flagship tier):
   - Archive Member: `bench-release/livecodebench/test/deepseek-v3.1-terminus/livecodebench-test-deepseek-v3.1-terminus-20251024_102915.json`
   - Total source records: 1,055
   - Benchmark pass@1 performance: 0.6730 (67.30%)
   - Mean measured cost: \$0.0011249379 / request

2. **`deepseek-v3-0324`** (Mid-low flagship tier):
   - Archive Member: `bench-release/livecodebench/test/deepseek-v3-0324/livecodebench-test-deepseek-v3-0324-20251017_111427.json`
   - Total source records: 1,055
   - Benchmark pass@1 performance: 0.6664 (66.64%)
   - Mean measured cost: \$0.0014471013 / request

3. **`gemini-2.5-flash`** (Mid-high flagship flash tier):
   - Archive Member: `bench-release/livecodebench/test/gemini-2.5-flash/livecodebench-test-gemini-2.5-flash-20251013_003249.json`
   - Total source records: 1,055
   - Benchmark pass@1 performance: 0.6000 (60.00%)
   - Mean measured cost: \$0.0058342806 / request

4. **`deepseek-r1-0528`** (Frontier reasoning flagship tier):
   - Archive Member: `bench-release/livecodebench/test/deepseek-r1-0528/livecodebench-test-deepseek-r1-0528-20251018_090719.json`
   - Total source records: 1,055
   - Benchmark pass@1 performance: 0.8009 (80.09%)
   - Mean measured cost: \$0.0288723124 / request

All four models share the exact same 1,055 source evaluation instances.

---

## 4. Source Data Integrity and Validation

### Exact Metric Preserved
- Source field: `score` inside each object of the top-level `records` list.
- For LiveCodeBench, `score` is a binary float: `1.0` if generated code passed all unit test cases, and `0.0` if it failed.
- The scores are preserved exactly as given; no rounding, clamping, rescaling, or artificial smoothing was applied.

### Duplicate Checks
- Every record file was verified to have strictly unique `index` values ranging from 1 to 1,055.
- Zero duplicate indices were found within any model file.

### Alignment and Intersection
- Tasks were joined on the composite key `(benchmark, split, source_index)`.
- All 4 models had complete coverage of all 1,055 instances.
- Zero tasks were excluded due to missing records; intersection size is exactly 1,055.

---

## 5. Cost Estimation Methodology

In the MASBudget simulator, model costs are configured per-call in YAML (`models: {name: cost}`). In LLMRouterBench, each record includes an observed API invocation `cost` (computed from prompt and completion token counts at standard provider rates).

For each model, we computed descriptive statistics over all 1,055 common instances and saved them to `data/llmrouterbench_livecodebench_model_costs.json`:

| Model | Observations | Mean Cost (USD) | Median Cost (USD) | Min Cost (USD) | Max Cost (USD) |
|---|---|---|---|---|---|
| `deepseek-v3.1-terminus` | 1,055 | \$0.0011249379 | \$0.0005266800 | \$0.0000945600 | \$0.0103390300 |
| `deepseek-v3-0324` | 1,055 | \$0.0014471013 | \$0.0008674900 | \$0.0000789800 | \$0.0141064700 |
| `gemini-2.5-flash` | 1,055 | \$0.0058342806 | \$0.0007185000 | \$0.0000600000 | \$0.0631679000 |
| `deepseek-r1-0528` | 1,055 | \$0.0288723124 | \$0.0255947000 | \$0.0012321500 | \$0.0699975000 |

These mean costs are directly utilized in `configs/llmrouterbench_livecodebench.yaml`.

---

## 6. Task Difficulty Methodology & Experimental Caveats

### Difficulty Assignment
The MASBudget dataset loader requires each task to have a `difficulty` field in `{"easy", "medium", "hard"}`. Because raw benchmark instances in LLMRouterBench do not carry an official tripartite difficulty tag, we derived an **explicit proxy** from observed performance across the four selected models:

1. For each task $i$, compute the mean score $\bar{q}_i = \frac{1}{4} \sum_{m \in \mathcal{M}} q_{i,m}$.
2. Sort tasks descending by $\bar{q}_i$, breaking ties deterministically by task index $i$ ascending.
3. Partition the sorted array into tertiles:
   - Top tertile (ranks 0 to 350, 351 tasks): labeled `"easy"` (highest consensus pass rate)
   - Middle tertile (ranks 351 to 702, 352 tasks): labeled `"medium"`
   - Bottom tertile (ranks 703 to 1054, 352 tasks): labeled `"hard"` (lowest consensus pass rate)

### Scientific Caveats on Outcome-Informed Difficulty
Because this difficulty label is derived from the evaluated model outcomes rather than external task specifications (such as LeetCode problem tags or human complexity ratings):
- **Outcome correlation:** The label reflects consensus empirical solvability under this specific model pool.
- **Advantage to difficulty-aware policies:** Arrival policies that prioritize by difficulty (e.g., `hardest_task_first` or `easiest_task_first`) receive an empirical signal that correlates with actual model outcomes. When evaluating these policies, researchers should interpret the results as an upper bound on what difficulty-aware ordering can achieve, because the difficulty ranking contains information derived from the observed benchmark outcomes.

---

## 7. Difficulty-Mix Datasets

Following the MASBudget experiment specifications, three difficulty-mix datasets were created by deterministic sampling without replacement from the base dataset (seed = 42):

1. **Uniform Mix (`data/llmrouterbench_livecodebench_uniform.json`):**
   - 1,053 tasks total (351 easy, 351 medium, 351 hard; exactly 33.33% each).
   - Draws all available tasks from the limiting bucket (easy: 351) and balanced samples from medium and hard.

2. **Easy-Heavy Mix (`data/llmrouterbench_livecodebench_easy_heavy_80_20.json`):**
   - 400 tasks total.
   - 320 easy tasks (80.0%).
   - 40 medium tasks (10.0%).
   - 40 hard tasks (10.0%).

3. **Hard-Heavy Mix (`data/llmrouterbench_livecodebench_hard_heavy_20_80.json`):**
   - 400 tasks total.
   - 40 easy tasks (10.0%).
   - 40 medium tasks (10.0%).
   - 320 hard tasks (80.0%).

---

## 8. Exhaustive Source-Trace Audit

The converter includes an `--audit` routine that supports full exhaustive verification of all task-model pairs and cost statistics against the original archive.

Exhaustive audit execution:
```bash
python scripts/convert_llmrouterbench_real.py \
    --archive ~/.cache/huggingface/hub/datasets--NPULH--LLMRouterBench/blobs/b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1 \
    --audit data/llmrouterbench_livecodebench.json \
    --audit-full
```

**Audit Results:**
- **Task-Model Pairs Checked:** Exactly **4,220 out of 4,220** (1,055 tasks $\times$ 4 models).
  - `deepseek-v3.1-terminus`: 1,055/1,055 verified
  - `gemini-2.5-flash`: 1,055/1,055 verified
  - `deepseek-r1-0528`: 1,055/1,055 verified
  - `deepseek-v3-0324`: 1,055/1,055 verified
- **Score Mismatches:** 0 (all scores match the source archive records to floating-point precision).
- **Cost Statistics Validation:** 4/4 models validated against per-record costs; min, max, mean, median, and observation count match with zero error.

---

## 9. Reproduction Instructions

To reproduce all datasets and audit from scratch:

```bash
# 1. Convert base dataset and compute costs (requires source archive or extracted dir)
python scripts/convert_llmrouterbench_real.py \
    --archive <PATH_TO_BLOB_OR_TAR_GZ> \
    --benchmark livecodebench --split test \
    --models deepseek-v3.1-terminus deepseek-v3-0324 gemini-2.5-flash deepseek-r1-0528 \
    --output data/llmrouterbench_livecodebench.json

# 2. Generate difficulty mix files
python scripts/convert_llmrouterbench_real.py \
    --create-mixes data/llmrouterbench_livecodebench.json

# 3. Run exhaustive source-trace audit (all 4,220 pairs)
python scripts/convert_llmrouterbench_real.py \
    --archive <PATH_TO_BLOB_OR_TAR_GZ> \
    --audit data/llmrouterbench_livecodebench.json \
    --audit-full

# 4. Run test suite
python -m pytest tests/test_llmrouterbench_conversion.py -v

# 5. Run smoke experiment
python -m masbudget configs/llmrouterbench_livecodebench.yaml --only smoke_livecodebench_mckp_vs_dsp
```
