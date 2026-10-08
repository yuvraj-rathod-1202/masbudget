# Dataset format (JSON)

One JSON object with a `tasks` list. Each task becomes one agent's task t_i.
Example: [data/example_tasks.json](../data/example_tasks.json) (synthetic, for smoke tests only).

```json
{
  "name": "llmrouterbench_subset",
  "description": "optional free text",
  "tasks": [
    {
      "task_id": "humaneval/12",
      "difficulty": "hard",
      "category": "code_generation",
      "quality": {
        "gpt-4o-mini": 0.31,
        "claude-3.5-sonnet": 0.64,
        "o1-reasoning": 0.88
      },
      "features": [0.12, -0.53, 0.07],
      "metadata": {"source": "HumanEval"}
    }
  ]
}
```

| Field | Required | Type | Meaning |
| --- | --- | --- | --- |
| `task_id` | yes | string | Unique across the file. |
| `difficulty` | yes | `"easy"`, `"medium"` or `"hard"` | Used by difficulty-based arrival policies (`Task.difficulty_rank`: 0/1/2). |
| `quality` | yes | object: model name → number in [0, 1] | Expected quality q_{i,m}. **Must contain every model named in the experiment's `models`** (extra models are fine). |
| `category` | no | string | Task type. |
| `features` | no | list of numbers | Task embedding / features (e.g. for `predictive_router`). |
| `metadata` | no | object | Anything else; available as `Task.metadata`. |

Notes:
- Model **costs are not in the dataset**. They come from the config's `models` block, so the
  same dataset can be run under different pricing.
- The fallback model (no model assigned) always has quality 0 and cost 0; don't list it.
- For difficulty-mix experiments (uniform, easy-heavy 80/20, hard-heavy 20/80) create one
  dataset file per mix and point each experiment at it.
- The loader validates the file and reports the offending task index on error.
