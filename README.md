# tokenizer-lp

Train and evaluate tokenizers from a directory of UTF-8 text files:

```bash
uv run tokenizer-lp-train --data-dir ./texts --vocab-size 8192 --kind both --output-dir ./runs
```

The LP trainer uses SciPy HiGHS on CPU. The trained LP tokenizer is encoded with an exact dynamic-programming shortest-path tokenizer over the selected vocabulary. The BPE trainer implements the standard greedy merge loop.

Add same-colour byte-boundary cuts iteratively:

```bash
uv run tokenizer-lp-train \
  --data-dir ./texts \
  --vocab-size 8192 \
  --kind lp \
  --lp-cut-rounds 5 \
  --lp-cuts-per-round 500 \
  --lp-cut-families boundary,word_packing,global_token_packing,global_pair_packing,path_config,path_multicover,window_overlap,window_overlap_deep,window_pair \
  --lp-solution-cache-dir /tmp/tokenizer_lp_solution_cache
```

Each LP iteration logs the relaxed token lower bound, its implied bytes/token upper bound, and the rounded DP tokenizer compression rate.

Use the LP DP tokenizer directly:

```python
from tokenisation_lp.dp_tokenizer import LpDpTokenizer

tok = LpDpTokenizer.from_file("./runs/lp/lp_dp_tokenizer.json")
print(tok.encode("hello world").tokens)
print(tok.count_tokens_batch(["hello", "world"], num_workers=2))
```
