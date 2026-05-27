# tokenizer-lp

Train and evaluate tokenizers from a directory of UTF-8 text files:

```bash
uv run tokenizer-lp-train --data-dir ./texts --vocab-size 8192 --kind both --output-dir ./runs
```

The LP trainer uses HiGHS on CPU through the `highspy` Python package by default. The trained LP tokenizer is encoded with an exact dynamic-programming shortest-path tokenizer over the selected vocabulary. The BPE trainer implements the standard greedy merge loop.

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

Each LP iteration logs the relaxed token lower bound, its implied bytes/token upper bound, and the rounded DP tokenizer compression rate. With `--lp-solver highspy`, the iterative cut loop reuses one dual-simplex model and adds cut rows in place, so later rounds can warm start from the previous basis. Use `--lp-solver scipy` to run the older stateless SciPy `linprog` backend.

LP runs are resumable when `--output-dir` is set. The trainer writes active cuts as a compressed sparse matrix, cut de-duplication keys, iteration metadata, the latest raw LP solution vector, candidate LP values, final candidates, separator cache state, and the HiGHS simplex basis under `./runs/lp/training_state/`. Re-running the same command resumes from that checkpoint automatically; pass `--no-resume` to ignore it and start fresh.

Use the LP DP tokenizer directly:

```python
from tokenisation_lp.dp_tokenizer import LpDpTokenizer

tok = LpDpTokenizer.from_file("./runs/lp/lp_dp_tokenizer.json")
print(tok.encode("hello world").tokens)
print(tok.count_tokens_batch(["hello", "world"], num_workers=2))
```

## Experiment Log

### 2026-05-28: Resumed 512-token books LP with direct templates

Run directory:
`/tmp/tokenizer_lp_books_512_reduced_random_8xpool_50kcuts_sigproj_10rounds`

Log:
`/tmp/tokenizer_lp_books_512_reduced_random_8xpool_50kcuts_sigproj_10rounds/run_templates_30k.log`

Command shape:

```bash
uv run tokenizer-lp-train \
  --data-dir /Users/alex/Desktop/books \
  --output-dir /tmp/tokenizer_lp_books_512_reduced_random_8xpool_50kcuts_sigproj_10rounds \
  --kind lp \
  --vocab-size 512 \
  --pretokenizer nanochat \
  --min-token-count 5 \
  --max-token-length 8 \
  --lp-solver highspy \
  --lp-cut-rounds 1000 \
  --lp-cuts-per-round 30000 \
  --lp-cut-families short_word_pair_single_chain,short_word_pair_bridge_chain,short_word_triple_triangle,short_word_triple_4cycle \
  --lp-run-all-cut-families \
  --lp-short-word-pair-template-candidate-word-multiplier 8.0 \
  --lp-short-word-pair-template-top-supports-per-shape 64 \
  --lp-short-word-pair-template-max-chain-edges 6 \
  --lp-short-word-pair-template-max-cuts 30000 \
  --lp-short-word-triple-template-candidate-word-multiplier 8.0 \
  --lp-short-word-triple-template-top-supports-per-shape 64 \
  --lp-short-word-triple-template-max-cuts 30000
```

Partial results:

- Resumed from the previous big run at iteration 29 with 37,685 active cuts.
- This run uses direct templates only: pair single-chain, pair bridge-chain, triplet triangle, and triplet 4-cycle. The reduced pair/triple LP brute-force separators are not enabled.
- `--lp-run-all-cut-families` runs every requested family in each separation pass. Each family is independently capped to the top `--lp-cuts-per-round` candidates before global selection, and the global round cap is also 30,000 added cuts.
- Checkpoint through iteration 40: active cuts reached 361,921 before the iteration 40 separation; iteration 40 then found 1,405 further template cuts, bringing the checkpoint to 363,326 active cuts at `next_iteration=41`.
- Bounds improved from iteration 29 objective 259,113.304 to iteration 40 objective 259,310.616. The rounded tokenizer around iteration 40 used 260,471 tokens on the books corpus, with a rounded gap of 0.4455%.
- Template returns are now dropping sharply. At iteration 40, pair single-chain found 0 cuts, pair bridge-chain found 0, triplet triangle found 869, and triplet 4-cycle found 536.
