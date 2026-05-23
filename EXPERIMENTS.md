# Experiments

All runs used `~/Desktop/books/bl.txt`, copied from `10.9.0.4:~/Desktop/books`, unless noted otherwise.

## Smoke Corpus

Tiny two-file corpus in `/tmp/tokenizer_lp_smoke`, with `--vocab-size 280 --max-token-length 8`.

| Setup | Tokens | Bytes/token | Notes |
|---|---:|---:|---|
| LP, bytelevel pretok | 33 | 2.7576 | Initial CPU LP smoke. |
| BPE, bytelevel pretok | 32 | 2.8438 | BPE won this tiny smoke by 1 token. |
| LP exact DP, bytelevel pretok | 33 | 2.7576 | DP matched the relaxed bound exactly on this toy corpus. |

## Full Book Baselines

Common settings:

```bash
--data-dir ~/Desktop/books
--vocab-size 512
--max-token-length 8
--min-token-count 5
```

| Setup | Pretokenizer | LP Solve Time | Tokens | Bytes/token | Result |
|---|---|---:|---:|---:|---|
| LP, top-k rounding, Unigram runtime | bytelevel | 208.318s | 278,688 | 2.1193 | LP beat BPE by 182 tokens. |
| Greedy BPE | bytelevel | n/a | 278,870 | 2.1179 | Baseline. |
| LP, top-k rounding, Unigram runtime | nanochat | 209.306s | 268,378 | 2.2007 | LP beat BPE by 328 tokens. |
| Greedy BPE | nanochat | n/a | 268,706 | 2.1980 | Nanochat improved both tokenizers. |
| LP, top-k rounding, exact DP runtime | nanochat | 200.558s | 268,378 | 2.2007 | Exact DP matched prior Unigram count on this run. |

## LP Relaxation Bound

Run:

```bash
uv run tokenizer-lp-train \
  --data-dir ~/Desktop/books \
  --vocab-size 512 \
  --kind both \
  --output-dir /tmp/tokenizer_lp_books_nanochat_bound_out \
  --max-token-length 8 \
  --min-token-count 5 \
  --pretokenizer nanochat \
  --eval-workers 2
```

| Metric | Value |
|---|---:|
| LP relaxed lower bound | 258,417.000 tokens |
| LP implied bytes/token upper bound | 2.2855 |
| LP rounded exact-DP tokens | 268,378 |
| LP rounded exact-DP bytes/token | 2.2007 |
| Greedy BPE tokens | 268,706 |
| Greedy BPE bytes/token | 2.1980 |
| LP rounded gap vs relaxation | 3.7116% |

LP beat BPE by 328 tokens, while the relaxation showed roughly 3.7% headroom relative to rounded LP.

## Same-Colour Byte-Boundary Cuts

Cut family:

```text
sum_{e=(i,j), color(e)=tau, i <= b < j} f[e] <= t[tau]
```

Full book run:

```bash
uv run tokenizer-lp-train \
  --data-dir ~/Desktop/books \
  --vocab-size 512 \
  --kind lp \
  --output-dir /tmp/tokenizer_lp_books_nanochat_cuts_out \
  --max-token-length 8 \
  --min-token-count 5 \
  --pretokenizer nanochat \
  --eval-workers 2 \
  --lp-cut-rounds 5 \
  --lp-cuts-per-round 500
```

| Iteration | Cuts Added | Max Violation | LP Bound Tokens | Bound Bytes/token | Rounded Tokens | Rounded Bytes/token | Rounded Gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 23 | 0.5 | 258,417.000 | 2.2855 | 268,378 | 2.2007 | 3.7116% |
| 1 | 0 | 0 | 258,420.750 | 2.2855 | 268,378 | 2.2007 | 3.7102% |

The cuts tightened the LP bound by 3.75 tokens but did not change the rounded DP tokenizer. The loop terminated early after iteration 1 because no violated cuts remained.

