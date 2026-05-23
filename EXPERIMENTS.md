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

## Fractional LP Diagnostics

Diagnostic command:

```bash
uv run python -m tokenisation_lp.analyze_fractional_lp \
  --data-dir ~/Desktop/books \
  --vocab-size 512 \
  --max-token-length 8 \
  --min-token-count 5 \
  --pretokenizer nanochat \
  --top 20
```

The base relaxation had 288 fractional token-selection variables and 47,517 fractional non-free edge variables. Most high-ranked fractional values were exactly 0.5.

Useful violated cut families found:

| Family | Violations | Best Violation | Example |
|---|---:|---:|---|
| Same-colour byte-boundary | 23 | 0.5 | repeated newlines with token `ĊĊĊĊ` |
| Same-colour word-packing | 3 | 1.0 | `word='âĢľWooooow'`, token `oo`, `lhs=2`, `max_pack * t = 1` |
| Aggregate boundary activation | 23 | 0.5 | reduced to single-colour boundary cases in this run |

The word-packing cut is:

```text
sum_{e in occurrences(word, tau)} f[e] <= max_pack(word, tau) * t[tau]
```

where `max_pack` is the maximum number of non-overlapping occurrences of token colour `tau` in the pretokenized word.

## Boundary + Word-Packing Cuts

Run:

```bash
uv run tokenizer-lp-train \
  --data-dir ~/Desktop/books \
  --vocab-size 512 \
  --kind lp \
  --output-dir /tmp/tokenizer_lp_books_wordpack_out \
  --max-token-length 8 \
  --min-token-count 5 \
  --pretokenizer nanochat \
  --eval-workers 2 \
  --lp-cut-rounds 5 \
  --lp-cuts-per-round 500 \
  --lp-cut-families boundary,word_packing
```

| Iteration | Cuts Added | Max Violation | LP Bound Tokens | Bound Bytes/token | Rounded Tokens | Rounded Bytes/token | Rounded Gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 26 | 1.0 | 258,417.000 | 2.2855 | 268,378 | 2.2007 | 3.7116% |
| 1 | 0 | 0 | 258,420.750 | 2.2855 | 268,378 | 2.2007 | 3.7102% |

The combined separator found the 23 boundary cuts plus 3 word-packing cuts. The final bound matched the boundary-only run, indicating these packing cuts were implied by the boundary cuts after re-solving on this corpus.

## Path-Configuration Cuts And LP Cache

Path-configuration cuts were added in conservative hitting-set form. For a word and integer threshold `K`, enumerate all complete paths using at most `K` tokens. If `S` is a hitting set of token colours that intersects every such short path, add:

```text
word_cost + K * sum_{tau in S} t[tau] >= K + 1
```

This is valid for the ILP: if no token in `S` is active, every path with `<= K` tokens is blocked, so the word needs at least `K + 1` tokens. If some token in `S` is active, the RHS relaxes to at most `1`, which every non-empty word path satisfies.

A synthetic separator check produced the expected row:

```text
path_config key=('path_config', 0, 1, (0,))
row: -f0 -g0 -g1 -t0 <= -2
```

Full book run:

```bash
uv run tokenizer-lp-train \
  --data-dir ~/Desktop/books \
  --vocab-size 512 \
  --kind lp \
  --output-dir /tmp/tokenizer_lp_books_pathconfig_out \
  --max-token-length 8 \
  --min-token-count 5 \
  --pretokenizer nanochat \
  --eval-workers 2 \
  --lp-cut-rounds 5 \
  --lp-cuts-per-round 500 \
  --lp-cut-families boundary,word_packing,path_config \
  --lp-solution-cache-dir /tmp/tokenizer_lp_solution_cache
```

| Iteration | Cuts Added | Max Violation | LP Bound Tokens | Rounded Tokens | Rounded Gap |
|---:|---:|---:|---:|---:|---:|
| 0 | 26 | 1.0 | 258,417.000 | 268,378 | 3.7116% |
| 1 | 0 | 0 | 258,420.750 | 268,378 | 3.7102% |

The path-configuration separator did not find extra violated cuts beyond the 23 boundary + 3 word-packing cuts on this corpus. Re-running the identical command with `--lp-solution-cache-dir /tmp/tokenizer_lp_solution_cache` hit both cached LP solves:

```text
iteration 0 cache hit solve time: 0.184s
iteration 1 cache hit solve time: 0.170s
```

## Path-Multicover Cuts

Path-multicover cuts generalize the hitting-set path cut. For a word and threshold `K`, if every complete path with at most `K` tokens uses at least `r` token colours from a set `S`, add:

```text
word_cost + (K / r) * sum_{tau in S} t[tau] >= K + 1
```

This is ILP-valid: an integral short path activates at least `r` tokens in `S`; otherwise the word must use at least `K + 1` tokens.

A synthetic separator check produced a violated row:

```text
path_multicover key=('path_multicover', 0, 2, 1, (1,))
row: -f0 -f1 -g0 -g1 -g2 -2*t1 <= -3
```

Full book run with all families:

```bash
uv run tokenizer-lp-train \
  --data-dir ~/Desktop/books \
  --vocab-size 512 \
  --kind lp \
  --output-dir /tmp/tokenizer_lp_books_multicover_out \
  --max-token-length 8 \
  --min-token-count 5 \
  --pretokenizer nanochat \
  --eval-workers 2 \
  --lp-cut-rounds 5 \
  --lp-cuts-per-round 500 \
  --lp-cut-families boundary,word_packing,path_config,path_multicover \
  --lp-solution-cache-dir /tmp/tokenizer_lp_solution_cache
```

The run loaded cached LP solutions. It still found only the same 26 first-round cuts and no second-round cuts:

| Iteration | Cuts Added | Max Violation | LP Bound Tokens | Rounded Tokens | Rounded Gap |
|---:|---:|---:|---:|---:|---:|
| 0 | 26 | 1.0 | 258,417.000 | 268,378 | 3.7116% |
| 1 | 0 | 0 | 258,420.750 | 268,378 | 3.7102% |

So the current multicover separator did not expose additional full-book violations beyond boundary and word-packing cuts.

## Weighted Global Token-Packing Cuts

Global token-packing cuts aggregate same-token packing across all pretokenized word types with corpus frequencies:

```text
sum_w freq[w] * sum_{e in occurrences(w, tau)} f[e]
  <= sum_w freq[w] * max_pack(w, tau) * t[tau]
```

This is just the weighted sum of same-colour word-packing inequalities for a single token colour, but it is compact and objective-weighted.

A synthetic separator check produced:

```text
global_token_packing key=('global_token_packing', 0)
row: 10*f0 + 10*f1 + f2 - 10*t0 <= 0
```

Full book run with only `global_token_packing`:

| Iteration | Cuts Added | Max Violation | LP Bound Tokens | Rounded Tokens | Rounded Gap |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 25.5 | 258,417.000 | 268,378 | 3.7116% |
| 1 | 0 | 0 | 258,417.750 | 268,378 | 3.7113% |

Full book run with all current cut families:

```bash
uv run tokenizer-lp-train \
  --data-dir ~/Desktop/books \
  --vocab-size 512 \
  --kind lp \
  --output-dir /tmp/tokenizer_lp_books_allcuts_out \
  --max-token-length 8 \
  --min-token-count 5 \
  --pretokenizer nanochat \
  --eval-workers 2 \
  --lp-cut-rounds 5 \
  --lp-cuts-per-round 500 \
  --lp-cut-families boundary,word_packing,global_token_packing,path_config,path_multicover \
  --lp-solution-cache-dir /tmp/tokenizer_lp_solution_cache
```

| Iteration | Cuts Added | Max Violation | LP Bound Tokens | Rounded Tokens | Rounded Gap |
|---:|---:|---:|---:|---:|---:|
| 0 | 27 | 25.5 | 258,417.000 | 268,378 | 3.7116% |
| 1 | 0 | 0 | 258,420.750 | 268,378 | 3.7102% |

The global weighted cut was present in round 0, but the final bound still matched the local boundary + word-packing bound.

## Small-Rank Window-Overlap Cuts

Window-overlap cuts implement local subset-capacity separation:

1. Rank suspicious word types by weighted fractional edge mass.
2. Scan small byte windows in those words.
3. Pick the top fractional token colours in the window.
4. Enumerate every subset of those colours and compute the maximum compatible non-overlapping clipped window coverage for that subset.
5. Fit a modular upper bound on that subset-capacity function at the current fractional `t`.
6. Add:

```text
sum_{e in window, color(e) in C} overlap(e, window) * f[e]
  <= alpha + sum_{tau in C} beta_tau * t[tau]
```

Full book run with only `window_overlap`:

```bash
uv run tokenizer-lp-train \
  --data-dir ~/Desktop/books \
  --vocab-size 512 \
  --kind lp \
  --output-dir /tmp/tokenizer_lp_books_window_out \
  --max-token-length 8 \
  --min-token-count 5 \
  --pretokenizer nanochat \
  --eval-workers 2 \
  --lp-cut-rounds 2 \
  --lp-cuts-per-round 500 \
  --lp-cut-families window_overlap \
  --lp-solution-cache-dir /tmp/tokenizer_lp_solution_cache
```

| Iteration | Cuts Added | Max Violation | LP Bound Tokens | Rounded Tokens | Rounded Gap |
|---:|---:|---:|---:|---:|---:|
| 0 | 12 | 1.0 | 258,417.000 | 268,378 | 3.7116% |
| 1 | 0 | 0 | 258,417.167 | 268,239 | 3.6616% |

These cuts barely tightened the bound, but changed rounding and improved the rounded tokenizer by 139 tokens.

Full book run with every implemented family:

```bash
uv run tokenizer-lp-train \
  --data-dir ~/Desktop/books \
  --vocab-size 512 \
  --kind lp \
  --output-dir /tmp/tokenizer_lp_books_all_window_out \
  --max-token-length 8 \
  --min-token-count 5 \
  --pretokenizer nanochat \
  --eval-workers 2 \
  --lp-cut-rounds 5 \
  --lp-cuts-per-round 500 \
  --lp-cut-families boundary,word_packing,global_token_packing,path_config,path_multicover,window_overlap \
  --lp-solution-cache-dir /tmp/tokenizer_lp_solution_cache
```

| Iteration | Cuts Added | Max Violation | LP Bound Tokens | Rounded Tokens | Rounded Gap |
|---:|---:|---:|---:|---:|---:|
| 0 | 39 | 25.5 | 258,417.000 | 268,378 | 3.7116% |
| 1 | 0 | 0 | 258,420.750 | 268,378 | 3.7102% |

With all families enabled, the final rounded tokenizer returned to the prior 268,378-token result; the window cuts were most useful when applied alone.

## Pairwise Window-Conflict Cuts

Note: an earlier pairwise formula was asymmetric and invalid for cases where
`cA != cB`. The implementation was corrected to fit the valid modular upper
bound for the four binary activation subsets. The corrected pairwise family
matches the valid `window_overlap` behaviour on this corpus.

Pairwise window cuts are a cheaper special case of small-rank window overlap. For two colours `A,B` in a local window, compute:

- `cA`: max compatible clipped coverage using only `A`
- `cB`: max compatible clipped coverage using only `B`
- `cAB`: max compatible clipped coverage using either `A` or `B`

Then add:

```text
y_A + y_B <= cAB + (cA-cAB)*(1-t_A) + (cB-cAB)*(1-t_B)
```

Equivalently:

```text
y_A + y_B + (cA-cAB)*t_A + (cB-cAB)*t_B <= cA + cB - cAB
```

Corrected full book run:

```bash
uv run tokenizer-lp-train \
  --data-dir ~/Desktop/books \
  --vocab-size 512 \
  --kind lp \
  --output-dir /tmp/tokenizer_lp_books_windowpair_valid_out \
  --max-token-length 8 \
  --min-token-count 5 \
  --pretokenizer nanochat \
  --eval-workers 2 \
  --lp-cut-rounds 2 \
  --lp-cuts-per-round 500 \
  --lp-cut-families window_pair \
  --lp-solution-cache-dir /tmp/tokenizer_lp_solution_cache
```

| Iteration | Cuts Added | Max Violation | LP Bound Tokens | Rounded Tokens | Rounded Gap |
|---:|---:|---:|---:|---:|---:|
| 0 | 12 | 1.0 | 258,417.000 | 268,378 | 3.7116% |
| 1 | 0 | 0 | 258,417.167 | 268,239 | 3.6616% |

The corrected pairwise cut no longer gives the earlier apparent +7.85-token bound improvement; that result came from the invalid asymmetric formula and should not be used.

## Deep Window-Overlap Search

The deeper valid window search raises the suspicious-word scan from 250 to 1000, the colour rank from 4 to 6, and scans window lengths `(4,5,6,8,10,12,16)`.

```bash
uv run tokenizer-lp-train \
  --data-dir ~/Desktop/books \
  --vocab-size 512 \
  --kind lp \
  --output-dir /tmp/tokenizer_lp_books_windowdeep_out \
  --max-token-length 8 \
  --min-token-count 5 \
  --pretokenizer nanochat \
  --eval-workers 2 \
  --lp-cut-rounds 3 \
  --lp-cuts-per-round 1000 \
  --lp-cut-families window_overlap_deep \
  --lp-solution-cache-dir /tmp/tokenizer_lp_solution_cache
```

| Iteration | Cuts Added | Max Violation | LP Bound Tokens | Rounded Tokens | Rounded Gap |
|---:|---:|---:|---:|---:|---:|
| 0 | 42 | 1.0 | 258,417.000 | 268,378 | 3.7116% |
| 1 | 0 | 0 | 258,417.167 | 268,378 | 3.7115% |

It finds more valid local cuts than the default scan, but the same bound improvement.

## Global Pair-Packing Cuts

Global pair-packing cuts aggregate full-word interval capacities for token-colour pairs across high-fractionality words. For a token pair `(A,B)`, compute a weighted set capacity over each word and fit the valid two-colour modular upper bound globally.

Note: an earlier implementation sorted token-pair keys after computing capacities, which could attach singleton capacities to the wrong token. The corrected implementation keeps the capacity bit order consistent with the token tuple.

Full book run:

```bash
uv run tokenizer-lp-train \
  --data-dir ~/Desktop/books \
  --vocab-size 512 \
  --kind lp \
  --output-dir /tmp/tokenizer_lp_books_globalpair_out \
  --max-token-length 8 \
  --min-token-count 5 \
  --pretokenizer nanochat \
  --eval-workers 2 \
  --lp-cut-rounds 2 \
  --lp-cuts-per-round 500 \
  --lp-cut-families global_pair_packing \
  --lp-solution-cache-dir /tmp/tokenizer_lp_solution_cache
```

Corrected full book result:

| Iteration | Cuts Added | Max Violation | LP Bound Tokens | Rounded Tokens | Rounded Gap |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 87 | 258,417.000 | 268,378 | 3.7116% |
| 1 | 0 | 0 | 258,417.069 | 268,378 | 3.7115% |

The corrected global-pair cut is valid but much weaker than the buggy first result.

## Global Triple-Packing Cuts

The global rank-packing separator was extended from pairs to triples with consistent capacity bit ordering.

```bash
uv run tokenizer-lp-train \
  --data-dir ~/Desktop/books \
  --vocab-size 512 \
  --kind lp \
  --output-dir /tmp/tokenizer_lp_books_globaltriple_out \
  --max-token-length 8 \
  --min-token-count 5 \
  --pretokenizer nanochat \
  --eval-workers 2 \
  --lp-cut-rounds 2 \
  --lp-cuts-per-round 500 \
  --lp-cut-families global_triple_packing \
  --lp-solution-cache-dir /tmp/tokenizer_lp_solution_cache
```

| Iteration | Cuts Added | Max Violation | LP Bound Tokens | Rounded Tokens | Rounded Gap |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 15 | 258,417.000 | 268,378 | 3.7116% |
| 1 | 0 | 0 | 258,417.000 | 268,378 | 3.7116% |

The triple cut was violated at the base point but did not move the objective by itself.
