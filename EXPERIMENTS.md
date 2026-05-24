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

## Residual Fractionality and Additional Cut Search

After adding the known valid same-colour byte-boundary and word-packing cuts on `~/Desktop/books`, the remaining solution is still highly fractional:

| Quantity | Value |
|---|---:|
| LP bound | 258,420.750 tokens |
| Positive token-selection variables | 399 |
| Fractional token-selection variables | 288 |
| Sum of token-selection variables | 254.000 |
| Fractional non-free edge variables | 47,730 |
| Fractional free edge variables | 20,820 |

The fractional token values are almost entirely half-integral:

| Fractional `t` value | Count |
|---:|---:|
| 0.50 | 278 |
| 0.25 | 7 |
| 0.75 | 3 |

Representative high-weight residual words look like mixtures of two equally good segmentations:

```text
word='Ġthere', freq=205, cost=2.0
  0.5 * ['Ġth', 'ere']  with t('Ġth')=1.0, t('ere')=0.5
  0.5 * ['Ġthe', 're']  with t('Ġthe')=1.0, t('re')=0.5

word='Ġyour', freq=240, cost=1.5
  0.5 * ['Ġyou', 'r']   with t('Ġyou')=1.0, byte fallback 'r'
  0.5 * ['Ġyour']       with t('Ġyour')=0.5
```

For a single word, these are not necessarily cuttable by a valid linear local inequality: the point can be a convex combination of two valid integral segmentations. This is why many local packing families either find nothing after the boundary cleanup or move only rounding/tie-breaking.

### New Families Tried

I implemented and tested these additional conservative families:

- `word_rank_count`: full-word small-rank interval capacity using edge count.
- `word_rank_length`: full-word small-rank interval capacity using covered byte length.
- `global_rank_count`: weighted corpus-level small-rank count-capacity cuts.
- `path_min_cover`: exact minimum-weight hitting-set version of the short-path cover cut.
- `group_value`: small token-set lower-envelope cuts over groups of words. For a token set `S`, it computes `F(U)` as the minimum weighted token count with `U subset S` active and every token outside `S` allowed, then adds an affine lower bound on `F`. This is valid because allowing all outside tokens can only make the bound weaker.

Separation at the base LP:

| Family | Cuts Found | Max Violation |
|---|---:|---:|
| `boundary` | 23 | 0.5 |
| `word_packing` | 3 | 1 |
| `global_token_packing` | 1 | 25.5 |
| `global_pair_packing` | 1 | 87 |
| `global_triple_packing` | 1 | 15 |
| `global_rank_count` | 2 | 10.5 |
| `word_rank_count` | 0 | 0 |
| `word_rank_length` | 2 | 1 |
| `path_config` | 0 | 0 |
| `path_multicover` | 0 | 0 |
| `path_min_cover` | 0 | 0 |
| `group_value` | 0 | 0 |
| `window_overlap` | 12 | 1 |
| `window_overlap_deep` | 42 | 1 |
| `word_path_cover` | 0 | 0 |
| `window_pair` | 12 | 1 |

Separation after the 26 boundary + word-packing cleanup cuts:

| Family | Cuts Found | Max Violation |
|---|---:|---:|
| all implemented families above | 0 | 0 |

Bound-moving runs for the new families:

| Family | Bound After Cuts | Bound Gain | Rounded Tokens | Notes |
|---|---:|---:|---:|---|
| `word_rank_count` | 258,417.000 | 0 | 268,378 | No cuts found at base. |
| `word_rank_length` | 258,417.100 | +0.100 | 266,972 | Valid but only changes rounding/tie-breaking materially. |
| `global_rank_count` | 258,417.000 | 0 | 268,378 | Two violated cuts, no objective movement. |
| `path_min_cover` | 258,417.000 | 0 | 268,378 | No cuts found. |
| `group_value` | 258,417.000 | 0 | 268,378 | No cuts found with rank-8 candidate sets. |
| `boundary,word_packing,word_rank_length` | 258,420.750 | +3.750 | 268,378 | Same bound as boundary + word-packing alone. |

The `word_rank_length` rounded result is not evidence of a stronger relaxation; the LP bound only improves by 0.100 token, so the rounded tokenizer changed because the tiny perturbation changed token ordering around fractional ties.

### Validity Checks and Rejected Ideas

The large-looking base violations for `global_pair_packing`, `global_rank_count`, and related cuts are not enough by themselves. After solving with the cuts, the bound movement is tiny or zero. I treated earlier big jumps as suspect until the formulas were checked against all binary activation subsets; the previous asymmetric pair formula and the old global pair bit-ordering bug were invalid and should remain discarded.

The tempting "conflicting edges <= max(active colour)" cut is not useful as a linear LP cut in this formulation. For a clique of overlapping intervals, `sum f_e <= 1` is already implied by the word flow across the shared byte boundary. A max activation RHS would require an auxiliary `m = max(t_i)`. The convex linear relaxation has `m >= t_i`, and because `m` only appears on the RHS, the LP can raise `m` to satisfy the cut. Without a nonconvex equality or an objective penalty on `m`, it collapses to the existing boundary/flow information.

The remaining gap appears to be a global decomposition issue: many words are locally convex combinations of valid paths, and the token budget polytope itself is integral, but the LP can use different fractional path mixtures for different words under the same half-integral token vector. Local interval-packing, same-colour, small-window, and small-rank value cuts do not detect an inconsistency after the boundary cleanup.

Potential next directions that are more likely to tighten the relaxation:

- **Global scenario/decomposition cuts:** try to prove that the current `(t, f)` cannot be decomposed into a convex combination of a small number of integral vocabularies and their induced shortest paths.
- **Column-generation over vocabularies:** solve a master problem over integral token sets/tokenizers for the current fractional support; violated decomposition constraints would be directly meaningful.
- **Larger group value cuts:** increase `group_value` rank and construct token sets by clustering half-integral tokens globally rather than seeding from one word. This is valid but may get expensive quickly.
- **Branch-and-cut style local branching:** temporarily branch on a cluster of half-integral tokens, solve both child LPs, and derive disjunctive cuts if both children imply a higher bound. This is stronger than modular local cuts but substantially more complex.

## Split Branching Bounds

I added a diagnostic branch tool:

```bash
uv run tokenizer-lp-branch-split \
  --data-dir ~/Desktop/books \
  --vocab-size 512 \
  --pretokenizer nanochat \
  --min-token-count 5 \
  --max-token-length 8 \
  --cut-rounds 1 \
  --cuts-per-round 500 \
  --cut-families boundary,word_packing \
  --max-candidates 30 \
  --max-nodes 40 \
  --max-depth 3 \
  --incumbent-tokens 266972
```

The tool starts from a live HiGHS basis, adds the cleanup cuts, and then changes token activation bounds in place. For a binary token variable `t_i`, the split disjunction is:

```text
t_i = 0  OR  t_i = 1
```

If the child LP bounds are `L0` and `L1`, then every integral tokenizer must have objective at least `min(L0, L1)`. Equivalently, the branch proof can add the ILP-valid objective cutoff:

```text
c^T x >= min(L0, L1)
```

This is not a local packing inequality, but it is a valid split cut for the integer problem.

Book run results:

| Bound Source | Certified Bound | Gain vs Cleanup LP |
|---|---:|---:|
| Cleanup LP (`boundary,word_packing`) | 258,420.750 | 0 |
| Best one-token split | 258,550.000 | +129.250 |
| Depth-3 adaptive split tree | 258,698.500 | +277.750 |

Best one-token splits from the 30-token sweep:

| Token | Root `t` | Child `t=0` Bound | Child `t=1` Bound | Split Bound |
|---|---:|---:|---:|---:|
| `me` | 0.5 | 258,598.750 | 258,550.000 | 258,550.000 |
| `hi` | 0.5 | 258,637.500 | 258,542.500 | 258,542.500 |
| `ou` | 0.5 | 258,562.250 | 258,535.250 | 258,535.250 |
| `ng` | 0.5 | 258,530.750 | 258,535.500 | 258,530.750 |
| `Ġm` | 0.5 | 258,864.000 | 258,525.000 | 258,525.000 |

The shallow branch tree branched adaptively on high-impact fractional tokens and fully closed the depth-3 tree in this run. The correct certified bound is the minimum leaf bound, not the maximum leaf bound; an earlier smoke check caught this reporting pitfall. The depth-3 result therefore certifies `258,698.500`, not the largest observed leaf bound.

This is the first tested direction after the local cuts that materially tightens the bound. It supports the diagnosis that the remaining relaxation gap is mostly token-activation integrality, not missing local interval packing.

## Conflict Clique and Odd-Cycle Cuts

I also tested clique and odd-cycle cuts of the form:

```text
sum_{e in clique} x_e <= 1
sum_{e in odd cycle C} x_e <= floor(|C| / 2)
```

The safe conflict graph here is the per-word interval conflict graph: vertices are concrete token/free-byte interval edges in one pretokenized word, and two vertices conflict when their spans overlap. This is valid because an integral segmentation path can only use non-overlapping intervals.

I did **not** apply these cuts directly to token colours globally. Two token colours are generally allowed to coexist in the same vocabulary, so a global token-colour conflict graph would not be ILP-valid without an additional disjunctive proof.

Book separation result:

| LP Point | Family | Cuts Found | Max Violation |
|---|---|---:|---:|
| Base LP | `conflict_clique` | 0 | 0 |
| Base LP | `conflict_odd_cycle` | 0 | 0 |
| Base LP | `boundary,word_packing` | 26 | 1 |
| After boundary + word-packing cleanup | `conflict_clique` | 0 | 0 |
| After boundary + word-packing cleanup | `conflict_odd_cycle` | 0 | 0 |

This matches the structure of interval graphs. Pairwise overlap graphs of intervals are perfect, so odd-cycle inequalities are implied by clique inequalities. The clique inequalities themselves are already implied by the word flow constraints: every complete path crosses each byte boundary exactly once, so the sum of all interval-edge flow crossing a boundary is already at most/equal to one. The half-integral residual is therefore not an uncut stable-set odd-cycle artifact in the edge-overlap graph.

Conclusion: clique/odd-cycle cuts are valid in the edge-conflict graph, but redundant for this LP. The useful “odd-cycle-like” phenomenon appears to live in token activation/disjunction space, which is why explicit split branching tightened the bound while local conflict graph cuts did not.

## Word Segmentation-Support Cuts

I tested a path-mixture version of the proposed repeated-word support idea.

For one repeated word type `w`, enumerate every segmentation path `p` when the
path count is small enough. For a selected token-colour set `S`, each path has
a required support value:

```text
r(p) = |R(p) cap S|
```

where `R(p)` is the set of non-free token colours used by `p`. In any integral
tokenizer using path `p`:

```text
r(p) <= sum_{tau in S} t_tau
```

To project this to the existing edge-flow variables, solve the small dual:

```text
max   y · f_w + gamma
s.t.  y · incidence(p) + gamma <= r(p)   for every segmentation path p of w
```

Any feasible dual solution gives the valid cut:

```text
y · f_w + gamma <= sum_{tau in S} t_tau
```

The separator only emits a cut if all segmentations of the word were enumerated
exactly. This avoids the common invalid shortcut where a cut is generated from
an incomplete path list.

Book results:

| Cut Families | Final Bound | Gain vs Base | Gain vs Boundary Cleanup | Rounded Tokens |
|---|---:|---:|---:|---:|
| `word_support` | 258,417.750 | +0.750 | n/a | 268,378 |
| `boundary,word_packing` | 258,420.750 | +3.750 | 0 | 268,378 |
| `boundary,word_packing,word_support` | 258,421.500 | +4.500 | +0.750 | 268,378 |

Separated support cuts:

| LP Point | Cuts Found | Max Violation | Example Words |
|---|---:|---:|---|
| Base LP | 5 | 0.5 | `Ġsentence`, `Ġsentences`, `Ġindependent`, `Ġoptimistic`, `Ġfurniture` |
| After boundary + word-packing cleanup | 2 | 0.5 | `Ġoptimistic`, `Ġfurniture` |

This family is valid and matches the intended “average segmentation mixture needs enough colour support” idea. On this corpus it is real but weak: it adds another `+0.750` tokens to the relaxed bound after cleanup. The useful cuts came from relatively low-frequency repeated words, so the objective movement is small.

## Bad-Vocabulary Escape Cuts

I also tried the more literal Benders-like version based on a nearby integral vocabulary. The separator rounds the current fractional colour vector to the top `sum(t)` token colours, computes each word's rounded-vocabulary cost `K`, and enumerates every path with `< K` tokens.

Two variants were tested:

- `bad_vocab_escape`: find a hitting set `H` of all paths better than the rounded vocabulary and add:

  ```text
  word_cost >= K * (1 - sum_{tau in H} t_tau)
  ```

- `bad_vocab_improvement`: for the off-vocabulary escape tokens, compute the set function `cap(U)` = best token-count improvement achievable by a better path whose off-vocab tokens are contained in `U`; add a modular upper bound:

  ```text
  K - word_cost <= alpha + sum beta_tau t_tau
  ```

The second form is stronger because it bounds the amount of improvement, not just whether any better path is possible.

Diagnostics at the base LP showed many words where the rounded vocab is much worse than the fractional LP:

```text
bad rounded words: 5,851
total weighted rounded-vs-LP gap: 33,976.25 token occurrences
```

Examples:

| Word | Freq | Rounded Cost K | LP Word Cost | Gap |
|---|---:|---:|---:|---:|
| `Ġwhen` | 231 | 4 | 1.5 | 2.5 |
| `âĢĻd` | 224 | 4 | 1.5 | 2.5 |
| `Ġmore` | 169 | 4 | 1.5 | 2.5 |
| `Ġalways` | 107 | 5 | 2.0 | 3.0 |

However, the simple hitting-set cut was too weak: for these high-weight words, the current fractional `t` vector already has enough total escape-token support, so the cut is satisfied.

Book results:

| Cut Families | Final Bound | Gain vs Base | Gain vs Boundary Cleanup | Rounded Tokens |
|---|---:|---:|---:|---:|
| `bad_vocab_escape` | 258,417.000 | 0 | n/a | 268,378 |
| `bad_vocab_improvement` | 258,417.750 | +0.750 | n/a | 268,378 |
| `boundary,word_packing,bad_vocab_improvement` | 258,420.750 | +3.750 | 0 | 268,378 |

So the bad-vocabulary Benders idea is valid in this escape-set form, and the improvement-capacity version does cut the base LP. But after the boundary/word-packing cleanup, it does not add anything on this corpus. The stronger `word_support` dual cut remains slightly better because it is derived from the current fractional word-flow mixture rather than from one rounded vocabulary.

## Word-Support Effort Sweep

I added three CLI knobs for the exact `word_support` separator:

- `--lp-word-support-max-words`: number of suspicious word types scanned.
- `--lp-word-support-max-rank`: number of fractional token colours selected for one support cut.
- `--lp-word-support-max-paths`: exact segmentation path enumeration cap per word.

The goal was to check whether cranking CPU finds materially stronger support cuts. All runs below used the books corpus, `vocab_size=512`, nanochat pretokenization, `max_token_length=8`, `min_token_count=5`, HiGHS, and the LP solution cache in `/tmp/tokenizer_lp_solution_cache`.

Full train-loop results with `word_support` alone:

| Setting | Limits `(words, rank, paths)` | Final Bound | Gain vs Base | Cuts by Round | Wall Time |
|---|---:|---:|---:|---|---:|
| small | `(500, 10, 20k)` | 258,417.750 | +0.750 | `5, 1, 0` | 14.73s |
| default | `(2000, 20, 100k)` | 258,417.750 | +0.750 | `5, 1, 0` | 14.70s |
| deep | `(5000, 30, 300k)` | 258,417.750 | +0.750 | `5, 1, 0` | 14.82s |

Full train-loop results with cleanup plus `word_support`:

| Setting | Limits `(words, rank, paths)` | Final Bound | Gain vs Base | Cuts by Round | Wall Time |
|---|---:|---:|---:|---|---:|
| small | `(500, 10, 20k)` | 258,421.500 | +4.500 | `31, 2, 0` | 16.51s |
| default | `(2000, 20, 100k)` | 258,421.500 | +4.500 | `31, 2, 0` | 16.58s |
| deep | `(5000, 30, 300k)` | 258,421.500 | +4.500 | `31, 2, 0` | 16.69s |
| xdeep | `(12000, 50, 1000k)` | 258,421.500 | +4.500 | `31, 2, 0` | 16.64s |

Direct separation timing on cached LP points shows where the CPU goes:

| LP Point | Setting | Limits `(words, rank, paths)` | Cuts | Max Violation | Separator Time |
|---|---|---:|---:|---:|---:|
| Base | tiny | `(100, 8, 5k)` | 0 | 0 | 0.115s |
| Base | small | `(500, 10, 20k)` | 0 | 0 | 0.393s |
| Base | default | `(2000, 20, 100k)` | 5 | 0.5 | 3.723s |
| Base | deep | `(5000, 30, 300k)` | 8 | 1.0 | 7.639s |
| Base | xdeep | `(12000, 50, 1000k)` | 8 | 1.0 | 8.069s |
| Cleanup | tiny | `(100, 8, 5k)` | 0 | 0 | 0.087s |
| Cleanup | small | `(500, 10, 20k)` | 0 | 0 | 0.391s |
| Cleanup | default | `(2000, 20, 100k)` | 2 | 0.5 | 3.722s |
| Cleanup | deep | `(5000, 30, 300k)` | 4 | 1.0 | 7.277s |
| Cleanup | xdeep | `(12000, 50, 1000k)` | 4 | 1.0 | 8.056s |

The direct separator can find a few additional candidate inequalities at deeper settings, but those candidates did not improve the final LP bound on this corpus. The useful bound movement saturates at `+0.750` from `word_support`, and the combined cleanup result saturates at `+4.500`. Past the default setting, extra CPU mostly finds redundant or non-binding cuts.

## Ordered Cut-Family Followup

I tried the remaining proposed cut directions in order on the books corpus with
`vocab_size=512`, nanochat pretokenization, `max_token_length=8`,
`min_token_count=5`, HiGHS, and cached LP solves in
`/tmp/tokenizer_lp_solution_cache`.

### 1. Threshold / Value-Function Cuts

Implemented as `threshold_value`. For one word and selected fractional token
colours `S`, compute the exact value function:

```text
F(U) = shortest segmentation cost when U subset S is active
```

All token colours outside `S` are left unrestricted, so `F` is a valid lower
bound for the full ILP. The separator adds the best affine lower bound on `F`
at the current fractional `t[S]`.

Result: no violated cuts.

| Families | Final Bound | Added vs Base | Wall Time |
|---|---:|---:|---:|
| `threshold_value` | 258,417.000 | 0 | 3.52s |
| `boundary,word_packing,threshold_value` | 258,420.750 | +3.750 | 5.86s |

This suggests the current remaining fractional examples are not explained by a
single-word shortest-cost lower envelope over only the selected colours. The
path-mixture support constraints are the stronger per-word view here.

### 2. Exact Per-Word Path-Hull Support Cuts

Implemented as `word_hull`. For a word and selected colours `S`, enumerate all
segmentation paths and optimize a nonnegative token-support dual:

```text
y · incidence(path) + gamma <= sum_{tau in R(path) cap S} b_tau
b_tau >= 0
sum_tau b_tau <= 1
```

This gives the valid cut:

```text
y · flow + gamma <= b · t
```

The `sum b <= 1` constraint only normalizes the separator; any nonnegative
`b` satisfying the path inequalities gives a valid support cut. The separator
only emits cuts when every path for the word was enumerated.

Result: this is the best new family so far.

| Families | Final Bound | Gain vs Base | Gain vs Boundary Cleanup | Cuts by Round | Wall Time |
|---|---:|---:|---:|---|---:|
| `word_hull` | 258,418.750 | +1.750 | n/a | `5, 1, 0` | 207.28s |
| `boundary,word_packing,word_hull` | 258,422.500 | +5.500 | +1.750 | `31, 1, 0` | 218.05s |
| `boundary,word_packing,word_hull,word_support` | 258,422.500 | +5.500 | +1.750 | `36, 0` | 235.54s |

The first-round `word_hull` cuts target the same five words as
`word_support`, but the optimized coefficients move the LP more:

| Word | Frequency | Violation | Selected Colours |
|---|---:|---:|---:|
| `Ġsentence` | 10 | 0.5 | 6 |
| `Ġsentences` | 5 | 0.5 | 5 |
| `Ġfurniture` | 4 | 0.5 | 6 |
| `Ġoptimistic` | 3 | 0.5 | 8 |
| `Ġindependent` | 2 | 0.5 | 8 |

`word_support` adds no extra bound once `word_hull` is present, so this looks
like the cleaner replacement for that branch.

### 3. Multi-Word Value Cuts

Retested existing `group_value` and added a deeper alias `group_value_deep`
that scans more seed words, more candidate words, larger groups, and rank-10
token sets. These are valid multi-word lower-envelope cuts over a shared token
set, again leaving all outside tokens unrestricted.

Result: no violated cuts.

| Families | Final Bound | Added vs Base | Wall Time |
|---|---:|---:|---:|
| `group_value` | 258,417.000 | 0 | 4.44s |
| `group_value_deep` | 258,417.000 | 0 | 15.03s |
| `boundary,word_packing,group_value_deep` | 258,420.750 | +3.750 | 31.36s |

The candidate shared-token groups I tried do not expose a multi-word
incompatibility beyond what the per-word and cleanup cuts already see.

### 4. Ranked Local-Vocabulary Budget Cuts

Implemented as `group_budget_value`. For a group of words `G` and selected
token set `S`, compute:

```text
B(k) = best weighted segmentation cost using at most k active colours from S
```

Then add the best affine lower bound depending only on `sum_{tau in S} t_tau`.
This is weaker than full `group_value` for the same `(G,S)`, but it tests the
specific local-vocabulary-budget hypothesis.

Result: no violated cuts.

| Families | Final Bound | Added vs Base | Wall Time |
|---|---:|---:|---:|
| `group_budget_value` | 258,417.000 | 0 | 4.91s |
| `boundary,word_packing,group_budget_value` | 258,420.750 | +3.750 | 8.82s |

This does not seem like the remaining weakness on the books corpus.

### 5. Lifted Window / Boundary Cuts

Retested the existing lifted local window families together:
`window_pair,window_overlap_deep`. These compute compatible local interval
packing capacities over suspicious byte windows and selected token colours.

Result: many cuts separate, but they barely move the bound, and after standard
cleanup they do not move it at all.

| Families | Final Bound | Gain vs Base | Gain vs Boundary Cleanup | Cuts by Round | Wall Time |
|---|---:|---:|---:|---|---:|
| `window_pair,window_overlap_deep` | 258,417.167 | +0.167 | n/a | `54, 0` | 238.96s |
| `boundary,word_packing,window_pair,window_overlap_deep` | 258,420.750 | +3.750 | 0 | `80, 11, 0` | 253.90s |

The local window cuts are valid and visibly violated, but they are mostly
redundant with the standard cleanup rows for objective purposes.

Current best relaxation from these tests:

| Families | Final Bound | Gain vs Base | Rounded Tokens |
|---|---:|---:|---:|
| Base LP | 258,417.000 | 0 | 268,378 |
| `boundary,word_packing` | 258,420.750 | +3.750 | 268,378 |
| `boundary,word_packing,word_support` | 258,421.500 | +4.500 | 268,378 |
| `boundary,word_packing,word_hull` | 258,422.500 | +5.500 | 268,378 |

The useful next direction is therefore not broader single-word value functions
or local window packing. The clearest remaining signal is exact path-support
hull separation; improving its CPU cost or batching many word-hull cuts is the
most promising path from this set.

## Experimental Full Short-Word Hulls

I implemented `short_word_full_hull` to test how much the restricted
`word_hull` separator leaves on the table.

This separator is broader than `word_hull`. For a short word, it uses all local
token colours when the count is below a cap, enumerates every segmentation path,
and separates over the upward local hull:

```text
choose one path p
t_S may be any binary superset of colours used by p
```

Equivalently, it searches arbitrary signed inequalities over word edge-flow and
`t[S]`, with an L1 coefficient normalization, and enforces validity for the
whole upward closure of every path. This is still a projection onto selected
local colours; it does not add auxiliary path variables to the main LP.

The upward closure is important. A token can be globally selected but unused by
this word, so a local colour-support variable must not be equated with global
`t`. The valid local integer vertices are path plus any selected-token superset.

On the base books LP, raw-frequency short words did not produce cuts:

| Scan | Candidates | Scanned | Cuts | Separator Time |
|---|---:|---:|---:|---:|
| `len<=8`, top 250 frequent/fractional | 7,420 | 250 | 0 | 0.225s |
| `len<=8`, top 1000 frequent/fractional | 7,420 | 1000 | 0 | 0.942s |
| `len<=10`, top 1000 frequent/fractional | 9,689 | 1000 | 0 | 1.243s |
| `len<=12`, top 1000 frequent/fractional | 10,592 | 1000 | 0 | 1.389s |

Ranking by fractional signal instead finds cuts, mostly on low-frequency short
words:

| Scan | Checked | Cuts | Max Normalized Violation | Separator Time |
|---|---:|---:|---:|---:|
| `len<=10`, top 2000 signal | 1,999 | 3 | 0.125 | 4.049s |
| `len<=12`, top 2000 signal | 1,999 | 5 | 0.125 | 7.316s |
| `len<=12`, all signal | 10,262 | 31 | 0.1667 | 37.676s |
| `len<=16`, all signal | 10,592 | 32 | 0.1667 | 86.237s |

Example full-hull cuts from the all-signal scan:

| Word | Length | Frequency | Local Colours | Paths | Violation |
|---|---:|---:|---:|---:|---:|
| `Ġheeeere` | 8 | 1 | 10 | 72 | 0.1667 |
| `âĢľWooow` | 8 | 1 | 13 | 72 | 0.1667 |
| `âĢľWooooow` | 10 | 1 | 13 | 248 | 0.1667 |
| `âĢľNoooo` | 8 | 2 | 12 | 88 | 0.1667 |
| `Ġsentences` | 10 | 5 | 41 | 509 | 0.125 |
| `Ġsentence` | 9 | 10 | 34 | 255 | 0.125 |

Book training results with `len<=12`, up to 12k signal-ranked words, and up to
96 local colours:

| Families | Active Cuts | Final Bound | Gain vs Base | Rounded Tokens | Wall Time |
|---|---:|---:|---:|---:|---:|
| `short_word_full_hull` | 42 | 258,431.750 | +14.750 | 268,378 | 330.43s |
| `boundary,word_packing,short_word_full_hull` | 67 | 258,432.000 | +15.000 | 268,378 | 342.12s |

This is the largest tightening so far. The standard cleanup rows add only
`+0.250` once these full local hull cuts are present, so this family is
capturing most of the previous cleanup signal and more.

The bottleneck remains the main LP resolve, not hull separation:

- Full-hull separation for `len<=12`, all signal-ranked words: `37.676s`.
- First LP resolve after 38 full-hull cuts: `210.718s`.
- First LP resolve after 64 cleanup+full-hull cuts: `219.543s`.

The result looks plausible rather than too good to be true: the bound is still
well below the rounded tokenizer count (`268,378`), leaving about `9,946` tokens
of rounded gap after the best cut batch.

## Experimental Short-Word Pair Hulls

I added `short_word_pair_hull` after checking that pair hulls should only be
separated after individual word hulls are already satisfied. The combined
separator enforces this by skipping pair separation in any round where
`short_word_full_hull` still finds cuts.

The pair separator:

- ranks short words by fractional colour/edge signal;
- proposes pairs through shared fractional token colours;
- enumerates both words' paths exactly;
- separates the full upward hull of the path-pair product;
- skips pairs whose path-product row count exceeds a cap;
- tests pair separator LPs in parallel worker processes.

Books run:

```bash
uv run tokenizer-lp-train \
  --data-dir ~/Desktop/books \
  --vocab-size 512 \
  --kind lp \
  --max-token-length 8 \
  --min-token-count 5 \
  --pretokenizer nanochat \
  --lp-cut-rounds 5 \
  --lp-cuts-per-round 1000 \
  --lp-cut-families short_word_full_hull,short_word_pair_hull \
  --lp-short-word-full-hull-max-words 12000 \
  --lp-short-word-full-hull-max-length 12 \
  --lp-short-word-full-hull-max-colors 96 \
  --lp-short-word-pair-hull-max-words 500 \
  --lp-short-word-pair-hull-max-length 12 \
  --lp-short-word-pair-hull-max-colors 96 \
  --lp-short-word-pair-hull-max-pair-rows 250000 \
  --lp-short-word-pair-hull-max-pairs 800 \
  --lp-short-word-pair-hull-top-words-per-color 36 \
  --lp-short-word-pair-hull-workers 8 \
  --lp-word-support-max-paths 100000 \
  --lp-solver highspy \
  --lp-solution-cache-dir /tmp/tokenizer_lp_solution_cache
```

Result:

| Iteration | Active Cuts | New Cuts | Family Effect | LP Bound | Rounded Tokens |
|---:|---:|---:|---|---:|---:|
| 0 | 0 | 38 | individual full hull | 258,417.000 | 268,378 |
| 1 | 38 | 4 | individual full hull | 258,431.750 | 268,378 |
| 2 | 42 | 19 | pair hull | 258,431.750 | 268,378 |
| 3 | 61 | 4 | individual full hull | 258,512.542 | 268,218 |
| 4 | 65 | 1 | pair hull | 258,513.542 | 268,218 |
| 5 | 66 | 0 | done | 258,513.542 | 268,218 |

This needed five cut-separation rounds before convergence under the configured
limits. The sixth LP solve found no remaining individual or pair hull cuts, with
66 active cuts total.

Best bound so far:

| Families | Final Bound | Gain vs Base | Rounded Tokens |
|---|---:|---:|---:|
| Base LP | 258,417.000 | 0 | 268,378 |
| `short_word_full_hull` | 258,431.750 | +14.750 | 268,378 |
| `short_word_full_hull,short_word_pair_hull` | 258,513.542 | +96.542 | 268,218 |

Pair examples found after individual word hulls were active:

| Left Word | Right Word | Violation |
|---|---|---:|
| `Ġromance` | `Ġroom` | 0.0833 |
| `Ġsense` | `Ġseen` | 0.0833 |
| `Ġseen` | `Ġsensation` | 0.0833 |
| `Ġroom` | `Ġromantic` | 0.0833 |
| `Ġwater` | `Ġmatter` | 0.0769 |
| `Ġlast` | `Ġleast` | 0.0714 |
| `Ġwhere` | `Ġwere` | 0.0714 |
| `Ġmake` | `Ġmistake` | 0.0625 |

Multiprocessing timing:

| Pair Round | Candidates | Tested Tasks | Pair Cuts | Workers | Wall Time | Worker Build Time | Worker Solve Time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 9,346 | 792 | 19 | 8 | 22.490s | 4.763s | 10.544s |
| 4 | 9,487 | 783 | 1 | 8 | 22.979s | 0.368s | 0.830s |

The pair-testing bottleneck is mostly many independent small separator LPs plus
multiprocessing overhead, not path enumeration. In serial, the same first pair
test took about `154s`; with 8 workers it dropped to about `22.5s`. The main
end-to-end bottleneck is still the large LP resolve after adding cuts: the
iteration after pair rows took `235s` to resolve.
