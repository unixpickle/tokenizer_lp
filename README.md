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

## LP Cut Separators

The LP cut loop is controlled by `--lp-cut-families`. Most older families are
direct combinatorial separators. The newer short-word families below were added
to mine cuts from fractional local tokenization structure; they are the ones
used by the current books experiments.

### Brute-Force Hull Separators

`short_word_pair_hull` and `short_word_triple_hull` build a small projected
validity LP for each candidate pair or triple of words. These are slower than
templates but can discover cuts that do not match any known pattern.

For pair hulls, `--lp-short-word-pair-hull-pruning fractional_edges_shared_colors`
keeps only currently fractional edge variables and shared fractional token
colors, deduplicates projected rows, and applies
`--lp-short-word-pair-hull-max-pair-rows` after that reduction. This is the
main practical mode: it usually leaves tiny LPs while preserving the useful
late-stage pair cuts seen so far.

For triple hulls, `--lp-short-word-triple-hull-token-mode at_least_two` keeps
fractional token colors that appear in at least two of the three words. The
row limit is `--lp-short-word-triple-hull-max-rows`; candidates over the
reduced row budget are skipped.

Important exploration knobs:

- `--lp-short-word-pair-hull-candidate-strategy random` shuffles pair
  candidates instead of using the shared-color score.
- `--lp-short-word-pair-hull-max-pairs` limits pair LPs solved per round.
- `--lp-short-word-triple-hull-candidate-sample` controls the shuffled triple
  candidate pool.
- `--lp-short-word-triple-hull-max-triples` limits triple LPs solved per round.
- `--lp-cuts-per-round` is still the final global number of cuts added. Final
  cut selection should generally stay `--lp-cut-selection-score violation` so
  added cuts are ordered deterministically by violation size.

### Direct Pair Templates

The pair template families are `short_word_pair_single_chain`,
`short_word_pair_bridge_chain`, and `short_word_pair_chains` (both). They scan
short fractional words for chains of consecutive overlapping token edges.
Only fractional edge variables whose token colors are also fractional are used.

Each chain support has positive coefficient `+1` on its edge columns and
negative coefficient `-1` on the involved token-color variables.

`short_word_pair_single_chain` handles a single word chain whose first and last
edge have the same token color. It keeps chains of length at least 4 and adds:

```text
sum(chain_edges) - t[color] <= rhs
```

The `rhs` is computed exactly for that word by enumerating projected path
signatures, so this family is validated by construction rather than relying on
a closed-form constant.

`short_word_pair_bridge_chain` combines two different words that each have a
chain from token color `a` to token color `b`. It adds:

```text
sum(left_chain_edges) + sum(right_chain_edges) - t[a] - t[b] <= rhs
```

Again, `rhs` is the exact maximum over the projected local path product for
the two words. This is why these pair templates are safe even though the
chains found by the scan vary in length and geometry.

Pair template limits:

- `--lp-short-word-pair-template-max-words`
- `--lp-short-word-pair-template-max-length`
- `--lp-short-word-pair-template-candidate-word-multiplier`
- `--lp-short-word-pair-template-top-supports-per-shape`
- `--lp-short-word-pair-template-max-chain-edges`
- `--lp-short-word-pair-template-max-cuts`

Pair template scoring flags:

- `--lp-short-word-pair-template-word-score weighted_fractionality|random`
- `--lp-short-word-pair-template-support-score support_value|random`
- `--lp-short-word-pair-template-cut-score violation|random`
- `--lp-short-word-pair-template-random-seed`

Random scoring only changes top-k exploration. Filtering for possible supports
still requires fractional edge values, fractional token-color values, and the
overlap/chain structure needed by the template.

### Direct Triple And 5-Cycle Templates

These families use small token-color hypergraph templates over supports found
inside short words. A pair support is two overlapping edges in one word with
distinct token colors. A triple support is three pairwise-overlapping edges in
one word with three distinct token colors. For each `(token shape, word)`, only
one support is retained before the per-shape top-k filter.

`short_word_triple_triangle` finds three token colors `a,b,c` and three
different words supporting the token pairs `(a,b)`, `(a,c)`, and `(b,c)`.
The cut is:

```text
sum(pair_support_edges) - t[a] - t[b] - t[c] <= 1
```

This is a triangle inequality over mutually exclusive pair supports. The
mutual exclusion requirement is important: for each support word, the two
positive edge variables overlap in that word's tokenization, so an integral
path can take at most one of them.

`short_word_triple_4cycle` uses two triple supports on 3-token shapes that
overlap in exactly two token colors, plus a pair support on the two remaining
colors. For union token colors `a,b,c,d`, it adds:

```text
sum(two_triple_supports_and_pair_support_edges) - t[a] - t[b] - t[c] - t[d] <= 1
```

`short_word_5cycle` enumerates 5-cycles in the graph of available pair
supports. For cycle token colors `a,b,c,d,e`, it picks one pair support for
each cycle edge and adds:

```text
sum(five_pair_support_edges) - t[a] - t[b] - t[c] - t[d] - t[e] <= 2
```

This is the odd-cycle analogue of the triangle template. It is valid for the
same reason: each selected pair support consists of two mutually exclusive
token edges in its word. The separator requires five distinct support words.

The triple/5-cycle templates can optionally run expensive validation with
`--lp-short-word-triple-template-validate`. This enumerates projected path
signatures and rejects any template whose maximum integral slack is positive.
It is useful when developing new templates, but it is usually unnecessary and
too expensive for train-time runs once the template has been checked.

Triple and 5-cycle limits:

- `--lp-short-word-triple-template-max-words`
- `--lp-short-word-triple-template-max-length`
- `--lp-short-word-triple-template-candidate-word-multiplier`
- `--lp-short-word-triple-template-top-supports-per-shape`
- `--lp-short-word-triple-template-max-cuts`
- `--lp-short-word-5cycle-supports-per-edge`
- `--lp-short-word-5cycle-support-assignments`

`--lp-short-word-5cycle-supports-per-edge` controls how many retained pair
supports are considered for each edge of a token 5-cycle. With
`--lp-short-word-5cycle-support-assignments 0`, the separator checks the best
assignment plus single-edge substitutions. With a positive value, it caps the
total support assignments checked per token 5-cycle and samples additional
assignments randomly after the deterministic neighborhood.

Triple/5-cycle scoring flags:

- `--lp-short-word-triple-template-word-score weighted_fractionality|random`
- `--lp-short-word-triple-template-support-score support_value|random`
- `--lp-short-word-triple-template-cut-score violation|random`
- `--lp-short-word-triple-template-random-seed`

As with pair templates, random scoring broadens exploration but does not relax
the structural validity filters.

### Running Multiple Families

By default, some later families may be skipped when earlier separators already
found cuts. Use `--lp-run-all-cut-families` to run every requested family in
each separation round. Each family is independently limited to
`--lp-cuts-per-round` cuts before global selection, and the final added set is
also limited by `--lp-cuts-per-round`.

Use separate per-family max-cut flags, such as
`--lp-short-word-pair-template-max-cuts` and
`--lp-short-word-triple-template-max-cuts`, when a separator can produce a very
large candidate list before the global cap.

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
