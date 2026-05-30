package main

import (
	"container/heap"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type ExportData struct {
	Words            []WordInput      `json:"words"`
	Candidates       []Candidate      `json:"candidates"`
	BaseVocab        []string         `json:"base_vocab"`
	Budget           int              `json:"budget"`
	PretokenizerMode string           `json:"pretokenizer_mode"`
	UnkToken         string           `json:"unk_token"`
	State            ExportState      `json:"state"`
	Stats            map[string]int64 `json:"stats"`
}

type WordInput struct {
	Text string `json:"text"`
	Freq int64  `json:"freq"`
}

type Candidate struct {
	Token         string  `json:"token"`
	LPValue       float64 `json:"lp_value"`
	InstanceCount int     `json:"instance_count"`
	Index         int     `json:"index"`
}

type ExportState struct {
	CheckpointNextIteration any      `json:"checkpoint_next_iteration"`
	ActiveCuts              any      `json:"active_cuts"`
	LPLowerBound            *float64 `json:"lp_lower_bound"`
}

type Edge struct {
	End      int
	TokenIdx int
}

type Counter struct {
	words      [][]rune
	freqs      []int64
	edges      [][][]Edge
	tokenWords [][]int
	selected   []bool
	wordCosts  []int
	total      int64
	marks      []int
	markGen    int
	maxWordLen int
}

type EvalScratch struct {
	marks   []int
	markGen int
	costs   []int
}

type Move struct {
	Remove []int `json:"remove"`
	Add    []int `json:"add"`
	Delta  int64 `json:"delta"`
}

type Args struct {
	InputJSON               string
	OutputJSON              string
	OutputTokenizer         string
	SeedJSON                string
	SeedTokenizer           string
	MaxIterations           int
	MaxSwapSize             int
	RemovePool              int
	AddPool                 int
	Exhaustive1Pool         int
	Exhaustive2Pool         int
	Exhaustive3Pool         int
	RandomProposals         int
	MoveTimeLimitSeconds    float64
	ProgressIntervalSeconds float64
	MinImprovement          int64
	NoiseRestarts           int
	NoiseSwaps              int
	NoiseMaxWorsen          int64
	TimeLimitSeconds        float64
	Workers                 int
	Seed                    int64
	RankHeapLimit           int
}

type BestState struct {
	Tokens   int64
	Selected []bool
	Iter     int
	Source   string
}

func main() {
	args := parseArgs()
	log.SetFlags(log.LstdFlags)
	start := time.Now()
	rng := rand.New(rand.NewSource(args.Seed))

	var data ExportData
	mustReadJSON(args.InputJSON, &data)
	if args.Workers <= 0 {
		args.Workers = runtime.NumCPU()
	}
	runtime.GOMAXPROCS(args.Workers)
	log.Printf("loaded instance: words=%d candidates=%d budget=%d workers=%d", len(data.Words), len(data.Candidates), data.Budget, args.Workers)

	counter := NewCounter(data.Words, data.Candidates)
	selected := initialSelection(args, data)
	initialTokens := counter.Initialize(selected)
	best := BestState{
		Tokens:   initialTokens,
		Selected: cloneBool(counter.selected),
		Iter:     0,
		Source:   "initial",
	}
	writeOutputs(args, data, best, initialTokens, counter.total, 0, 0, 0, time.Since(start), nil)
	log.Printf("INITIAL %s", compactSummary(data, best, initialTokens, counter.total, 0, 0, 0, time.Since(start)))

	deadline := time.Time{}
	if args.TimeLimitSeconds > 0 {
		deadline = start.Add(durationSeconds(args.TimeLimitSeconds))
	}
	acceptedMoves := 0
	evaluatedMoves := int64(0)
	localMinima := 0

	for iteration := 1; iteration <= args.MaxIterations; iteration++ {
		if !deadline.IsZero() && time.Now().After(deadline) {
			break
		}
		move, evals := findBestMove(args, counter, rng, deadline, iteration, start, best.Tokens)
		evaluatedMoves += evals
		if move != nil && move.Delta < -args.MinImprovement {
			applied := counter.ApplyMove(move.Remove, move.Add)
			if applied != move.Delta {
				log.Fatalf("move delta changed from %d to %d", move.Delta, applied)
			}
			acceptedMoves++
			log.Printf(
				"MOVE iteration=%d k=%d delta=%d tokens=%d remove=%v add=%v evals=%d elapsed=%.1fs",
				iteration,
				len(move.Remove),
				move.Delta,
				counter.total,
				tokenNames(data.Candidates, move.Remove),
				tokenNames(data.Candidates, move.Add),
				evals,
				time.Since(start).Seconds(),
			)
			if counter.total < best.Tokens {
				best = BestState{
					Tokens:   counter.total,
					Selected: cloneBool(counter.selected),
					Iter:     iteration,
					Source:   fmt.Sprintf("move_%d", acceptedMoves),
				}
				log.Printf("NEW_BEST %s", compactSummary(data, best, initialTokens, counter.total, acceptedMoves, evaluatedMoves, localMinima, time.Since(start)))
				writeOutputs(args, data, best, initialTokens, counter.total, acceptedMoves, evaluatedMoves, localMinima, time.Since(start), nil)
			}
			continue
		}

		localMinima++
		log.Printf(
			"LOCAL_MIN iteration=%d tokens=%d best=%d local_minima=%d evals=%d elapsed=%.1fs",
			iteration,
			counter.total,
			best.Tokens,
			localMinima,
			evals,
			time.Since(start).Seconds(),
		)
		if localMinima > args.NoiseRestarts {
			break
		}
		noise := chooseNoiseMove(args, counter, rng)
		if noise == nil {
			break
		}
		noise.Delta = counter.EvaluateMove(noise.Remove, noise.Add)
		if args.NoiseMaxWorsen >= 0 && noise.Delta > args.NoiseMaxWorsen {
			log.Printf("NOISE_SKIPPED iteration=%d k=%d delta=%d max_worsen=%d", iteration, len(noise.Remove), noise.Delta, args.NoiseMaxWorsen)
			counter.Initialize(selectedFromBool(best.Selected))
		} else {
			counter.ApplyMove(noise.Remove, noise.Add)
			log.Printf(
				"NOISE iteration=%d k=%d delta=%d tokens=%d remove=%v add=%v",
				iteration,
				len(noise.Remove),
				noise.Delta,
				counter.total,
				tokenNames(data.Candidates, noise.Remove),
				tokenNames(data.Candidates, noise.Add),
			)
		}
	}

	extra := map[string]any{
		"initial_tokens":       initialTokens,
		"final_current_tokens": counter.total,
		"accepted_moves":       acceptedMoves,
		"evaluated_moves":      evaluatedMoves,
		"local_minima":         localMinima,
		"elapsed_seconds":      time.Since(start).Seconds(),
		"budget":               data.Budget,
		"candidate_count":      len(data.Candidates),
		"workers":              args.Workers,
	}
	log.Printf("SUMMARY %s", compactSummary(data, best, initialTokens, counter.total, acceptedMoves, evaluatedMoves, localMinima, time.Since(start)))
	writeOutputs(args, data, best, initialTokens, counter.total, acceptedMoves, evaluatedMoves, localMinima, time.Since(start), extra)
}

func parseArgs() Args {
	var args Args
	flag.StringVar(&args.InputJSON, "input-json", "", "Exported 3-opt search instance JSON.")
	flag.StringVar(&args.OutputJSON, "output-json", "", "Write best/current summary JSON here.")
	flag.StringVar(&args.OutputTokenizer, "output-tokenizer", "", "Write best tokenizer JSON here whenever a new best is found.")
	flag.StringVar(&args.SeedJSON, "seed-json", "", "Optional CMA/3-opt summary JSON with selected_tokens.")
	flag.StringVar(&args.SeedTokenizer, "seed-tokenizer", "", "Optional LP DP tokenizer JSON to use as the initial vocabulary.")
	flag.IntVar(&args.MaxIterations, "max-iterations", 1000, "Maximum local-search iterations.")
	flag.IntVar(&args.MaxSwapSize, "max-swap-size", 3, "Maximum swap size: 1, 2, or 3.")
	flag.IntVar(&args.RemovePool, "remove-pool", 96, "Number of ranked selected tokens to consider removing.")
	flag.IntVar(&args.AddPool, "add-pool", 2048, "Number of ranked unselected tokens to consider adding.")
	flag.IntVar(&args.Exhaustive1Pool, "exhaustive-1-pool", 96, "Exhaustive pool for 1-swaps.")
	flag.IntVar(&args.Exhaustive2Pool, "exhaustive-2-pool", 32, "Exhaustive pool for 2-swaps.")
	flag.IntVar(&args.Exhaustive3Pool, "exhaustive-3-pool", 12, "Exhaustive pool for 3-swaps.")
	flag.IntVar(&args.RandomProposals, "random-proposals", 20000, "Random proposals per swap size after exhaustive checks.")
	flag.Float64Var(&args.MoveTimeLimitSeconds, "move-time-limit-seconds", 0, "Optional wall-clock cap for one best-move search; 0 means no per-iteration cap.")
	flag.Float64Var(&args.ProgressIntervalSeconds, "progress-interval-seconds", 30, "Log best-move search progress at this interval; 0 disables progress logs.")
	flag.Int64Var(&args.MinImprovement, "min-improvement", 0, "Minimum strict token improvement required to apply a move.")
	flag.IntVar(&args.NoiseRestarts, "noise-restarts", 5, "Number of local minima to perturb through before stopping.")
	flag.IntVar(&args.NoiseSwaps, "noise-swaps", 3, "Number of tokens to swap in each noise move.")
	flag.Int64Var(&args.NoiseMaxWorsen, "noise-max-worsen", -1, "Maximum accepted noise worsening; negative means unlimited.")
	flag.Float64Var(&args.TimeLimitSeconds, "time-limit-seconds", 0, "Optional wall-clock cap; 0 means no cap.")
	flag.IntVar(&args.Workers, "workers", runtime.NumCPU(), "Number of worker goroutines/OS threads.")
	flag.Int64Var(&args.Seed, "seed", 20260530, "Random seed.")
	flag.IntVar(&args.RankHeapLimit, "rank-heap-limit", 0, "Optional heap limit for rank scans; 0 ranks all exactly.")
	flag.Parse()
	if args.InputJSON == "" {
		log.Fatal("--input-json is required")
	}
	if args.MaxSwapSize < 1 || args.MaxSwapSize > 3 {
		log.Fatal("--max-swap-size must be 1, 2, or 3")
	}
	return args
}

func NewCounter(inputWords []WordInput, candidates []Candidate) *Counter {
	words := make([][]rune, len(inputWords))
	freqs := make([]int64, len(inputWords))
	maxWordLen := 0
	for i, word := range inputWords {
		words[i] = []rune(word.Text)
		freqs[i] = word.Freq
		if len(words[i]) > maxWordLen {
			maxWordLen = len(words[i])
		}
	}
	edges := make([][][]Edge, len(words))
	tokenWords := make([][]int, len(candidates))
	tokenByText := map[string]int{}
	maxTokenLen := 0
	for i, cand := range candidates {
		tokenRunes := []rune(cand.Token)
		if len(tokenRunes) == 0 {
			continue
		}
		tokenByText[cand.Token] = i
		if len(tokenRunes) > maxTokenLen {
			maxTokenLen = len(tokenRunes)
		}
	}
	seenStamp := make([]int, len(candidates))
	for wordIdx, word := range words {
		edges[wordIdx] = make([][]Edge, len(word))
		stamp := wordIdx + 1
		for start := range word {
			limit := start + maxTokenLen
			if limit > len(word) {
				limit = len(word)
			}
			for end := start + 1; end <= limit; end++ {
				idx, ok := tokenByText[string(word[start:end])]
				if !ok {
					continue
				}
				edges[wordIdx][start] = append(edges[wordIdx][start], Edge{End: end, TokenIdx: idx})
				if seenStamp[idx] != stamp {
					seenStamp[idx] = stamp
					tokenWords[idx] = append(tokenWords[idx], wordIdx)
				}
			}
		}
	}
	return &Counter{
		words:      words,
		freqs:      freqs,
		edges:      edges,
		tokenWords: tokenWords,
		selected:   make([]bool, len(candidates)),
		wordCosts:  make([]int, len(words)),
		marks:      make([]int, len(words)),
		maxWordLen: maxWordLen,
	}
}

func (c *Counter) NewScratch() *EvalScratch {
	return &EvalScratch{
		marks: make([]int, len(c.words)),
		costs: make([]int, c.maxWordLen+1),
	}
}

func (c *Counter) Initialize(selected []int) int64 {
	for i := range c.selected {
		c.selected[i] = false
	}
	for _, idx := range selected {
		if idx >= 0 && idx < len(c.selected) {
			c.selected[idx] = true
		}
	}
	var total int64
	scratch := c.NewScratch()
	for i := range c.words {
		cost := c.wordCostScratch(i, nil, nil, scratch)
		c.wordCosts[i] = cost
		total += int64(cost) * c.freqs[i]
	}
	c.total = total
	return total
}

func (c *Counter) EvaluateMove(remove, add []int) int64 {
	return c.EvaluateMoveScratch(remove, add, c.NewScratch())
}

func (c *Counter) EvaluateMoveScratch(remove, add []int, scratch *EvalScratch) int64 {
	affected := c.affectedWordsScratch(remove, add, scratch)
	var delta int64
	for _, wordIdx := range affected {
		newCost := c.wordCostScratch(wordIdx, remove, add, scratch)
		delta += int64(newCost-c.wordCosts[wordIdx]) * c.freqs[wordIdx]
	}
	return delta
}

func (c *Counter) ApplyMove(remove, add []int) int64 {
	affected := c.affectedWords(remove, add)
	for _, idx := range remove {
		c.selected[idx] = false
	}
	for _, idx := range add {
		c.selected[idx] = true
	}
	var delta int64
	scratch := c.NewScratch()
	for _, wordIdx := range affected {
		newCost := c.wordCostScratch(wordIdx, nil, nil, scratch)
		delta += int64(newCost-c.wordCosts[wordIdx]) * c.freqs[wordIdx]
		c.wordCosts[wordIdx] = newCost
	}
	c.total += delta
	return delta
}

func (c *Counter) affectedWords(remove, add []int) []int {
	c.markGen++
	if c.markGen == math.MaxInt {
		for i := range c.marks {
			c.marks[i] = 0
		}
		c.markGen = 1
	}
	var result []int
	for _, tokenIdx := range remove {
		for _, wordIdx := range c.tokenWords[tokenIdx] {
			if c.marks[wordIdx] == c.markGen {
				continue
			}
			c.marks[wordIdx] = c.markGen
			result = append(result, wordIdx)
		}
	}
	for _, tokenIdx := range add {
		for _, wordIdx := range c.tokenWords[tokenIdx] {
			if c.marks[wordIdx] == c.markGen {
				continue
			}
			c.marks[wordIdx] = c.markGen
			result = append(result, wordIdx)
		}
	}
	return result
}

func (c *Counter) affectedWordsLocal(remove, add []int) []int {
	return c.affectedWordsScratch(remove, add, c.NewScratch())
}

func (c *Counter) affectedWordsScratch(remove, add []int, scratch *EvalScratch) []int {
	scratch.markGen++
	if scratch.markGen == math.MaxInt {
		for i := range scratch.marks {
			scratch.marks[i] = 0
		}
		scratch.markGen = 1
	}
	result := make([]int, 0, 32)
	for _, tokenIdx := range remove {
		for _, wordIdx := range c.tokenWords[tokenIdx] {
			if scratch.marks[wordIdx] == scratch.markGen {
				continue
			}
			scratch.marks[wordIdx] = scratch.markGen
			result = append(result, wordIdx)
		}
	}
	for _, tokenIdx := range add {
		for _, wordIdx := range c.tokenWords[tokenIdx] {
			if scratch.marks[wordIdx] == scratch.markGen {
				continue
			}
			scratch.marks[wordIdx] = scratch.markGen
			result = append(result, wordIdx)
		}
	}
	return result
}

func (c *Counter) wordCost(wordIdx int, remove, add []int) int {
	return c.wordCostScratch(wordIdx, remove, add, c.NewScratch())
}

func (c *Counter) wordCostScratch(wordIdx int, remove, add []int, scratch *EvalScratch) int {
	wordLen := len(c.words[wordIdx])
	costs := scratch.costs[:wordLen+1]
	costs[wordLen] = 0
	for start := wordLen - 1; start >= 0; start-- {
		best := 1 + costs[start+1]
		for _, edge := range c.edges[wordIdx][start] {
			if !c.moveActive(edge.TokenIdx, remove, add) {
				continue
			}
			candidate := 1 + costs[edge.End]
			if candidate < best {
				best = candidate
			}
		}
		costs[start] = best
	}
	return costs[0]
}

func (c *Counter) moveActive(idx int, remove, add []int) bool {
	active := c.selected[idx]
	for _, removed := range remove {
		if idx == removed {
			active = false
			break
		}
	}
	for _, added := range add {
		if idx == added {
			active = true
			break
		}
	}
	return active
}

func findBestMove(
	args Args,
	counter *Counter,
	rng *rand.Rand,
	globalDeadline time.Time,
	iteration int,
	start time.Time,
	bestTokens int64,
) (*Move, int64) {
	selected, unselected := selectedAndUnselected(counter.selected)
	removeScores := rankMoves(args, counter, selected, true)
	addScores := rankMoves(args, counter, unselected, false)
	removePool := scoreIndices(removeScores, min(args.RemovePool, len(removeScores)))
	addPool := scoreIndices(addScores, min(args.AddPool, len(addScores)))
	searchStart := time.Now()
	deadline := globalDeadline
	if args.MoveTimeLimitSeconds > 0 {
		moveDeadline := searchStart.Add(durationSeconds(args.MoveTimeLimitSeconds))
		if deadline.IsZero() || moveDeadline.Before(deadline) {
			deadline = moveDeadline
		}
	}
	log.Printf(
		"SEARCH_START iteration=%d tokens=%d best_tokens=%d remove_pool=%d add_pool=%d max_swap=%d",
		iteration,
		counter.total,
		bestTokens,
		len(removePool),
		len(addPool),
		args.MaxSwapSize,
	)

	jobs := make(chan Move, args.Workers*4)
	results := make(chan Move, args.Workers*4)
	var evals atomic.Int64
	var wg sync.WaitGroup
	for worker := 0; worker < args.Workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			scratch := counter.NewScratch()
			for move := range jobs {
				move.Delta = counter.EvaluateMoveScratch(move.Remove, move.Add, scratch)
				evals.Add(1)
				results <- move
			}
		}()
	}
	go func() {
		wg.Wait()
		close(results)
	}()

	generatorDone := make(chan struct{})
	go func() {
		defer close(jobs)
		defer close(generatorDone)
		for k := 1; k <= args.MaxSwapSize; k++ {
			exhaustivePool := map[int]int{1: args.Exhaustive1Pool, 2: args.Exhaustive2Pool, 3: args.Exhaustive3Pool}[k]
			if exhaustivePool > 0 {
				limitedRemove := removePool[:min(exhaustivePool, len(removePool))]
				limitedAdd := addPool[:min(exhaustivePool, len(addPool))]
				if !generateExhaustive(k, limitedRemove, limitedAdd, jobs, deadline) {
					return
				}
			}
			for i := 0; i < args.RandomProposals; i++ {
				if !deadline.IsZero() && time.Now().After(deadline) {
					return
				}
				if len(removePool) < k || len(addPool) < k {
					break
				}
				move := Move{
					Remove: sampleK(rng, removePool, k),
					Add:    sampleK(rng, addPool, k),
				}
				sort.Ints(move.Remove)
				sort.Ints(move.Add)
				if !sendMove(jobs, move, deadline) {
					return
				}
			}
		}
	}()

	var best *Move
	nextProgress := searchStart.Add(durationSeconds(args.ProgressIntervalSeconds))
	for {
		select {
		case move, ok := <-results:
			if !ok {
				return best, evals.Load()
			}
			if best == nil || move.Delta < best.Delta {
				copyMove := move
				best = &copyMove
			}
		case <-time.After(250 * time.Millisecond):
		}
		if args.ProgressIntervalSeconds > 0 && time.Now().After(nextProgress) {
			elapsed := time.Since(searchStart).Seconds()
			rate := float64(evals.Load()) / math.Max(elapsed, 1e-9)
			bestDelta := "null"
			if best != nil {
				bestDelta = fmt.Sprintf("%d", best.Delta)
			}
			log.Printf(
				"SEARCH_PROGRESS iteration=%d tokens=%d best_tokens=%d evals=%d rate=%.1f/s best_delta=%s elapsed=%.1fs total_elapsed=%.1fs",
				iteration,
				counter.total,
				bestTokens,
				evals.Load(),
				rate,
				bestDelta,
				elapsed,
				time.Since(start).Seconds(),
			)
			for time.Now().After(nextProgress) {
				nextProgress = nextProgress.Add(durationSeconds(args.ProgressIntervalSeconds))
			}
		}
		select {
		case <-generatorDone:
			generatorDone = nil
		default:
		}
	}
}

func generateExhaustive(k int, removePool, addPool []int, jobs chan<- Move, deadline time.Time) bool {
	switch k {
	case 1:
		for _, r0 := range removePool {
			for _, a0 := range addPool {
				if !sendMove(jobs, Move{Remove: []int{r0}, Add: []int{a0}}, deadline) {
					return false
				}
			}
		}
	case 2:
		for i := 0; i < len(removePool); i++ {
			for j := i + 1; j < len(removePool); j++ {
				for a := 0; a < len(addPool); a++ {
					for b := a + 1; b < len(addPool); b++ {
						if !sendMove(jobs, Move{Remove: []int{removePool[i], removePool[j]}, Add: []int{addPool[a], addPool[b]}}, deadline) {
							return false
						}
					}
				}
			}
		}
	case 3:
		for i := 0; i < len(removePool); i++ {
			for j := i + 1; j < len(removePool); j++ {
				for l := j + 1; l < len(removePool); l++ {
					for a := 0; a < len(addPool); a++ {
						for b := a + 1; b < len(addPool); b++ {
							for c := b + 1; c < len(addPool); c++ {
								if !sendMove(jobs, Move{Remove: []int{removePool[i], removePool[j], removePool[l]}, Add: []int{addPool[a], addPool[b], addPool[c]}}, deadline) {
									return false
								}
							}
						}
					}
				}
			}
		}
	}
	return true
}

func sendMove(jobs chan<- Move, move Move, deadline time.Time) bool {
	if !deadline.IsZero() && time.Now().After(deadline) {
		return false
	}
	if len(move.Remove) != len(move.Add) || len(move.Remove) == 0 {
		return true
	}
	select {
	case jobs <- move:
		return true
	case <-time.After(100 * time.Millisecond):
		if !deadline.IsZero() && time.Now().After(deadline) {
			return false
		}
		jobs <- move
		return true
	}
}

type Score struct {
	Delta int64
	Idx   int
}

func rankMoves(args Args, counter *Counter, indices []int, remove bool) []Score {
	jobs := make(chan int, args.Workers*4)
	results := make(chan Score, args.Workers*4)
	var wg sync.WaitGroup
	for worker := 0; worker < args.Workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			scratch := counter.NewScratch()
			for idx := range jobs {
				var delta int64
				if remove {
					delta = counter.EvaluateMoveScratch([]int{idx}, nil, scratch)
				} else {
					delta = counter.EvaluateMoveScratch(nil, []int{idx}, scratch)
				}
				results <- Score{Delta: delta, Idx: idx}
			}
		}()
	}
	go func() {
		for _, idx := range indices {
			jobs <- idx
		}
		close(jobs)
		wg.Wait()
		close(results)
	}()
	if args.RankHeapLimit > 0 && args.RankHeapLimit < len(indices) {
		h := &scoreMaxHeap{}
		heap.Init(h)
		for score := range results {
			if h.Len() < args.RankHeapLimit {
				heap.Push(h, score)
			} else if scoreLess(score, (*h)[0]) {
				heap.Pop(h)
				heap.Push(h, score)
			}
		}
		out := make([]Score, h.Len())
		for i := range out {
			out[i] = heap.Pop(h).(Score)
		}
		sort.Slice(out, func(i, j int) bool { return scoreLess(out[i], out[j]) })
		return out
	}
	scores := make([]Score, 0, len(indices))
	for score := range results {
		scores = append(scores, score)
	}
	sort.Slice(scores, func(i, j int) bool { return scoreLess(scores[i], scores[j]) })
	return scores
}

type scoreMaxHeap []Score

func (h scoreMaxHeap) Len() int           { return len(h) }
func (h scoreMaxHeap) Less(i, j int) bool { return scoreLess(h[j], h[i]) }
func (h scoreMaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *scoreMaxHeap) Push(x any)        { *h = append(*h, x.(Score)) }
func (h *scoreMaxHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func scoreLess(a, b Score) bool {
	if a.Delta != b.Delta {
		return a.Delta < b.Delta
	}
	return a.Idx < b.Idx
}

func scoreIndices(scores []Score, limit int) []int {
	out := make([]int, 0, limit)
	for i := 0; i < limit && i < len(scores); i++ {
		out = append(out, scores[i].Idx)
	}
	return out
}

func initialSelection(args Args, data ExportData) []int {
	tokenToIdx := map[string]int{}
	for i, candidate := range data.Candidates {
		tokenToIdx[candidate.Token] = i
	}
	var seedTokens []string
	if args.SeedJSON != "" {
		var payload struct {
			SelectedTokens []string `json:"selected_tokens"`
		}
		mustReadJSON(args.SeedJSON, &payload)
		seedTokens = append(seedTokens, payload.SelectedTokens...)
	}
	if args.SeedTokenizer != "" {
		var payload struct {
			Vocab []string `json:"vocab"`
		}
		mustReadJSON(args.SeedTokenizer, &payload)
		seedTokens = append(seedTokens, payload.Vocab...)
	}
	selected := make([]int, 0, data.Budget)
	seen := map[string]bool{}
	ignored := 0
	for _, token := range seedTokens {
		if seen[token] {
			continue
		}
		idx, ok := tokenToIdx[token]
		if !ok {
			ignored++
			continue
		}
		selected = append(selected, idx)
		seen[token] = true
	}
	if ignored > 0 {
		log.Printf("ignored %d seed tokens that are not searchable candidates", ignored)
	}
	if len(selected) > data.Budget {
		sort.Slice(selected, func(i, j int) bool {
			a := data.Candidates[selected[i]]
			b := data.Candidates[selected[j]]
			if a.LPValue != b.LPValue {
				return a.LPValue > b.LPValue
			}
			return a.InstanceCount > b.InstanceCount
		})
		selected = selected[:data.Budget]
	}
	selectedSet := map[int]bool{}
	for _, idx := range selected {
		selectedSet[idx] = true
	}
	for i := range data.Candidates {
		if len(selected) >= data.Budget {
			break
		}
		if selectedSet[i] {
			continue
		}
		selected = append(selected, i)
		selectedSet[i] = true
	}
	return selected
}

func chooseNoiseMove(args Args, counter *Counter, rng *rand.Rand) *Move {
	selected, unselected := selectedAndUnselected(counter.selected)
	k := min(args.NoiseSwaps, len(selected), len(unselected))
	if k <= 0 {
		return nil
	}
	remove := sampleK(rng, selected, k)
	add := sampleK(rng, unselected, k)
	sort.Ints(remove)
	sort.Ints(add)
	return &Move{Remove: remove, Add: add}
}

func selectedAndUnselected(selected []bool) ([]int, []int) {
	var yes, no []int
	for i, value := range selected {
		if value {
			yes = append(yes, i)
		} else {
			no = append(no, i)
		}
	}
	return yes, no
}

func selectedFromBool(selected []bool) []int {
	out := make([]int, 0)
	for i, value := range selected {
		if value {
			out = append(out, i)
		}
	}
	return out
}

func sampleK(rng *rand.Rand, values []int, k int) []int {
	out := make([]int, 0, k)
	used := map[int]bool{}
	for len(out) < k {
		idx := rng.Intn(len(values))
		if used[idx] {
			continue
		}
		used[idx] = true
		out = append(out, values[idx])
	}
	return out
}

func writeOutputs(args Args, data ExportData, best BestState, initialTokens, currentTokens int64, acceptedMoves int, evaluatedMoves int64, localMinima int, elapsed time.Duration, extra map[string]any) {
	payload := summaryPayload(data, best, initialTokens, currentTokens, acceptedMoves, evaluatedMoves, localMinima, elapsed)
	for key, value := range extra {
		payload[key] = value
	}
	if args.OutputJSON != "" {
		mustWriteJSON(args.OutputJSON, payload)
	}
	if args.OutputTokenizer != "" {
		vocab := append([]string{}, data.BaseVocab...)
		for _, idx := range selectedFromBool(best.Selected) {
			vocab = append(vocab, data.Candidates[idx].Token)
		}
		tokenizer := map[string]any{
			"model":             "lp-dp",
			"vocab":             vocab,
			"pretokenizer_mode": data.PretokenizerMode,
			"unk_token":         data.UnkToken,
		}
		mustWriteJSON(args.OutputTokenizer, tokenizer)
	}
}

func summaryPayload(data ExportData, best BestState, initialTokens, currentTokens int64, acceptedMoves int, evaluatedMoves int64, localMinima int, elapsed time.Duration) map[string]any {
	selected := selectedFromBool(best.Selected)
	selectedTokens := make([]string, 0, len(selected))
	minLP := math.Inf(1)
	sumLP := 0.0
	zeroSelected := 0
	for _, idx := range selected {
		candidate := data.Candidates[idx]
		selectedTokens = append(selectedTokens, candidate.Token)
		if candidate.LPValue < minLP {
			minLP = candidate.LPValue
		}
		sumLP += candidate.LPValue
		if candidate.LPValue <= 1e-9 {
			zeroSelected++
		}
	}
	var selectedMinLP any
	var selectedAvgLP any
	if len(selected) > 0 {
		selectedMinLP = minLP
		selectedAvgLP = sumLP / float64(len(selected))
	}
	var gap any
	if data.State.LPLowerBound != nil {
		gap = float64(best.Tokens) - *data.State.LPLowerBound
	}
	return map[string]any{
		"tokens":                     best.Tokens,
		"iteration":                  best.Iter,
		"source":                     best.Source,
		"selected_tokens":            selectedTokens,
		"selected_token_count":       len(selectedTokens),
		"selected_zero_lp_count":     zeroSelected,
		"selected_positive_lp_count": len(selectedTokens) - zeroSelected,
		"selected_min_lp":            selectedMinLP,
		"selected_avg_lp":            selectedAvgLP,
		"checkpoint_next_iteration":  data.State.CheckpointNextIteration,
		"active_cuts":                data.State.ActiveCuts,
		"lp_lower_bound":             data.State.LPLowerBound,
		"gap_to_lower_bound":         gap,
		"initial_tokens":             initialTokens,
		"final_current_tokens":       currentTokens,
		"accepted_moves":             acceptedMoves,
		"evaluated_moves":            evaluatedMoves,
		"local_minima":               localMinima,
		"elapsed_seconds":            elapsed.Seconds(),
		"budget":                     data.Budget,
		"candidate_count":            len(data.Candidates),
	}
}

func compactSummary(data ExportData, best BestState, initialTokens, currentTokens int64, acceptedMoves int, evaluatedMoves int64, localMinima int, elapsed time.Duration) string {
	payload := summaryPayload(data, best, initialTokens, currentTokens, acceptedMoves, evaluatedMoves, localMinima, elapsed)
	delete(payload, "selected_tokens")
	encoded, _ := json.Marshal(payload)
	return string(encoded)
}

func tokenNames(candidates []Candidate, indices []int) []string {
	out := make([]string, len(indices))
	for i, idx := range indices {
		out[i] = candidates[idx].Token
	}
	return out
}

func cloneBool(values []bool) []bool {
	out := make([]bool, len(values))
	copy(out, values)
	return out
}

func durationSeconds(seconds float64) time.Duration {
	return time.Duration(seconds * float64(time.Second))
}

func mustReadJSON(path string, target any) {
	data, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("read %s: %v", path, err)
	}
	if err := json.Unmarshal(data, target); err != nil {
		log.Fatalf("parse %s: %v", path, err)
	}
}

func mustWriteJSON(path string, payload any) {
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		log.Fatalf("encode %s: %v", path, err)
	}
	if !strings.HasSuffix(string(data), "\n") {
		data = append(data, '\n')
	}
	tmp := fmt.Sprintf(".%s.tmp", path)
	if idx := strings.LastIndex(path, "/"); idx >= 0 {
		tmp = path[:idx+1] + "." + path[idx+1:] + ".tmp"
	}
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		log.Fatalf("write %s: %v", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		log.Fatalf("rename %s -> %s: %v", tmp, path, err)
	}
}

func min(values ...int) int {
	if len(values) == 0 {
		return 0
	}
	result := values[0]
	for _, value := range values[1:] {
		if value < result {
			result = value
		}
	}
	return result
}
