package main

import (
	"compress/flate"
	"log"
	"math"
	"sync"
	"time"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/lazyseq/lazyrnn"
	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

const (
	ParallelEnvs = 3
	BatchSize    = 24
	LogInterval  = 8

	TimePerSubstep  = time.Second / 30
	SubstepsPerStep = 3
)

const (
	NetworkSaveFile = "trained_policy"
)

func main() {
	// Setup vector creator.
	creator := anyvec32.CurrentCreator()

	// Create a neural network policy.
	policy := loadOrCreateNetwork(creator)
	actionSpace := &anyrl.Tuple{}
	for i := 0; i < SubstepsPerStep; i++ {
		actionSpace.Spaces = append(actionSpace.Spaces, anyrl.Gaussian{},
			&anyrl.Bernoulli{})
		actionSpace.ParamSizes = append(actionSpace.ParamSizes, 4, 1)
		actionSpace.SampleSizes = append(actionSpace.SampleSizes, 2, 1)
	}

	// Setup an RNNRoller for producing rollouts.
	roller := &anyrl.RNNRoller{
		Block:       policy,
		ActionSpace: actionSpace,

		// Compress the input frames as we store them.
		// If we used a ReferenceTape for the input, the
		// program would use way too much memory.
		MakeInputTape: func() (lazyseq.Tape, chan<- *anyseq.Batch) {
			return lazyseq.CompressedUint8Tape(flate.DefaultCompression)
		},
	}

	// Setup Trust Region Policy Optimization for training.
	trpo := &anypg.TRPO{
		NaturalPG: anypg.NaturalPG{
			Policy:      policy,
			Params:      policy.Parameters(),
			ActionSpace: actionSpace,

			// Speed things up a bit.
			Iters: 10,
			Reduce: (&anyrl.FracReducer{
				Frac:          0.1,
				MakeInputTape: roller.MakeInputTape,
			}).Reduce,

			ApplyPolicy: func(seq lazyseq.Rereader, b anyrnn.Block) lazyseq.Rereader {
				out := lazyrnn.FixedHSM(30, true, seq, b)
				return lazyseq.Lazify(lazyseq.Unlazify(out))
			},
			ActionJudger: &anypg.QJudger{Discount: 0.99},
		},
	}

	// Train on a background goroutine so that we can
	// listen for Ctrl+C on the main goroutine.
	var trainLock sync.Mutex
	go func() {
		for batchIdx := 0; true; batchIdx++ {
			log.Println("Gathering batch of experience...")

			// Join the rollouts into one set.
			rollouts := gatherRollouts(roller)
			r := anyrl.PackRolloutSets(rollouts)

			// Print the stats for the batch.
			log.Printf("batch %d: mean=%f stddev=%f", batchIdx,
				r.Rewards.Mean(), math.Sqrt(r.Rewards.Variance()))

			// Train on the rollouts.
			log.Println("Training on batch...")
			grad := trpo.Run(r)
			trainLock.Lock()
			grad.AddToVars()
			trainLock.Unlock()
		}
	}()

	log.Println("Running. Press Ctrl+C to stop.")
	<-rip.NewRIP().Chan()

	// Avoid the race condition where we save during
	// parameter updates.
	trainLock.Lock()
	must(serializer.SaveAny(NetworkSaveFile, policy))
}

func gatherRollouts(roller *anyrl.RNNRoller) []*anyrl.RolloutSet {
	resChan := make(chan *anyrl.RolloutSet, BatchSize)

	requests := make(chan struct{}, BatchSize)
	for i := 0; i < BatchSize; i++ {
		requests <- struct{}{}
	}
	close(requests)

	var wg sync.WaitGroup
	for i := 0; i < ParallelEnvs; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			spec := muniverse.SpecForName("KatanaFruits-v0")
			if spec == nil {
				panic("environment not found")
			}
			env, err := muniverse.NewEnv(spec)

			// Used to debug on my end.
			//env, err := muniverse.NewEnvChrome("localhost:9222", "localhost:8080", spec)

			must(err)
			defer env.Close()

			preproc := &PreprocessEnv{
				Env:     env,
				Creator: anynet.AllParameters(roller.Block)[0].Vector.Creator(),
			}
			for _ = range requests {
				rollout, err := roller.Rollout(preproc)
				must(err)
				resChan <- rollout
			}
		}()
	}

	go func() {
		wg.Wait()
		close(resChan)
	}()

	var res []*anyrl.RolloutSet
	var batchRewardSum float64
	var numBatchReward int
	for item := range resChan {
		res = append(res, item)
		numBatchReward++
		batchRewardSum += item.Rewards.Mean()
		if numBatchReward == LogInterval || len(res) == BatchSize {
			log.Printf("sub_mean=%f", batchRewardSum/float64(numBatchReward))
			numBatchReward = 0
			batchRewardSum = 0
		}
	}
	return res
}

func loadOrCreateNetwork(creator anyvec.Creator) anyrnn.Stack {
	var res anyrnn.Stack
	if err := serializer.LoadAny(NetworkSaveFile, &res); err == nil {
		log.Println("Loaded network from file.")
		return res
	} else {
		log.Println("Created new network.")
		markup := `
			Input(w=160, h=214, d=2)

			Linear(scale=0.01)

			Conv(w=4, h=4, n=16, sx=2, sy=2)
			Tanh
			Conv(w=4, h=4, n=32, sx=2, sy=2)
			Tanh
			FC(out=256)
			Tanh
		`
		convNet, err := anyconv.FromMarkup(creator, markup)
		must(err)
		net := convNet.(anynet.Net)
		net = setupVisionLayers(net)
		return anyrnn.Stack{
			anyrnn.NewMarkov(creator, 1, PreprocessedSize, true),
			&anyrnn.LayerBlock{Layer: net},
			&anyrnn.LayerBlock{
				Layer: biasLastLayer(anynet.NewFCZero(creator, 256,
					5*SubstepsPerStep)),
			},
		}
	}
}

func setupVisionLayers(net anynet.Net) anynet.Net {
	for _, layer := range net {
		projectOutSolidColors(layer)
	}
	return net
}

func projectOutSolidColors(layer anynet.Layer) {
	switch layer := layer.(type) {
	case *anyconv.Conv:
		filters := layer.Filters.Vector
		inDepth := layer.InputDepth
		numFilters := layer.FilterCount
		filterSize := filters.Len() / numFilters
		for i := 0; i < numFilters; i++ {
			filter := filters.Slice(i*filterSize, (i+1)*filterSize)

			// Compute the mean for each input channel.
			negMean := anyvec.SumRows(filter, inDepth)
			negMean.Scale(negMean.Creator().MakeNumeric(-1 / float64(filterSize/inDepth)))
			anyvec.AddRepeated(filter, negMean)
		}
	case *anynet.FC:
		negMean := anyvec.SumCols(layer.Weights.Vector, layer.OutCount)
		negMean.Scale(negMean.Creator().MakeNumeric(-1 / float64(layer.InCount)))
		anyvec.AddChunks(layer.Weights.Vector, negMean)
	}
}

func biasLastLayer(layer *anynet.FC) *anynet.FC {
	// Bias towards pressing down the mouse (i.e. dragging).
	idx := layer.Biases.Vector.Len() - 1
	c := layer.Biases.Vector.Creator()
	layer.Biases.Vector.Slice(idx, idx+1).AddScalar(c.MakeNumeric(1))
	return layer
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
