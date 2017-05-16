package main

import (
	"compress/flate"
	"log"
	"math"
	"sync"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/lazyseq/lazyrnn"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

const (
	Host         = "localhost:5001"
	ParallelEnvs = 2
	BatchSize    = 4
)

const (
	RenderEnv = false

	NetworkSaveFile = "trained_policy"
)

func main() {
	// Setup vector creator.
	creator := anyvec32.CurrentCreator()

	// Create a neural network policy.
	policy := loadOrCreateNetwork(creator)
	actionSpace := &anyrl.Bernoulli{}

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
				out := lazyrnn.FixedHSM(30, false, seq, b)
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
			env, err := gym.Make(Host, "flashgames.DuskDrive-v0")
			must(err)
			defer env.Close()

			// Universe-specific configuration.
			must(env.UniverseWrap("CropObservations", nil))
			must(env.UniverseWrap("Vision", nil))
			must(env.UniverseConfigure(map[string]interface{}{
				"remotes": 1,
				"fps":     5,
			}))

			preproc := &PreprocessEnv{
				Env:     env,
				Creator: anynet.AllParameters(roller.Block)[0].Vector.Creator(),
			}
			for _ = range requests {
				rollout, err := roller.Rollout(preproc)
				must(err)
				log.Printf("rollout: sub_reward=%f",
					rollout.Rewards.Mean())
				resChan <- rollout
			}
		}()
	}

	go func() {
		wg.Wait()
		close(resChan)
	}()

	var res []*anyrl.RolloutSet
	for item := range resChan {
		res = append(res, item)
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
			Input(w=200, h=128, d=2)

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
			NewStacker(creator, 1, PreprocessedSize),
			&anyrnn.LayerBlock{Layer: net},
			&anyrnn.LayerBlock{
				Layer: anynet.NewFCZero(creator, 256, 3),
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

func must(err error) {
	if err != nil {
		panic(err)
	}
}
