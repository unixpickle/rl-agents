package main

import (
	"bytes"
	"compress/flate"
	"encoding/gob"
	"io/ioutil"
	"log"
	"math"
	"sync"
	"time"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/treeagent"
	"github.com/unixpickle/weakai/idtrees"
)

const (
	ParallelEnvs = 4
	BatchSize    = 128
	LogInterval  = 16

	TimePerStep = time.Second / 10
)

const (
	SaveFile = "trained_policy"
)

func main() {
	gob.Register(&idtrees.Tree{})
	gob.Register(idtrees.Forest{})

	// Setup vector creator.
	creator := anyvec32.CurrentCreator()

	// Create a decision tree policy.
	policy := loadOrCreatePolicy(creator)

	// Setup a Roller for producing rollouts.
	roller := &treeagent.Roller{
		Policy:  policy,
		Creator: creator,

		// Compress the input frames as we store them.
		// If we used a ReferenceTape for the input, the
		// program would use way too much memory.
		MakeInputTape: func() (lazyseq.Tape, chan<- *anyseq.Batch) {
			return lazyseq.CompressedUint8Tape(flate.DefaultCompression)
		},
	}

	// Setup a trainer for producing new policies.
	trainer := &treeagent.Trainer{
		NumTrees:    20,
		NumFeatures: PreprocessedSize,
		RolloutFrac: 0.2,
		UseFeatures: PreprocessedSize / 10,
		BuildTree: func(samples []idtrees.Sample, attrs []idtrees.Attr) *idtrees.Tree {
			return idtrees.LimitedID3(samples, attrs, 0, 4)
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
			policy.Classifier = trainer.Train(r)

			// Save the new policy.
			trainLock.Lock()
			var data bytes.Buffer
			enc := gob.NewEncoder(&data)
			must(enc.Encode(policy))
			must(ioutil.WriteFile(SaveFile, data.Bytes(), 0755))
			trainLock.Unlock()
		}
	}()

	log.Println("Running. Press Ctrl+C to stop.")
	<-rip.NewRIP().Chan()

	// Avoid the race condition where we save during
	// exit.
	trainLock.Lock()
}

func gatherRollouts(roller *treeagent.Roller) []*anyrl.RolloutSet {
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
			spec := muniverse.SpecForName("Twins-v0")
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
				Creator: roller.Creator,
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

func loadOrCreatePolicy(creator anyvec.Creator) *treeagent.Policy {
	data, err := ioutil.ReadFile(SaveFile)
	if err != nil {
		log.Println("Created new policy.")
		return &treeagent.Policy{
			Classifier: &idtrees.Tree{
				// Uniform choices.
				Classification: map[idtrees.Class]float64{
					0: 1.0 / 3,
					1: 1.0 / 3,
					2: 1.0 / 3,
				},
			},
			NumActions: 3,
			Epsilon:    0.05,
		}
	}
	var res *treeagent.Policy
	dec := gob.NewDecoder(bytes.NewReader(data))
	must(dec.Decode(&res))
	log.Println("Loaded policy from file.")
	return res
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
