// Use model-based reinforcement learning to solve
// CartPole in as few episodes as possible.

package main

import (
	"log"
	"math"
	"math/rand"
	"time"

	gym "github.com/openai/gym-http-api/binding-go"
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/rip"
)

const (
	BaseURL    = "http://localhost:5000"
	MonitorDir = "/tmp/cartpole-monitor"
)

const (
	PolicyEpisodes = 10
	MaxReward      = 500
)

func main() {
	client, err := gym.NewClient(BaseURL)
	must(err)

	id, err := client.Create("CartPole-v1")
	must(err)

	c := anyvec64.CurrentCreator()

	// Policy takes an observation and outputs log probs for
	// the two moves.
	policy := anynet.Net{
		anynet.NewFC(c, 4, 2),
		anynet.LogSoftmax,
	}

	// Takes an observation + action and predicts next
	// observation.
	nextModel := anynet.Net{
		anynet.NewFC(c, 5, 30),
		anynet.Tanh,
		anynet.NewFC(c, 30, 4),
	}

	// Like nextModel, but predicts (pre-sigmoid) probability
	// for the episode ending.
	doneModel := anynet.Net{
		anynet.NewFC(c, 5, 30),
		anynet.Tanh,
		anynet.NewFC(c, 30, 1),
	}

	log.Println("Press Ctrl+C to stop and upload to Gym.")
	must(client.StartMonitor(id, MonitorDir, true, false, false))
	waiter := rip.NewRIP()

	var nextData, doneData anyff.SliceSampleList
	var starts []anyvec.Vector
	for !waiter.Done() {
		for i := 0; i < 5; i++ {
			log.Println("episode", len(starts))
			start, nextSamples, doneSamples := runTrial(client, id, policy)
			nextData = append(nextData, nextSamples...)
			doneData = append(doneData, doneSamples...)
			starts = append(starts, start)
		}
		trainModel(nextData, nextModel, anynet.MSE{})
		trainModel(doneData, doneModel, anynet.SigmoidCE{})
		if len(starts)%5 == 0 {
			trainPolicy(starts, nextModel, doneModel, policy)
		}
	}

	must(client.CloseMonitor(id))
	must(client.Close(id))

	// Set OPENAI_GYM_API_KEY env var.
	must(client.Upload(MonitorDir, "", ""))
}

func runTrial(c *gym.Client, id gym.InstanceID, policy anynet.Layer) (start anyvec.Vector,
	nextSamples, doneSamples anyff.SliceSampleList) {
	obs, err := c.Reset(id)
	must(err)
	start = anyvec64.MakeVectorData(obs.([]float64))
	var totalReward float64
	for {
		policyIn := anyvec64.MakeVectorData(obs.([]float64))
		policyOut := policy.Apply(anydiff.NewConst(policyIn), 1).Output()
		action := selectAction(policyOut)

		var done bool
		var reward float64
		obs, reward, done, _, err = c.Step(id, action, false)
		must(err)

		totalReward += reward

		nextSamples = append(nextSamples, &anyff.Sample{
			Input:  modelInput(policyIn, action),
			Output: anyvec64.MakeVectorData(obs.([]float64)),
		})

		if reward != MaxReward {
			doneSamples = append(doneSamples, &anyff.Sample{
				Input:  modelInput(policyIn, action),
				Output: anyvec64.MakeVectorData([]float64{boolToFloat(done)}),
			})
		}

		if done {
			log.Printf("actual reward: %f", totalReward)
			return
		}
	}
}

func trainModel(samples anysgd.SampleList, model anynet.Net, cost anynet.Cost) {
	timeout := make(chan struct{})
	go func() {
		time.Sleep(time.Second * 4)
		close(timeout)
	}()

	tr := &anyff.Trainer{
		Net:    model,
		Params: model.Parameters(),
		Cost:   cost,
	}
	sgd := &anysgd.SGD{
		Fetcher:     tr,
		Gradienter:  tr,
		Transformer: &anysgd.Adam{},
		Samples:     samples,
		BatchSize:   30,
		Rater:       anysgd.ConstRater(0.001),
	}
	sgd.Run(timeout)

	log.Printf("model %T cost: %v", cost, tr.LastCost)
}

func trainPolicy(starts []anyvec.Vector, nextModel, doneModel, policy anynet.Net) {
	timeout := time.After(time.Second * 8)

	tr := &anyff.Trainer{
		Net:     policy,
		Params:  policy.Parameters(),
		Cost:    anynet.DotCost{},
		Average: true,
	}
	var adam anysgd.Adam

	for {
		samples, modeledReward := sampleModel(starts, nextModel, doneModel, policy)

		select {
		case <-timeout:
			log.Printf("modeled reward: %f", modeledReward)
			return
		default:
		}

		batch, _ := tr.Fetch(samples)
		grad := adam.Transform(tr.Gradient(batch))
		grad.Scale(-0.001)
		grad.AddToVars()
	}
}

func sampleModel(starts []anyvec.Vector, nextModel, doneModel,
	policy anynet.Net) (anyff.SliceSampleList, float64) {
	var samples anyff.SliceSampleList
	var totalReward float64
	for i := 0; i < PolicyEpisodes; i++ {
		state := starts[rand.Intn(len(starts))]
		var inputs []anyvec.Vector
		var outMasks []anyvec.Vector
		var reward float64

		for reward < MaxReward {
			inputs = append(inputs, state.Copy())
			policyOut := policy.Apply(anydiff.NewConst(state), 1).Output()
			action := selectAction(policyOut)
			mask := make([]float64, 2)
			mask[action] = 1
			outMasks = append(outMasks, anyvec64.MakeVectorData(mask))
			reward++

			modelIn := anydiff.NewConst(modelInput(state, action))
			state = nextModel.Apply(modelIn, 1).Output()
			done := anynet.Sigmoid.Apply(doneModel.Apply(modelIn, 1), 1).Output()

			if rand.Float64() < done.Data().([]float64)[0] {
				break
			}
		}

		totalReward += reward
		for i, x := range outMasks {
			x.Scale(reward)
			samples = append(samples, &anyff.Sample{
				Input:  inputs[i],
				Output: x,
			})
		}
	}
	return samples, totalReward / PolicyEpisodes
}

func modelInput(state anyvec.Vector, action int) anyvec.Vector {
	actionVec := anyvec64.MakeVectorData([]float64{float64(action)})
	return anyvec64.Concat(state, actionVec)
}

func selectAction(probs anyvec.Vector) int {
	vals := probs.Data().([]float64)
	if math.Exp(float64(vals[0])) > rand.Float64() {
		return 0
	} else {
		return 1
	}
}

func boolToFloat(b bool) float64 {
	if b {
		return 1
	} else {
		return 0
	}
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
