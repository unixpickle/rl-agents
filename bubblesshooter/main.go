package main

import (
	"log"
	"time"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anya3c"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/serializer"
)

const (
	ParallelEnvs = 8
	TimePerStep  = time.Second / 5
	SaveInterval = time.Minute * 5
)

const (
	SaveFile = "trained_agent"
)

func main() {
	// Setup vector creator.
	creator := anyvec32.CurrentCreator()

	// Create a neural network policy.
	agent := loadOrCreateAgent(creator)
	agent.ActionSpace = &anyrl.Tuple{
		Spaces:      []interface{}{anyrl.Gaussian{}, &anyrl.Bernoulli{}},
		ParamSizes:  []int{4, 1},
		SampleSizes: []int{2, 1},
	}

	// Create multiple environment instances.
	log.Println("Creating environments...")
	spec := muniverse.SpecForName("BubblesShooter-v0")
	if spec == nil {
		panic("environment not found")
	}
	var envs []anyrl.Env
	for i := 0; i < ParallelEnvs; i++ {
		env, err := muniverse.NewEnv(spec)
		// Used to debug on my end.
		//env, err := muniverse.NewEnvChrome("localhost:9222", "localhost:8080", spec)
		must(err)
		defer env.Close()
		envs = append(envs, &PreprocessEnv{
			Env:     env,
			Creator: agent.AllParameters()[0].Vector.Creator(),
		})
	}

	paramServer := anya3c.RMSPropParamServer(agent, agent.AllParameters(),
		1e-4, anysgd.RMSProp{DecayRate: 0.99})

	a3c := &anya3c.A3C{
		ParamServer: paramServer,
		Logger: &anya3c.AvgLogger{
			Creator: creator,
			Logger: &anya3c.StandardLogger{
				Episode:    true,
				Update:     true,
				Regularize: true,
			},
			// Only log updates and entropy periodically.
			Update:     60,
			Regularize: 120,
		},
		Discount: 0.9,
		MaxSteps: 5,
		Regularizer: &anypg.EntropyReg{
			Entropyer: agent.ActionSpace.(anyrl.Entropyer),
			Coeff:     0.003,
		},
	}

	go func() {
		for {
			agent, err := paramServer.LocalCopy()
			must(err)
			must(serializer.SaveAny(SaveFile, agent.Base, agent.Actor,
				agent.Critic))
			time.Sleep(SaveInterval)
		}
	}()

	log.Println("Running A3C...")
	a3c.Run(envs, nil)
}

func loadOrCreateAgent(creator anyvec.Creator) *anya3c.Agent {
	var base, actor, critic anyrnn.Block
	if err := serializer.LoadAny(SaveFile, &base, &actor, &critic); err == nil {
		log.Println("Loaded agent from file.")
		return &anya3c.Agent{
			Base:   base,
			Actor:  actor,
			Critic: critic,
		}
	} else {
		log.Println("Creating new agent.")
		markup := `
			Input(w=131, h=87, d=6)

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
		return &anya3c.Agent{
			Base: &anyrnn.LayerBlock{Layer: net},
			Actor: &anyrnn.LayerBlock{
				Layer: anynet.NewFCZero(creator, 256, 5),
			},
			Critic: &anyrnn.LayerBlock{
				Layer: anynet.NewFCZero(creator, 256, 1),
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
