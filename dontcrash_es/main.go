package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"sync"
	"time"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anyes"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/serializer"
)

const (
	TimePerStep = time.Second / 10
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "Usage: dontcrash_es <command> [args | -help]")
		fmt.Fprintln(os.Stderr)
		fmt.Fprintln(os.Stderr, "Available commands:")
		fmt.Fprintln(os.Stderr, " master   host a master node")
		fmt.Fprintln(os.Stderr, " slave    host a slave node")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "master":
		MasterMain(os.Args[2:])
	case "slave":
		SlaveMain(os.Args[2:])
	default:
		essentials.Die("unknown subcommand:", os.Args[1])
	}
}

func MasterMain(args []string) {
	var saveFile string
	var batchesPerUpdate int
	var batchSize int
	var listenAddr string
	var stepSize float64
	var noiseStddev float64
	fs := flag.NewFlagSet("master", flag.ExitOnError)
	fs.StringVar(&saveFile, "file", "trained_policy", "network output file")
	fs.IntVar(&batchesPerUpdate, "updates", 32, "batches per update")
	fs.IntVar(&batchSize, "batch", 16, "batch size (per log)")
	fs.StringVar(&listenAddr, "addr", ":1337", "address for listener")
	fs.Float64Var(&stepSize, "step", 0.01, "step size")
	fs.Float64Var(&noiseStddev, "stddev", 0.01, "mutation stddev")
	fs.Parse(args)

	creator := anyvec32.CurrentCreator()

	policy := loadOrCreateNetwork(creator, saveFile)

	// Setup the main coordinator for Evolution Strategies.
	master := &anyes.Master{
		Noise: anyes.NewNoise(1337, 1<<15),
		Params: anyes.MakeSafe(&anyes.AnynetParams{
			Params: anynet.AllParameters(policy),
		}),
		Normalize:   true,
		NoiseStddev: noiseStddev,
		StepSize:    stepSize,
		SlaveError: func(s anyes.Slave, e error) error {
			log.Println("slave disconnect:", e)
			s.(anyes.SlaveProxy).Close()
			return nil
		},
	}

	// Listen for incoming slaves.
	listener, err := net.Listen("tcp", listenAddr)
	if err != nil {
		essentials.Die(err)
	}
	log.Println("Listening on " + listenAddr)
	go anyes.ProxyListen(listener, master, log.Println)

	for {
		log.Println("Gathering batch of experience...")
		var bigBatch []*anyes.Rollout
		for i := 0; i < batchesPerUpdate; i++ {
			stopCond := &anyes.StopConds{MaxSteps: 600}
			batch, err := master.Rollouts(stopCond, batchSize/2)
			must(err)
			log.Printf("sub_mean=%f", anyes.MeanReward(batch))
			bigBatch = append(bigBatch, batch...)
		}
		log.Printf("mean=%f", anyes.MeanReward(bigBatch))
		must(master.Update(bigBatch))

		must(serializer.SaveAny(saveFile, policy))
	}
}

func SlaveMain(args []string) {
	var masterAddr string
	var numSlaves int
	fs := flag.NewFlagSet("slave", flag.ExitOnError)
	fs.StringVar(&masterAddr, "addr", "", "address for master")
	fs.IntVar(&numSlaves, "num", 1, "number of slaves")
	fs.Parse(args)

	if masterAddr == "" {
		essentials.Die("Missing -addr flag. See -help for more.")
	}

	var wg sync.WaitGroup
	for i := 0; i < numSlaves; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			creator := anyvec32.CurrentCreator()
			policy := createNetwork(creator)

			spec := muniverse.SpecForName("DontCrash-v0")
			if spec == nil {
				panic("environment not found")
			}
			env, err := muniverse.NewEnv(spec)

			// Used to debug on my end.
			//env, err := muniverse.NewEnvChrome("localhost:9222", "localhost:8080", spec)

			must(err)
			defer env.Close()

			slave := &anyes.AnynetSlave{
				Params: &anyes.AnynetParams{
					Params: anynet.AllParameters(policy),
				},
				Policy: policy,
				Env: &PreprocessEnv{
					Env:     env,
					Creator: creator,
				},
				Sampler: &anyrl.Bernoulli{},
			}
			conn, err := net.Dial("tcp", masterAddr)
			if err != nil {
				essentials.Die(err)
			}
			log.Println("connected a slave to", conn.RemoteAddr())
			log.Println("disconnected:", anyes.ProxyProvide(conn, slave))
		}()
	}

	wg.Wait()
	log.Println("all slaves disconnected")
}

func loadOrCreateNetwork(creator anyvec.Creator, path string) anyrnn.Stack {
	var res anyrnn.Stack
	if err := serializer.LoadAny(path, &res); err == nil {
		log.Println("Loaded network from file.")
		return res
	} else {
		res := createNetwork(creator)
		log.Println("Created new network.")
		return res
	}
}

func createNetwork(creator anyvec.Creator) anyrnn.Stack {
	markup := `
		Input(w=120, h=80, d=2)

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
			Layer: anynet.NewFCZero(creator, 256, 1),
		},
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
