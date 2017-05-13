package main

import (
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
)

const (
	FrameWidth  = 160
	FrameHeight = 210

	PreprocessedSize = 80 * 105 * 3
)

type PreprocessEnv struct {
	Env        anyrl.Env
	Subsampler anyvec.Mapper
}

func (p *PreprocessEnv) Reset() (observation anyvec.Vector, err error) {
	observation, err = p.Env.Reset()
	if observation != nil {
		observation = p.simplifyImage(observation)
	}
	return
}

func (p *PreprocessEnv) Step(action anyvec.Vector) (observation anyvec.Vector,
	reward float64, done bool, err error) {
	observation, reward, done, err = p.Env.Step(action)
	if observation != nil {
		observation = p.simplifyImage(observation)
	}
	return
}

func (p *PreprocessEnv) simplifyImage(in anyvec.Vector) anyvec.Vector {
	if p.Subsampler == nil {
		p.Subsampler = makeInputSubsampler(in.Creator())
	}
	return preprocessImage(p.Subsampler, in)
}

func preprocessImage(sampler anyvec.Mapper, image anyvec.Vector) anyvec.Vector {
	cr := image.Creator()
	out := cr.MakeVector(sampler.OutSize())
	sampler.Map(image, out)
	return out
}

func makeInputSubsampler(cr anyvec.Creator) anyvec.Mapper {
	// Scale down the image by factor of 2 on both axes.
	mapping := make([]int, 0, PreprocessedSize)
	for y := 0; y < FrameHeight; y += 2 {
		for x := 0; x < FrameWidth; x += 2 {
			sourceIdx := y*FrameWidth*3 + x*3
			for d := 0; d < 3; d++ {
				mapping = append(mapping, sourceIdx+d)
			}
		}
	}
	return cr.MakeMapper(FrameWidth*FrameHeight*3, mapping)
}
