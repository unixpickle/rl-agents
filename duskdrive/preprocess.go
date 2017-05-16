package main

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
)

const (
	FrameWidth  = 800
	FrameHeight = 512

	MaxTimestep = 60 * 2 * 5

	PreprocessedSize = (FrameWidth * FrameHeight) / 16
)

type PreprocessEnv struct {
	Env     gym.Env
	Creator anyvec.Creator

	Timestep int
}

func (p *PreprocessEnv) Reset() (observation anyvec.Vector, err error) {
	rawObs, err := p.Env.Reset()
	if rawObs != nil {
		observation = p.simplifyImage(rawObs.(gym.Uint8Obs).Uint8Obs())
	}
	p.Timestep = 0
	return
}

func (p *PreprocessEnv) Step(action anyvec.Vector) (observation anyvec.Vector,
	reward float64, done bool, err error) {
	keys := []string{"ArrowUp", "ArrowLeft", "ArrowRight"}
	var events []interface{}
	for i, key := range keys {
		ops := action.Creator().NumOps()
		val := anyvec.Sum(action.Slice(i, i+1))
		pressed := ops.Greater(val, action.Creator().MakeNumeric(0))
		events = append(events, []interface{}{"KeyEvent", key, pressed})
	}
	rawObs, reward, done, _, err := p.Env.Step(events)
	if rawObs != nil {
		observation = p.simplifyImage(rawObs.(gym.Uint8Obs).Uint8Obs())
	}
	p.Timestep++
	if p.Timestep > MaxTimestep {
		done = true
	}
	return
}

func (p *PreprocessEnv) simplifyImage(in []uint8) anyvec.Vector {
	data := make([]float64, 0, PreprocessedSize)
	for y := 0; y < FrameHeight; y += 4 {
		for x := 0; x < FrameWidth; x += 4 {
			sourceIdx := y*FrameWidth*3 + x*3
			var value float64
			for d := 0; d < 3; d++ {
				value += float64(in[sourceIdx+d])
			}
			data = append(data, essentials.Round(value/3))
		}
	}
	return p.Creator.MakeVectorData(p.Creator.MakeNumericList(data))
}
