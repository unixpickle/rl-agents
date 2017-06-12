package main

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/muniverse/chrome"
)

const (
	FrameWidth  = 480
	FrameHeight = 270

	MaxTimestep = 60 * 2 * 5

	PreprocessedSize = (FrameWidth / 4) * (FrameHeight/4 + 1)
)

type PreprocessEnv struct {
	Env     muniverse.Env
	Creator anyvec.Creator

	Timestep int
}

func (p *PreprocessEnv) Reset() (observation anyvec.Vector, err error) {
	err = p.Env.Reset()
	if err != nil {
		return
	}
	rawObs, err := p.Env.Observe()
	if err != nil {
		return
	}
	buffer, _, _, err := muniverse.RGB(rawObs)
	if err != nil {
		return
	}
	observation = p.simplifyImage(buffer)
	p.Timestep = 0
	return
}

func (p *PreprocessEnv) Step(action anyvec.Vector) (observation anyvec.Vector,
	reward float64, done bool, err error) {
	ops := p.Creator.NumOps()
	thresh := p.Creator.MakeNumeric(0.5)
	click := ops.Greater(anyvec.Sum(action.Slice(0, 1)), thresh)

	var events []interface{}

	if click {
		evt := chrome.MouseEvent{
			Type:       chrome.MousePressed,
			X:          FrameWidth / 2,
			Y:          FrameHeight / 2,
			Button:     chrome.LeftButton,
			ClickCount: 1,
		}
		evt1 := evt
		evt1.Type = chrome.MouseReleased
		events = append(events, &evt, &evt1)
	}

	reward, done, err = p.Env.Step(TimePerStep, events...)
	if err != nil {
		return
	}
	rawObs, err := p.Env.Observe()
	if err != nil {
		return
	}
	buffer, _, _, err := muniverse.RGB(rawObs)
	if err != nil {
		return
	}
	observation = p.simplifyImage(buffer)

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
			sourceIdx := (y*FrameWidth + x) * 3
			var value float64
			for d := 0; d < 3; d++ {
				value += float64(in[sourceIdx+d])
			}
			data = append(data, essentials.Round(value/3))
		}
	}
	return p.Creator.MakeVectorData(p.Creator.MakeNumericList(data))
}
