package main

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/muniverse/chrome"
)

const (
	FrameWidth  = 522
	FrameHeight = 348

	MaxTimestep = 60 * 5

	PreprocessedSize = (FrameWidth/4 + 1) * (FrameHeight / 4)
)

type PreprocessEnv struct {
	Env     muniverse.Env
	Creator anyvec.Creator

	Timestep  int
	LastFrame anyvec.Vector
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
	p.LastFrame = p.simplifyImage(buffer)
	observation = joinFrames(p.LastFrame, p.LastFrame)
	p.Timestep = 0
	return
}

func (p *PreprocessEnv) Step(action anyvec.Vector) (observation anyvec.Vector,
	reward float64, done bool, err error) {
	idx := anyvec.MaxIndex(action)
	row := idx / ClickGridCols
	col := idx % ClickGridCols

	x := (float64(col) / ClickGridCols) * FrameWidth
	y := (float64(row) / ClickGridRows) * FrameHeight

	var events []interface{}
	for _, t := range []chrome.MouseEventType{chrome.MousePressed, chrome.MouseReleased} {
		events = append(events, &chrome.MouseEvent{
			Type:       t,
			X:          int(essentials.Round(x)),
			Y:          int(essentials.Round(y)),
			Button:     chrome.LeftButton,
			ClickCount: 1,
		})
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
	newFrame := p.simplifyImage(buffer)
	observation = joinFrames(newFrame, p.LastFrame)
	p.LastFrame = newFrame

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

func joinFrames(f1, f2 anyvec.Vector) anyvec.Vector {
	joined := f1.Creator().Concat(f1, f2)
	transpose := joined.Creator().MakeVector(joined.Len())
	anyvec.Transpose(joined, transpose, 2)
	return transpose
}
