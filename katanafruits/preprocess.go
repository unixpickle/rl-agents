package main

import (
	"math"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/muniverse/chrome"
)

const (
	FrameWidth  = 320
	FrameHeight = 427

	MaxTimestep = 60 * 2 * 5

	PreprocessedSize = (FrameWidth / 2) * (FrameHeight/2 + 1)
)

type PreprocessEnv struct {
	Env     *muniverse.Env
	Creator anyvec.Creator

	Timestep int

	// Number of timesteps for which mouse has been pressed.
	PressedTimesteps int
	LastX            int
	LastY            int
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
	for i := 0; i < SubstepsPerStep && !done; i++ {
		floatData := action.Data().([]float32)[i*3 : (i+1)*3]
		x := clipMouse(float64(floatData[0]), FrameWidth)
		y := clipMouse(float64(floatData[1]), FrameHeight)
		clicked := floatData[2] > 0.5

		var events []interface{}

		if p.PressedTimesteps == 0 && clicked {
			events = append(events, &chrome.MouseEvent{
				Type:       chrome.MousePressed,
				X:          x,
				Y:          y,
				Button:     chrome.LeftButton,
				ClickCount: 1,
			})
		} else if p.PressedTimesteps == 1 && !clicked {
			events = append(events, &chrome.MouseEvent{
				Type:       chrome.MouseReleased,
				X:          p.LastX,
				Y:          p.LastY,
				Button:     chrome.LeftButton,
				ClickCount: 1,
			}, &chrome.MouseEvent{
				Type: chrome.MouseMoved,
				X:    x,
				Y:    y,
			})
		} else if p.PressedTimesteps > 1 && !clicked {
			events = append(events, &chrome.MouseEvent{
				Type:   chrome.MouseReleased,
				X:      x,
				Y:      y,
				Button: chrome.LeftButton,
			})
		} else {
			evt := &chrome.MouseEvent{
				Type: chrome.MouseMoved,
				X:    x,
				Y:    y,
			}
			if clicked {
				evt.Button = chrome.LeftButton
			}
			events = append(events, evt)
		}

		if clicked {
			p.PressedTimesteps++
		} else {
			p.PressedTimesteps = 0
		}

		p.LastX, p.LastY = x, y

		var subReward float64
		subReward, done, err = p.Env.Step(TimePerSubstep, events...)
		if err != nil {
			return
		}
		reward += subReward
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
	for y := 0; y < FrameHeight; y += 2 {
		for x := 0; x < FrameWidth; x += 2 {
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

func clipMouse(pos, size float64) int {
	pos = (pos + 1) / 2
	pos = math.Max(0, math.Min(1, pos))
	return int(essentials.Round(pos * size))
}
