package main

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var s Stacker
	serializer.RegisterTypedDeserializer(s.SerializerType(), DeserializeStacker)
}

// Stacker is an RNN block which effectively stacks input
// tensors depth wise.
// Using a Stacker, we can feed the last N frames into a
// policy at once.
type Stacker struct {
	StartState  *anydiff.Var
	HistorySize int
}

// NewStacker creates a Stacker with the correct initial
// settings.
func NewStacker(c anyvec.Creator, history int, inSize int) *Stacker {
	if inSize == 0 || history == 0 {
		panic("input and history sizes must be non-zero")
	}
	return &Stacker{
		StartState:  anydiff.NewVar(c.MakeVector(history * inSize)),
		HistorySize: history,
	}
}

// DeserializeStacker deserializes a Stacker.
func DeserializeStacker(d []byte) (*Stacker, error) {
	var res Stacker
	var vec *anyvecsave.S
	if err := serializer.DeserializeAny(d, &vec, &res.HistorySize); err != nil {
		return nil, essentials.AddCtx("deserialize Stacker", err)
	}
	res.StartState = anydiff.NewVar(vec.Vector)
	return &res, nil
}

// Start returns a start state.
func (s *Stacker) Start(n int) anyrnn.State {
	return s.funcBlock().Start(n)
}

// PropagateStart propagates through the start state.
func (s *Stacker) PropagateStart(sg anyrnn.StateGrad, g anydiff.Grad) {
	s.funcBlock().PropagateStart(sg, g)
}

// Step applies the block, returning a stacked tensor and
// updating the frame history in the state.
func (s *Stacker) Step(state anyrnn.State, in anyvec.Vector) anyrnn.Res {
	return s.funcBlock().Step(state, in)
}

// Parameters returns the Stacker's parameters.
func (s *Stacker) Parameters() []*anydiff.Var {
	return []*anydiff.Var{s.StartState}
}

// SerializerType returns the unique ID used to serialize
// a Stacker with the serializer package.
func (s *Stacker) SerializerType() string {
	return "github.com/unixpickle/rl-agents/pong_conv.Stacker"
}

// Serialize serializes the Stacker.
func (s *Stacker) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		&anyvecsave.S{Vector: s.StartState.Vector},
		s.HistorySize,
	)
}

func (s *Stacker) funcBlock() *anyrnn.FuncBlock {
	return &anyrnn.FuncBlock{
		Func: func(in, state anydiff.Res, n int) (out, newState anydiff.Res) {
			oldRows := &anydiff.Matrix{
				Data: state,
				Rows: s.HistorySize,
				Cols: state.Output().Len() / s.HistorySize,
			}

			outRows := *oldRows
			outRows.Data = anydiff.Concat(in, oldRows.Data)
			outRows.Rows++

			stateRows := outRows
			stateRows.Rows--
			stateRows.Data = anydiff.Slice(outRows.Data, 0,
				stateRows.Cols*stateRows.Rows)

			return anydiff.Transpose(&outRows).Data, stateRows.Data
		},
		MakeStart: func(n int) anydiff.Res {
			var rep []anydiff.Res
			for i := 0; i < n; i++ {
				rep = append(rep, s.StartState)
			}
			return anydiff.Concat(rep...)
		},
	}
}
