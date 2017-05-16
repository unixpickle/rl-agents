package main

import (
	"reflect"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anydifftest"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/serializer"
)

func TestStackerOutput(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	stacker := NewStacker(c, 2, 3)
	stacker.StartState.Vector.SetData(
		c.MakeNumericList([]float64{-1, -2, -3, -4, -5, -6}),
	)

	inSeq := anyseq.ConstSeqList(c, [][]anyvec.Vector{
		{
			c.MakeVectorData(c.MakeNumericList([]float64{1, 2, 3})),
			c.MakeVectorData(c.MakeNumericList([]float64{4, 5, 6})),
			c.MakeVectorData(c.MakeNumericList([]float64{7, 8, 9})),
		},
	})

	actual := anyseq.SeparateSeqs(anyrnn.Map(inSeq, stacker).Output())
	expected := [][]anyvec.Vector{
		{
			c.MakeVectorData(c.MakeNumericList([]float64{1, -1, -4, 2, -2, -5, 3, -3, -6})),
			c.MakeVectorData(c.MakeNumericList([]float64{4, 1, -1, 5, 2, -2, 6, 3, -3})),
			c.MakeVectorData(c.MakeNumericList([]float64{7, 4, 1, 8, 5, 2, 9, 6, 3})),
		},
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected %#v but got %#v", expected, actual)
	}
}

func TestStackerGradients(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	stacker := NewStacker(c, 2, 3)
	stacker.StartState.Vector.SetData(
		c.MakeNumericList([]float64{-1, -2, -3, -4, -5, -6}),
	)
	inSeq, inVars := randomTestSequence(c, 3)
	checker := &anydifftest.SeqChecker{
		F: func() anyseq.Seq {
			return anyrnn.Map(inSeq, stacker)
		},
		V: append(inVars, stacker.Parameters()...),
	}
	checker.FullCheck(t)
}

func TestStackerSerialize(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	stacker := NewStacker(c, 2, 3)
	stacker.StartState.Vector.SetData(
		c.MakeNumericList([]float64{-1, -2, -3, -4, -5, -6}),
	)
	data, err := serializer.SerializeAny(stacker)
	if err != nil {
		t.Fatal(err)
	}
	var stacker1 *Stacker
	if err := serializer.DeserializeAny(data, &stacker1); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(stacker, stacker1) {
		t.Error("bad deserialized value")
	}
}

func randomTestSequence(c anyvec.Creator, inSize int) (anyseq.Seq, []*anydiff.Var) {
	// Taken from https://github.com/unixpickle/anynet/blob/828e924a4d86511ca8bf1ee51efab723c267c70d/anyrnn/layer_test.go#L35

	inVars := []*anydiff.Var{}
	inBatches := []*anyseq.ResBatch{}

	presents := [][]bool{{true, true, true}, {true, false, true}}
	numPres := []int{3, 2}
	chunkLengths := []int{2, 3}

	for chunkIdx, pres := range presents {
		for i := 0; i < chunkLengths[chunkIdx]; i++ {
			vec := c.MakeVector(inSize * numPres[chunkIdx])
			anyvec.Rand(vec, anyvec.Normal, nil)
			v := anydiff.NewVar(vec)
			batch := &anyseq.ResBatch{
				Packed:  v,
				Present: pres,
			}
			inVars = append(inVars, v)
			inBatches = append(inBatches, batch)
		}
	}
	return anyseq.ResSeq(c, inBatches), inVars
}
