// Slow and crappy hill-climbing algorithm for finding
// decision tree policy for CartPole.

package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"

	"github.com/unixpickle/gym-socket-api/binding-go"
	"github.com/unixpickle/rip"
)

const (
	Host        = "127.0.0.1:5001"
	Population  = 20
	PolicyDepth = 2
	MutateProb  = 0.05
)

func main() {
	policy := NewPolicy(PolicyDepth)
	envs := make([]gym.Env, Population)
	log.Printf("Creating %d environments...", Population)
	for i := range envs {
		var err error
		envs[i], err = gym.Make(Host, "CartPole-v0")
		must(err)
	}

	r := rip.NewRIP()

	var batch int
	for !r.Done() {
		policies := make([]PolicyNode, Population)
		policies[0] = policy
		for i := 1; i < Population; i++ {
			policies[i] = policy.Copy()
			policies[i].Mutate(MutateProb)
		}
		rewards := Rollouts(policies, envs)
		var maxReward float64
		policy, maxReward = BestPolicy(policies, rewards)

		log.Printf("batch=%d max_reward=%f", batch, maxReward)
		batch++
	}

	fmt.Println(policy)
}

func Rollouts(policies []PolicyNode, envs []gym.Env) []float64 {
	res := make([]float64, len(envs))
	var wg sync.WaitGroup
	for i, p := range policies {
		wg.Add(1)
		go func(p PolicyNode, e gym.Env, reward *float64) {
			defer wg.Done()
			obs, err := e.Reset()
			must(err)
			var done bool
			for !done {
				var obsVec []float64
				must(obs.Unmarshal(&obsVec))
				var rew float64
				obs, rew, done, _, err = e.Step(p.Decide(obsVec))
				must(err)
				*reward += rew
			}
		}(p, envs[i], &res[i])
	}
	wg.Wait()
	return res
}

func BestPolicy(policies []PolicyNode, rewards []float64) (policy PolicyNode, maxReward float64) {
	maxReward = math.Inf(-1)
	for i, r := range rewards {
		if r > maxReward {
			maxReward = r
			policy = policies[i]
		}
	}
	return
}

func NewPolicy(depth int) PolicyNode {
	if depth == 0 {
		return &LeafNode{Decision: rand.Intn(2)}
	}
	res := &BranchNode{
		Left:  NewPolicy(depth - 1),
		Right: NewPolicy(depth - 1),
	}
	res.Mutate(1)
	return res
}

type PolicyNode interface {
	fmt.Stringer

	Mutate(prob float64)
	Decide(in []float64) int
	Copy() PolicyNode
}

type BranchNode struct {
	Param  int
	Thresh float64
	Left   PolicyNode
	Right  PolicyNode
}

func (b *BranchNode) String() string {
	return fmt.Sprintf("if obs[%d] < %f {\n%s\n} else {\n%s\n}",
		b.Param, b.Thresh, b.Left.String(), b.Right.String())
}

func (b *BranchNode) Mutate(prob float64) {
	if rand.Float64() < prob {
		b.Param = rand.Intn(4)
	}
	if rand.Float64() < prob {
		b.Thresh = randomThreshold(b.Param)
	}
	b.Left.Mutate(prob)
	b.Right.Mutate(prob)
}

func (b *BranchNode) Decide(in []float64) int {
	if in[b.Param] > b.Thresh {
		return b.Right.Decide(in)
	} else {
		return b.Left.Decide(in)
	}
}

func (b *BranchNode) Copy() PolicyNode {
	return &BranchNode{
		Param:  b.Param,
		Thresh: b.Thresh,
		Left:   b.Left.Copy(),
		Right:  b.Right.Copy(),
	}
}

type LeafNode struct {
	Decision int
}

func (l *LeafNode) String() string {
	return fmt.Sprintf("return %d", l.Decision)
}

func (l *LeafNode) Mutate(prob float64) {
	if rand.Float64() < prob {
		l.Decision = rand.Intn(2)
	}
}

func (l *LeafNode) Decide(in []float64) int {
	return l.Decision
}

func (l *LeafNode) Copy() PolicyNode {
	return &LeafNode{Decision: l.Decision}
}

func randomThreshold(param int) float64 {
	return rand.NormFloat64() * 10
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
