import gym
from gym import wrappers
import numpy as np
import random

# Use policy gradients to train a linear model.
# Achieves a good success rate after 500 trials.

numFeatures = 8
numActions = 2

def softmax(vec):
    divisor = np.sum(np.sum(np.exp(vec)))
    return np.exp(vec) / divisor

def trial(env, policy):
    obs = env.reset()
    gradient = np.zeros((numActions, numFeatures))
    totalReward = 0
    while True:
        env.render()
        features = envFeatures(obs)
        obsColumn = np.reshape(features, (numFeatures, 1))
        policyOut = softmax(np.dot(policy, obsColumn).T[0])

        # Sample from softmax and compute upstream gradient.
        action = 0
        softmaxUpstream = [policyOut[1], -policyOut[1]]
        if random.random() > policyOut[0]:
            softmaxUpstream = [-policyOut[0], policyOut[0]]
            action = 1

        # Add the score to the gradient.
        gradient[0] += features * softmaxUpstream[0]
        gradient[1] += features * softmaxUpstream[1]

        obs, rew, done, info = env.step(action)
        totalReward += rew

        if done:
            return (gradient, totalReward)

def main():
    policy = np.zeros((numActions, numFeatures))

    batchSize = 10
    stepSize = 0.02
    baseline = 10
    numTrials = 0

    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, "/tmp/gym-results")
    for i in range(0, 70):
        stepSize *= 0.95
        totalGrad = np.zeros((numActions, numFeatures))
        totalReward = 0
        for i in range(0, batchSize):
            gradient, reward = trial(env, policy)
            totalGrad += gradient * (reward - baseline)
            totalReward += reward
            numTrials += 1

        baseline = totalReward / batchSize
        policy += totalGrad * (stepSize / batchSize)

        print('%d trials: reward=%f step=%f' % (numTrials, baseline, stepSize))

    env.close()
    #gym.upload("/tmp/gym-results", api_key="")

def envFeatures(obs):
    return np.concatenate((obs, obs*obs), axis=0)

main()
