# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        currentValues = util.Counter()
        i = 0
        for i in range(i, self.iterations):
            #copy current values (won't be updated and can be used for qValues)
            currentValues = self.values.copy()
            states = self.mdp.getStates()
            for state in states:
                if self.mdp.isTerminal(state):
                    self.values[state] = 0
                    continue
                else:
                    actions = self.mdp.getPossibleActions(state)
                    qValues = []
                    for action in actions:
                        qValue = 0 
                        stateAndProb = self.mdp.getTransitionStatesAndProbs(state, action)
                        for newState, prob in stateAndProb:
                            #use currentValues instead of updatedValues (self.values) for calculations
                            qValue += prob*(self.mdp.getReward(state, action, newState) + self.discount * currentValues[newState])
                        qValues.append(qValue)
                    self.values[state] = max(qValues)
            i+=1
        
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.

        qValue =
        sum of successor states (prob of newState)*[(reward of new state)+discount*value of new state]
        """
        qValue = 0 
        
        for newState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            qValue += prob*(self.mdp.getReward(state, action, newState) + self.discount*self.getValue(newState))
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        if (len(actions) == 0):
            return None
        values = []
        for action in actions:
            values.append(self.computeQValueFromValues(state,action))
        index = values.index(max(values))
        return actions[index]
        
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        i = 0
        #only update one state per iteration
        for i in range(i, self.iterations):
            currentValues = self.values.copy()
            states = self.mdp.getStates()
            #Check which state to update
            index = i%len(states)
            state = states[index]
            if self.mdp.isTerminal(state):
                self.values[state] = 0
                continue

            #calculate the qValues for the possible actions for the state 
            actions = self.mdp.getPossibleActions(state)
            qValues = []
            for action in actions:
                qValue = 0 
                stateAndProb = self.mdp.getTransitionStatesAndProbs(state, action)
                for newState, prob in stateAndProb:
                    qValue += prob*(self.mdp.getReward(state, action, newState) + self.discount * currentValues[newState])
                qValues.append(qValue)
            #update value with max qValue
            self.values[state] = max(qValues)
            i+=1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        from util import PriorityQueue
        predecessors = {}
        states = self.mdp.getStates()
        #Create sets for predecessors
        for state in states:
            predecessors[state] = set()
        #Compute predecessors of all states
        for state in states:
            if (not self.mdp.isTerminal(state)):
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    stateAndProb = self.mdp.getTransitionStatesAndProbs(state, action)
                    for successorState, prob in stateAndProb:
                        if(prob > 0):
                            predecessors[successorState].add(state)
        #initialize Empty Priority Queue
        queue = PriorityQueue()
        #For each state
        #Find the absValue of diff between current state value and max qValue
        #Push state into priQueue with -diff (minimizing queue)
        for state in states:
            if (not self.mdp.isTerminal(state)):
                actions = self.mdp.getPossibleActions(state)
                qValues = []
                for action in actions:
                    qValues.append(self.computeQValueFromValues(state, action))
                diff = abs(self.values[state] - max(qValues))
                queue.push(state, -diff)
        
        #for every iteration
        #check if queue is empty, return
        #pop state off of Queue and update value for that state
        #for each predecessor of the state
        #Find the absValue of diff between current state value and max qValue
        #if diff > theta Push state into priQueue with -diff (minimizing queue)
        
        i = 0
        for i in range(self.iterations):
            if (queue.isEmpty()):
                break
            state = queue.pop()
            if (not self.mdp.isTerminal(state)):
                actionTemp = self.computeActionFromValues(state)
                value = self.computeQValueFromValues(state, actionTemp)
                self.values[state] = value
                for predecessor in predecessors[state]:
                    if (self.mdp.isTerminal(predecessor)):
                        continue
                    actions = self.mdp.getPossibleActions(predecessor)
                    qValues = []
                    for action in actions:
                        qValue = 0 
                        stateAndProb = self.mdp.getTransitionStatesAndProbs(predecessor, action)
                        for newState, prob in stateAndProb:
                            qValue += prob*(self.mdp.getReward(predecessor, action, newState) + self.discount * self.values[newState])
                        qValues.append(qValue)
                    diff = abs(self.values[predecessor] - max(qValues))
                    if diff > self.theta:
                        queue.update(predecessor, -diff)
                i += 1
