# Documentation
Here we explain some technical terms about reinforcement learning algorithms and related terminology.
More definitions and explanations will be added as we advance in our research.

### Baseline
The less effort result we have to beat, for example, if a random set of movements can produce a score of 100, then 100 is the baseline to beat.

### Reproducibility
As we are producing outputs based on a predefined algorithm we want to set the random values to a certain value to make our results predictable and deterministic. This way, if other researchers want they can obtain the same results in each of the runs of the algorithm by setting the same seed.
This seed can and should be fixed at the beginning of our program by invoking random.seed(<number>).

### Deep learning
Deep reinforcement learning instead uses a neural network to approximate the Q-function.

### Policy
Decision-making function, accepts a state as input and “decides” on an action.
policy: state -> action

### Q-learning algorithm
Step 1. Create Q-table, all zeroes
Step 2. Choose an action
Step 3. Perform an action
Step 4. Measure the reward
Step 5. Update the Q-table

### Q-table
A simple lookup table where we calculate the maximum expected future rewards for action at each state.
This table will guide us to the best action at each state.

state	action	reward
state0	shoot	10
state0	right	3
state0	left	3

### Q-function
A Q-table is not sustainable as games are increasingly complex. For this reason, we need a way to approximate the Q-table, as most games have too many states to list in the table. In that cases, the Q-learning agent learns a Q-function instead of a Q-table.

Q(state, action) = reward

### Epsilon
It is a value that ranges from 1 to 0, when its near 1, our algorithm will be in the explotation phase, that means that it will try random movements. When its near 0, our algorithm will be in an exploration phase, it will try the movements from its memory that have a higher confidence. In each step, the algorithm chooses to perform a random or predefined movement depending on the epsilon value, that decays through time. 

# Bibliography
[https://junedmunshi.wordpress.com/2012/03/30/how-to-implement-epsilon-greedy-strategy-policy/](https://junedmunshi.wordpress.com/2012/03/30/how-to-implement-epsilon-greedy-strategy-policy/)

[https://medium.com/@dennybritz/exploration-vs-exploitation-f46af4cf62fe](https://medium.com/@dennybritz/exploration-vs-exploitation-f46af4cf62fe)

https://www.digitalocean.com/community/tutorials/how-to-build-atari-bot-with-openai-gym

https://www.freecodecamp.org/news/an-introduction-to-q-learning-reinforcement-learning-14ac0b4493cc/
