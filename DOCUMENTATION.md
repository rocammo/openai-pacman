
# Documentation
Here we explain some technical terms about Deep Q.

Epsilon: is a value that ranges from 1 to 0, when its near 1, our algorithm will be in the experiment phase, that means that it will try random movements. When its near 0, our algorithm will be in an exploration phase, it will try the movements from its memory that have a higher confidence. In each step, the algorithm chooses to perform a random or predefined movement depending on the epsilon value, that decays through time. 

# Bibliography
[https://junedmunshi.wordpress.com/2012/03/30/how-to-implement-epsilon-greedy-strategy-policy/](https://junedmunshi.wordpress.com/2012/03/30/how-to-implement-epsilon-greedy-strategy-policy/)

[https://medium.com/@dennybritz/exploration-vs-exploitation-f46af4cf62fe](https://medium.com/@dennybritz/exploration-vs-exploitation-f46af4cf62fe)
