Setting:
<pre>
+---------+
| . . . . |
| 0 0 X . |
| W . 0 . |
| . G . . |
+--------+
X location radomized

1. Stochastic Action:
	20% chance the optimal action will result in different outcome
	This way it significantly set back the training process. 
	It show signs of convergence, however it is taking very long time.
	Because 20 percent chance the agent will thought it performed optimal action
	but recieved suboptimal result, which cause the agent to doubt itself, and 
	stray from optimal action. 

2. state:
	(agent_pos, if_gold_exist, init_pos)

3. For the smooth version of plot, I used scipy savgol_filter(Total_reward,201,3)

4. reward for RL training(This is different than the evaluation)
move: -1
shoot with arrow: -1
shoot without arrow: -5
get out of the cave with gold: +100
bump into walls: -5
pick up gold: +5
fall into cave/bump into wumpus: -100


5. Evaluate matric for both logic and RL
move: -1
shoot: -1
get out of the cave with gold: 100

6. Result

For logic agent:
Tested over 11 possible initial position
score: 

For RL:
Average score over 1000 times random initial position
score: 8 out of 100

</pre>
