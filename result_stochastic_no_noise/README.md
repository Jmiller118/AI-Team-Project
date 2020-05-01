Setting:
+---------+
| . . . . |
| 0 0 X . |
| W . 0 . |
| . G . . |
+--------+
X location radomized

1. Stochastic Action:
	20% chance to do random action, but the action's outcome will recorded correctly in q-table

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
score: 72 out of 100


