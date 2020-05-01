Setting:
<pre>
+---------+  <br />
| . . . . |  <br />
| 0 0 X . |  <br />
| W . 0 . |  <br />
| . G . . |  <br />
+--------+  <br />
</pre>
X location radomized

1. Stochastic Action:
	20% chance it will not move as we talked during meeting

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
score: 95, -2, 93, -8, -9, -2, -8, -9, -8, -9, -9

For RL:
Average score over 1000 times random initial position
score: 92


