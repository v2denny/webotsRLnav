# webotsRLnav
Mobile Robot Basic Navigation Using RL

## objectives
1. train/test and benchmark diff RL algs
2. curriculum learning (difficulty levels)

## questions before benchmarking diff algs:
1. 100ms or 200ms action? 200ms>> why tho?
2. 3 or 4 action space? 3act converges faster. 4th action (backwards) just makes the env more complicated and is not worth it.
3. problems whit corric learning: how many steps per 'level' so the model doesnt overfit and affects the next level? really deppends on the alg.

## algs to test:
1. ppo (discrete action space) action: 1 or 2 or 3 - apply action
2. a2c (discrete action space)
3. dqn (discrete action space)
4. ddpg (continuous action space) action: (linear,angular) - apply move linear,angular
5. sac (continuous action space)
6. td3 (continuous action space)

## training steps:
1. ppo (25k, 125k, 50k (to prevent overfit on the easy positions), 750k, 1M) leverages curr learning very well. robust and fast results even for complex tasks.
2. a2c (25k, 50k-overfit (show action dist start vs easy = same), na, 500k, na) a2c doesnt benefit from curr learning. just handles simple tasks.
3. dqn (25k, 125k-overfit (not as much as a2c), 250k, na, na)
4. ddpg ()
5. sac ()
6. td3 ()

to show the overfit of a2c and dqn: show action distribution of start vs easy, they are probably ver alike
to show how a2c and dqn dont benefit from curr learning: compare to easy ppo
to show if a2c and dqn are better without curr learning: compare currlearn easy vs fresh start easy


## some reward adjustments after training some ppo, a2c and dqn (had to re-train everything):
1. noticed bot didnt 'fear' the obstacles: increment penalty the closest he is to the wall, instead of just treshold and constant value
2. same spot steps: instead of reseting, made it that when steps in same spot>=5 then always penalizes
3. a2c and dqn seem to not notice the small changes in the reward: scalled the rewards and added distance based reward instead of just penalty
4. decreased new location reward so the bot doesnt prioritize exploration over exploitation
5. noticed that bot went straight to the target but chose to hang out for a bit close to it before finishing the episode: increase time penalty from 0.5 to 1 to make each step "more expensive"; added a decreasing reward gradient when very close to the target (< 0.2m) so the bot gets less marginal benefit from hovering extremely close vs. just reaching the target; stricter "same spot" penalty that reduced the threshold from 10 steps to 5 steps, increased the penalty from -2.5 to -5.0 so it makes "hanging around" the target much more costly; added a more sophisticated proximity reward structure that gives diminishing returns when extremely close.
6. sometimes the robot just does left-right when facing a wall as if it was 'locked': reward robot if he keeps rotating in just one direction when stuck facing a wall
7. robot still prioritizes hovering near target over finishing episode: increased target reward;
8. robot gets stuck on corners/long walls: reduced angle reward overall and when corner is detected; increased exploration reward; reward for turning in same direction if close to wall is detected

## some env changes (had to retrain everything):
1. sometimes didnt detect collisions and keps running the episode: added collision chech before actions; improved collision detection
2. changed from 9 to 15 lidar rays, removed ray clipping
4. obstacle avvoidance isnt that good, sometimes the bot just collides with the wall without even trying to avoid: improved lidar readings and lidar position in webots env
