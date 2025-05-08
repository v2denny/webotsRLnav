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
5. sac (continuous action space)   ####### update reward funct before training ########
6. td3 (continuous action space)

## training steps:
1. ppo (25k, 125k, 100k, 750k, 1M) leverages curr learning very well. robust and fast results even for complex tasks.
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
2. same spot steps: instead of reseting, made it that when steps in same spot>=10 then always penalizes
3. a2c and dqn seem to not notice the small changes in the reward: scalled the rewards and added distance based reward instead of just penalty
4. decreased new location reward so the bot doesnt prioritize exploration over exploitation
