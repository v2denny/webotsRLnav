# webotsRLnav
Mobile Robot Basic Navigation Using RL

## objectives
1. train/test and benchmark diff RL algs
2. curriculum learning (difficulty levels)

## questions before benchmarking diff algs:
1. 100ms or 200ms action? 200ms>>
2. 3 or 4 action space? 3act converges faster
3. problems whit corric learning: how many steps per 'level' so the model doesnt overfit and affects the next level?

## algs to test:
1. ppo (discrete action space)
2. a2c (discrete action space)
3. dqn (discrete action space)
4. ddpg (continuous action space)
5. sac (continuous action space)
6. td3 (continuous action space)

## training steps:
1. ppo (25k, 125k, 100k, 750k, 1M)
2. a2c (25k, 50k-overfit, na, 1M, 1M) a2c doesnt benefit from curr learning
3. dqn ()
4. ddpg ()
5. sac ()
6. td3 ()
