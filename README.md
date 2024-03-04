# An ENN-based solution for the TensorRTS game.

### Note: load model
To run load_model from checkpoints successfully, you must patch `/opt/conda/envs/games38/lib/python3.8/site-packages/entity_gym/env/vec_env.py`
(Your path will be difference from mine.)

https://github.com/entity-neural-network/enn-trainer/issues/32

### Save model (checkpoints)
To generate the model, run the training this way:
```bash
python train.py --config=config.ron --checkpoint-dir=checkpoints
```

### Example command line scenario:
```bash
(games38) @drchangliu ➜ /workspaces/RL4SE/enn/TensorRTS (main) $ python TensorRTS.py 
LinearRTS -- Mapsize: 32
Environment: TensorRTS
Entity Cluster: position, dot
Entity Tensor: position, dimension, x, y
Categorical Move: advance, retreat, rush, boom

 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 #
 7   6             2                         2             6   7 ##
            2-0                                    2-0             ##
Step 0
Reward: 0.0
Total: 0.0
Entities
0 Cluster(position=0, dot=7) (id=('Cluster', 0))
1 Cluster(position=2, dot=6) (id=('Cluster', 1))
2 Cluster(position=9, dot=2) (id=('Cluster', 2))
3 Cluster(position=22, dot=2) (id=('Cluster', 3))
4 Cluster(position=29, dot=6) (id=('Cluster', 4))
5 Cluster(position=31, dot=7) (id=('Cluster', 5))
6 Tensor(position=6, dimension=1, x=2, y=0) (id=('Tensor', 0))
7 Tensor(position=25, dimension=1, x=2, y=0) (id=('Tensor', 1))
Choose Move (0/advance 1/retreat 2/rush 3/boom)
0 
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 #
 7   6             2                         2             6   7 ##
              2-0                                  2-0             ##
Step 1
Reward: 0.0
Total: 0.0
Entities
0 Cluster(position=0, dot=7) (id=('Cluster', 0))
1 Cluster(position=2, dot=6) (id=('Cluster', 1))
2 Cluster(position=9, dot=2) (id=('Cluster', 2))
3 Cluster(position=22, dot=2) (id=('Cluster', 3))
4 Cluster(position=29, dot=6) (id=('Cluster', 4))
5 Cluster(position=31, dot=7) (id=('Cluster', 5))
6 Tensor(position=7, dimension=1, x=2, y=0) (id=('Tensor', 0))
7 Tensor(position=25, dimension=1, x=2, y=0) (id=('Tensor', 1))
Choose Move (0/advance 1/retreat 2/rush 3/boom)
0
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 #
 7   6             2                         2             6   7 ##
                2-0                                2-0             ##
Step 2
Reward: 0.0
Total: 0.0
Entities
0 Cluster(position=0, dot=7) (id=('Cluster', 0))
1 Cluster(position=2, dot=6) (id=('Cluster', 1))
2 Cluster(position=9, dot=2) (id=('Cluster', 2))
3 Cluster(position=22, dot=2) (id=('Cluster', 3))
4 Cluster(position=29, dot=6) (id=('Cluster', 4))
5 Cluster(position=31, dot=7) (id=('Cluster', 5))
6 Tensor(position=8, dimension=1, x=2, y=0) (id=('Tensor', 0))
7 Tensor(position=25, dimension=1, x=2, y=0) (id=('Tensor', 1))
Choose Move (0/advance 1/retreat 2/rush 3/boom)
0
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 #
 7   6             0                         2             6   7 ##
                  4-0                              2-0             ##
Step 3
Reward: 0.0
Total: 0.0
Entities
0 Cluster(position=0, dot=7) (id=('Cluster', 0))
1 Cluster(position=2, dot=6) (id=('Cluster', 1))
2 Cluster(position=9, dot=0) (id=('Cluster', 2))
3 Cluster(position=22, dot=2) (id=('Cluster', 3))
4 Cluster(position=29, dot=6) (id=('Cluster', 4))
5 Cluster(position=31, dot=7) (id=('Cluster', 5))
6 Tensor(position=9, dimension=1, x=4, y=0) (id=('Tensor', 0))
7 Tensor(position=25, dimension=1, x=2, y=0) (id=('Tensor', 1))
Choose Move (0/advance 1/retreat 2/rush 3/boom)
0
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 #
 7   6             0                         2             6   7 ##
                    4-0                            2-0             ##
Step 4
Reward: 0.0
Total: 0.0
Entities
0 Cluster(position=0, dot=7) (id=('Cluster', 0))
1 Cluster(position=2, dot=6) (id=('Cluster', 1))
2 Cluster(position=9, dot=0) (id=('Cluster', 2))
3 Cluster(position=22, dot=2) (id=('Cluster', 3))
4 Cluster(position=29, dot=6) (id=('Cluster', 4))
5 Cluster(position=31, dot=7) (id=('Cluster', 5))
6 Tensor(position=10, dimension=1, x=4, y=0) (id=('Tensor', 0))
7 Tensor(position=25, dimension=1, x=2, y=0) (id=('Tensor', 1))
Choose Move (0/advance 1/retreat 2/rush 3/boom)

```
