# racer
Black-box, gradient-free optimization of car-racing policies.

## Results
| Method Name                                        | Max Reward |    Mean Reward | Mean # Function evaluations to reach 900 |
|----------------------------------------------------|-----------:|---------------:|-----------------------------------------:|
| Nelder Mead                                        |      713.2 | Not Applicable |                           Not Applicable |
| NN + Generation-based Evolution Strategy (3 repetitions) |      925.7 |          923.3 |                                     9.8k |
| NN + Iterative Evolution Strategy (4 repetitions)  |      915.5 |          906.3 |                                    50.3k |
| Genetic Program (4 repetitions)                    |      928.2 |          917.2 |                                     **5.8k** |
| Tuned Genetic Programming                          |      **930.6** | Not Applicable |                           Not Applicable |

Our best results were achieved by fine tuning constants of the best genetic program using evolution strategies.
## Environment
Graphics, physics engine and reward calculation adapted from [OpenAI gym](https://github.com/openai/gym).

To improve performance, we rewrote the graphics pipeline yielding ~40x sequential speedup.
Our modifications allow the evaluations to run headless and be parallelized.
All experiments were performed on the ETH Euler Supercomputer.
