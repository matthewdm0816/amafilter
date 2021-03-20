### Bilateral Filter Training TODOs 
1. ~~Test on modelnet40~~
2. ~~Implement on MPEG large dataset~~
3. ~~Implement parallel training~~
4. Learn displacement vector rather than filtered position
5. ~~Calculate Std. Dev. => Impl. 10-30 std. jitter~~
6. Use specific channels for loss calc. (i.e. color only)
7. Impl. further benchmark metrics
8. Try alternative optimizers(esp. SGD)
    - Tried: SGD with M+Nesterov
9.  ~~Why MSE mismatch? :NaN data!~~
10. ~~Smaller/faster model~~(See 19)
11. Re-generate dataset
    - ~~How to reduce number of patches to acceptable amount: take 10 patches~~
    - FPS might be clustering in point clound sequence
12. Re-split dataset in in-class fashion
13. Add attention? 
14. Patch aggregation (and test)
15. MAML scheme? meta-train/test on several train labels to acquire few-shot learning
17. Add graph reg. term, i.e. $\tau x^T L x$
18. Try different layer stucture
19. Try multiple $W_{ij}$(i.e. edge weight) type
20. ~~Move embedding layer into BF(i.e. reduce repeated node embedding computation)(by make an alternative Weight)~~
    - ~~Need testing~~
    - Why the performance is worse? same pipeline
    - increase of batchsize? 32->64(8 on each gpu) does not degenerates the performances
    - might due to multiplied lr on embedding MLPs
    - experiment using layer-specified learning rate
    - result: changed $X$ to $f_\phi(X)$ to filter => degenerates perf.
21. Add flag to switch v2/v0 BFs, add argparsers
22. Dynamic as graph connection => t as feature
22. Relative postion as DGCNN?

### Comparisons
| $\sigma$       | 1    | 5         | 10  |
| -------------- | ---- | --------- | --- |
| Original       | 1    | 25        | 100 |
| Plain BF       | 0.30 | 8.76      |     |
| AmaBF(w/o act) | 0.13 | 0.50      |     |
| DGCNN(w/o act) |      | 0.74      |     |
| DGCNN(w/ act)  |      | 0.731@370 |     |
| GAT(w/o act)   |      | <0.93     |     |
| GAT(w/ act)    |      | 0.75@290  |     |

- All comparisons @ 100 epochs
- w/ or w/o activation for GAT seems has no difference on denoising
- On SGD+M/Nesterov: slightly better generalization compared to Adam

### Theory Backgrounds

1. GATs
   $$
   \bold X^{l+1}=\sigma(||_k\bold D^{-1}_{W^k}\bold W^k\bold X^{l}\bold \Theta^k)\\
   w_{ij}=\exp(f(\bold x'_i, \bold x'_j))
   $$

2. GCNs
   $$
   \bold X^{l+1}=\sigma(\bold D^{-1/2}_{W}\bold W\bold D^{-1/2}_{W}\bold X^{l}\bold \Theta)\\
   \bold W=\bold A+\bold I
   $$
3. SGCs
   $$
   \bold X^{K}=\sigma((\bold D^{-1/2}_{W}\bold W\bold D^{-1/2}_{W})^K \bold X^{0} \bold \Theta)\\
   \bold W=\bold A+\bold I
   $$

4. DGCNNs
   $$
   \bold X^{l+1}=\sigma(\bold W\bold X^{l}\bold \Theta)\\
   w_{ij}=\exp(f(\bold x'_i||\bold x'_j-\bold x'_i)\approx f(\bold x'_i, \bold x'_j-\bold x'_i))
   $$

5. MoNet
   $$
   \bold X^{l+1}=\sigma(\bold D^{-1}_{A}\frac 1 K\sum_{k=[K]} \bold W^k\bold X^{l}\bold \Theta^k)\\
   w_{ij}=\exp(-\frac 1 2 (\bold {e'}_{ij}^T\bold \Sigma^{-1}\bold {e'}_{ij}))\\
   \bold {e'}_{ij} = \bold x_i - \bold x_j
   $$

6. AmaFilter(BF): activation-free
   $$
   \bold X^{l+1}=\bold D^{-1}_{W}\bold W\bold X^{l}\bold \Theta\\
   w_{ij}=\exp(-\|\bold \phi(\bold x'_i)-\phi(\bold x'_j)\|^2)
   $$

### Future Directions

1. Fuzzy
2. Meta-Learning 