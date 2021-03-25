### Bilateral Filter Training TODOs 
1. Learn displacement vector rather than filtered position
6. Use specific channels for loss calc. (i.e. color only)
7. Impl. further benchmark metrics
8. Try alternative optimizers(esp. SGD)
    - Tried: SGD with M+Nesterov
11. Re-generate dataset
    - ~~How to reduce number of patches to acceptable amount: take 10 patches~~
    - FPS might be clustering in point clound sequence
12. Re-split dataset in in-class fashion
13. Add attention? 
14. Patch aggregation (and test)
15. MAML scheme? meta-train/test on several train labels to acquire few-shot learning
17. ~~Add graph reg. term, i.e. $\tau x^T L x$~~
18. Try different layer stucture
19. Try multiple $W_{ij}$(i.e. edge weight) type

21. Add flag to switch v2/v0 BFs, add argparsers
22. Dynamic as graph connection => t as feature
23. Adversarial noise generator
    - Shared param?
24. Large PC eval test
25. ~~Add MoNet baseline~~
    - Need for test
26. PCA+ICA baseline
    - Infomax ~ Yet another loss!
    - Kurtosis/negentropy: use sklearn
27. Baselines in paper
28. Manifold-Manifold/Chamfer Measure distance as loss
    - Use existing Chamfer Loss CUDA impl. 

### Future Directions

1. Fuzzy
2. Meta-Learning 
3. Adversarial
4. Time Series


### Comparisons
| $\sigma$                 | 1    | 5         | 10        |
| ------------------------ | ---- | --------- | --------- |
| Original                 | 1    | 25        | 100       |
| Plain BF                 | 0.30 | 8.76      |           |
| AmaBF(w/o act)           | 0.13 | 0.50      |           |
| AmaBF(w/ act)            | 0.13 | 0.425@340 | <0.543@50 |
| AmaBF(w/ act+g. reg.)    |      | 0.368@300 |           |
| MoNet $\times 4$(w/ act) |      | 0.47@250 |           |
| DGCNN(w/o act)           |      | 0.74      |           |
| DGCNN(w/ act)            |      | 0.731@370 |           |
| GAT(w/o act)             |      | <0.93     |           |
| GAT(w/ act)              |      | 0.75@290  |           |

- All comparisons @ 100 epochs
- w/ or w/o activation for GAT seems has no difference on denoising
- On SGD+M/Nesterov: slightly better generalization compared to Adam
- MoNet: wavy performance on test set. do reduce consistently on train set

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
   Note: using original postion in relative position
   Note: Covariance $\bold \Sigma$ is diagonal
   - Can this be like in form of Cholesky Decomp. $\bold \Sigma = \bold \Gamma \bold \Gamma^T, \bold \Gamma \in \R^{n\times k}$
   - Cholesky Decomp. is low-rank decomp. any further?

6. AmaFilter(BF): activation-free is OK
   $$
   \bold X^{l+1}=\sigma(\bold D^{-1}_{W}\bold W\bold X^{l}\bold \Theta)\\
   w_{ij}=\exp(-\|\bold \phi(\bold x'_i)-\phi(\bold x'_j)\|^2)
   $$

### Done TODOs

1. ~~Test on modelnet40~~
2. ~~Implement on MPEG large dataset~~
3. ~~Implement parallel training~~
4. ~~Calculate Std. Dev. => Impl. 10-30 std. jitter~~
9.  ~~Why MSE mismatch? :NaN data!~~
10. ~~Smaller/faster model~~
11. ~~Move embedding layer into BF(i.e. reduce repeated node embedding computation)(by make an alternative Weight)~~
    - ~~Need testing~~
    - ~~Why the performance is worse? same pipeline~~
    - ~~increase of batchsize? 32->64(8 on each gpu) does not degenerates the performances~~
    - ~~might due to multiplied lr on embedding MLPs~~
    - ~~experiment using layer-specified learning rate~~
    - ~~result: changed $X$ to $f_\phi(X)$ to filter => degenerates perf.~~
