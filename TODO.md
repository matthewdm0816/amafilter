### Bilateral Filter Training TODOs 
1. Learn displacement vector rather than filtered position
6. Use specific channels for loss calc. (i.e. color only)
11. Re-generate dataset
    - ~~How to reduce number of patches to acceptable amount: take 10 patches~~
    - FPS might be clustering in point cloud sequence
12. Re-split dataset in in-class fashion
14. Patch aggregation (and test)
15. MAML scheme? meta-train/test on several train labels to acquire few-shot learning
16. Try different layer stucture
17. Try multiple $W_{ij}$(i.e. edge weight) type
18. Add flag to switch v2/v0 BFs, add argparsers
19. Dynamic as graph connection => t as feature
20. More noise type: blur/scanner
21. PCA+ICA baseline
    - Infomax ~ Yet another loss!
    - Kurtosis/negentropy: use sklearn
    - Need Test `pcaica.py`
22. Baselines in paper
23. B-spline interpolation?
24. Try lighter mlps
    - seems not reducing too much, since MLPs are not the bottleneck
25. What parameters are there in our network?
    - MLP, $f_\phi(x)$ for each layer
    - $\Theta$, feature projection for each layer
    - Merge $\Theta, f_\phi$?
      - Need test, not necessary
    - => Try 1/2 layer of BFs?
1.  Adversarial noise generator
    - Shared param?
    - Bad generalization: good train loss(0.37), bad on test(0.65)
    - comparatively slow(5-10 times slower)
    - pause for now
2. Impl. YAML config parser
3. Add Configurator class
    - Use it.

### Doing List
21. **Large PC eval test**
    - Overlap patches?
    - Need test
2.  **Large-scale test**
    - cosine annealing with **normal-paced** warmup restarts(LR Plan)
3.  **Investigate other methods**
    - MRPCA
    - ...
4. **Visualize denoising result**
    - Need test
    - RAW0-10.0: too smoothed. lossed match, but too smoothed
    - Try without regularization!
    - Try small sigma datasets

### Future Directions

1. Fuzzy
2. Meta-Learning 
3. Adversarial
4. Time Series


### Comparisons
| $\sigma$              | 1        | 5         | 10             |
| --------------------- | -------- | --------- | -------------- |
| Original              | 1        | 25        | 100            |
| Plain BF              | 0.30     | 8.76      |                |
| BF(-act)              | 0.13     | 0.50      |                |
| BF(act)               | 0.13     | 0.425@340 | 0.543@50       |
| BF(AW)                | On Query | On Query  | 0.529@290      |
| BF(RA)                | 0.188    | 0.368@300 | 0.510@290 \*\* |
| BF(RAW(30))           |          |           | 0.533@140      |
| BF(RAW(100))          | On Query | On Query  | \*\*\*         |
| BF(RAW+Cauchy kernel) | On Query | 0.455@160 | 0.506@165      |
| BF(AR+CM)             |          | 0.617@50  |                |
| BF(AR+singleMLP)      |          | <0.55@10  |                |
| Adversarial Noisy     |          | *         |                |
| MoNet $\times 4$(act) |          | 0.47@250  |                |
| DGCNN(-act)           |          | 0.74      |                |
| DGCNN(act)            |          | 0.731@370 |                |
| GAT(-act)             |          | <0.93     |                |
| GAT(act)              |          | 0.75@290  |                |
- \* on Ad noise: 0.267, on Gaussian: 0.657
- \*\* $\lambda=0.01$, minor improvement
- \*\*\* minor changes
- R ~ w/ regularization
- A ~ activation between filter layers
- W ~ warmup restart each 100 epoch
- Cauchy kernel: $f(x)=\frac{1}{1+x^2}$

- CM: Chamfer Measure:
    $$
    C(S,\hat S)=\frac 1 {|S|}\sum_{s\in S}\min_{\hat s'\in \hat S}\|s-\hat s'\|^2 + \frac 1 {|\hat S|}\sum_{\hat s\in \hat S}\min_{s'\in S}\|s'-\hat s\|^2
    $$
    - Chamfer loss does not make sense on Color denoising
- All comparisons @ 100 epochs
- w/ or w/o activation for GAT seems has no difference on denoising
- On SGD+M/Nesterov: slightly better generalization compared to Adam
- MoNet: wavy performance on test set. although consistently on train set
- ConsineAnealing appears to be enhancing performance after 2 $T_max$, i.e. re-warmup turn

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
   w_{ij}=f_c(\|\bold \phi(\bold x'_i)-\phi(\bold x'_j)\|^2)\\
   f_c(x)\in\{e^{-x^2}(\text{高斯}), \frac 1 {1+x^2}\text{(柯西)}\}
   $$
   - use low-rank orthogonal kernel:
    $$
    w_{ij} = {x'}_i^TLL^T{x'}_j, \bold L\in \R^{n\times l}
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
17. ~~Add graph reg. term, i.e. $\tau x^T L x$~~
25. ~~Add MoNet baseline~~
    - ~~
26. ~~Try alternative optimizers(esp. SGD)~~
    - Tried: SGD with M+Nesterov
13. ~~Add attention? ~~
    - Not neccessary
14. Manifold-Manifold/Chamfer Measure distance as loss
    - ~~Use existing Chamfer Loss CUDA impl.~~
7. ~~Impl. further benchmark metrics~~
