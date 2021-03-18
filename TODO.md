### Bilateral Filter Training TODOs 
1. ~~Test on modelnet40~~
2. ~~Implement on MPEG large dataset~~
3. ~~Implement parallel training~~
4. Learn displacement vector rather than filtered position
5. ~~Calculate Std. Dev. => Impl. 10-30 std. jitter~~
6. Use specific channels for loss calc. (i.e. color only)
7. Impl. further benchmark metrics
8. Try alternative optimizers(esp. SGD)
9. ~~Why MSE mismatch? :NaN data!~~
10. ~~Smaller/faster model~~(See 19)
11. Re-generate dataset
    - How to reduce number of patches to acceptable amount: take 10 patches
    - FPS might be clustering in point clound sequence
12. Re-split dataset in in-class fashion
13. Add attention? 
14. Patch aggregation (and test)
15. MAML scheme? meta-train/test on several train labels to acquire few-shot learning
16. Add graph reg. term, i.e. $\tau x^T L x$
17. Try different layer stucture
18. Try multiple $W_ij$(i.e. edge weight) type
19. ~~Move embedding layer into BF(i.e. reduce repeated node embedding computation)(by make an alternative Weight)~~
    - ~~Need testing~~
    - Why the performance is worse? same pipeline
    - increase of batchsize? 32->64(8 on each gpu) does not degenerates the performances
    - might due to multiplied lr on embedding MLPs
    - experiment using layer-specified learning rate
20. Add flag to switch v2/v0 BFs

