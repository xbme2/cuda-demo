```
cpu cost time: 18.314 ms        8.38861e+06
native kernel cost time: 0.234496 ms
check result success!
t0plus share memory kernel cost time: 0.377856 ms
check result success!
share memory kernel cost time: 0.371712 ms
check result success!
v2  kernel cost time: 0.197536 ms
check result success!
v3  kernel cost time: 0.11776 ms
check result success!
v4  kernel cost time: 0.118592 ms
check result success!
v5  kernel cost time: 0.111616 ms
check result success!
v6  kernel cost time: 0.113664 ms
check result success!
```

### reduce 之sum求和
我没招了,我不知道为什么使用__shfl_down_sync几乎没有提升
以及最朴素的全局内存实现比共享内存好不少
从v3减少同步开始才超过朴素实现
我猜测是L1缓存的功能,一次昂贵的全局内存+N次L1缓存命中 优于 2N次共享内存, 所以耗时是两倍少一点
估计当每个元素重用次数更多时会反转,在这个情况，每个元素只使用log(BLOCK_SZ)= 8次  

- 参考了“[GiantPandaCV 博客](http://giantpandacv.com/project/CUDA/%E3%80%90BBuf%E7%9A%84CUDA%E7%AC%94%E8%AE%B0%E3%80%91%E4%B8%89%EF%BC%8Creduce%E4%BC%98%E5%8C%96%E5%85%A5%E9%97%A8%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/#reduce)”
- v6参考   [pytorch源码](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/block_reduce.cuh)
