```
cpu cost time: 345.982 ms
kernel minest time: 0.550912 ms
kernel maxest time: 2.21347 ms
v0 kernel avg time: 1.9308 ms
check result success!
v00 kernel avg time: 0.898016 ms
check result success!
v1 kernel avg time: 0.9208 ms
check result success!
v2 kernel avg time: 0.622163 ms
check result success!
v3 kernel avg time: 0.474714 ms
check result success!
v4 kernel avg time: 0.468986 ms
check result success!
v5 kernel avg time: 0.46841 ms
check result success!
v6 kernel avg time: 0.473338 ms
check result success!```
```

### 1.矩阵转置 mat_transpose


#### v0 全局内存的朴素转置

#### v1 共享内存简单优化

#### v2 通过padding 解决bank conflict 
#### v3 增加了每个线程处理的数据量——指令效率提升，更好利用线程寄存器
#### v4 尝试减少分支判断,微略提升
#### v5 调整线程处理数据量
#### v6 使用__restrict__，希望编译器激进,但未见成效

1.9308 ms ->  0.46841  ms提速 4.12x