# 机器人学中的状态估计

> latex: ![](https://latex.codecogs.com/gif.latex?\pi)

## 引言

机器人：
+ 状态估计 state estimation
+ 控制 control

## 概率论

### 高斯概率密度

一维

![](https://latex.codecogs.com/gif.latex?p(x\mid\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}exp(-\frac{1}{2}\frac{(x-\mu)^2}{\sigma^2}))

多维

![](https://latex.codecogs.com/gif.latex?p(x\mid\mu,\Sigma)=\frac{1}{\sqrt{(2\pi)^{N}det\Sigma}}exp(-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)))


![][latex p(x\mid\mu,\Sigma)=\frac{1}{\sqrt{(2\pi)^{N}det\Sigma}}exp(-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu))]

[latex]: https://latex.codecogs.com/gif.latex?
