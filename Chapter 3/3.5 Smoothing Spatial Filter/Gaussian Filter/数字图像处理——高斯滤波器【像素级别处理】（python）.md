# 数字图像处理——高斯滤波器【像素级别处理】（python）

## 简介
高斯滤波通常用它来减少图像噪声以及降低细节层次。这种模糊技术生成的图像，其视觉效果就像是经过一个半透明屏幕在观察图像，这与镜头焦外成像效果散景以及普通照明阴影中的效果都明显不同。高斯平滑也用于计算机视觉算法中的预先处理阶段，以增强图像在不同比例大小下的图像效果.==从数学的角度来看，图像的高斯模糊过程就是图像与正态分布做卷积==。由于正态分布又叫作“高斯分布”，所以这项技术就叫作高斯模糊。图像与圆形方框模糊做卷积将会生成更加精确的焦外成像效果。由于高斯函数的傅立叶变换是另外一个高斯函数，所以高斯模糊对于图像来说就是一个低通滤波器。


## 代码实现



```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Fig0340.tif')  # 测试图片
H = img.shape[0]
W = img.shape[1]

#   产生5*5的Gaussian smoothing filter
# Σ=3，h(x,y)=e^((x^2+y^2)/(2*Σ*Σ))
h = np.zeros((5, 5))  # 高斯模板

for i in range(5):
    for j in range(5):
        x = i - 2
        y = j - 2
        h[i, j] = np.power(np.e, -(x * x + y * y) / 18)

h /= np.sum(h)  # 归一化处理
spanImg = np.zeros((H + 4, W + 4, 3), np.uint8)  # 5*5扩充后的图像

for i in range(H):
    for j in range(W):
        spanImg[i + 2, j + 2] = img[i, j]

blurImg = np.zeros((H, W, 3), np.uint8)  # 高斯模糊化之后的图像
for i in range(H):
    for j in range(W):
        pix = 0
        for x in range(5):
            for y in range(5):
                pix += h[x, y] * spanImg[i + x, j + y, 0]
        blurImg[i, j, 0] = round(pix)
        blurImg[i, j, 1] = blurImg[i, j, 0]
        blurImg[i, j, 2] = blurImg[i, j, 0]

plt.subplot(1, 2, 1)
plt.title('original image')
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('blur image')
plt.imshow(blurImg)
plt.axis('off')
plt.show()

```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191124190855380.png)




