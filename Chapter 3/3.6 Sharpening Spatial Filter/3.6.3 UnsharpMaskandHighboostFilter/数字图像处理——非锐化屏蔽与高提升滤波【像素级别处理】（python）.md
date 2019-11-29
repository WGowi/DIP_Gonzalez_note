# 数字图像处理——非锐化屏蔽与高提升滤波【像素级别处理】（python）

## 非锐化屏蔽

### 介绍
其中非锐化屏蔽是指在原图像中减去一个经过锐化的图层

其步骤为：

 1. 模糊原图像
 2. 原图像减去模糊图像(差值为模版)
 3. 把模版加到原图像上

 令$\overline{f}(x,y)$为模糊图像
 首先我们可以得到模版
$g_{mask}(x,y)=f(x,y)-\overline{f}(x,y)$
然后在原图像加上模版的一个权重k(k>0)：
$g(x,y)=f(x,y)+k*\overline{f}(x,y)$      
当k=1时，所得的为非锐化屏蔽

## 高提升滤波


### 介绍
其中非锐化屏蔽是指在原图像中加上一个经过锐化的图层

其步骤为：

 1. 模糊原图像
 2. 原图像减去模糊图像(差值为模版)
 3. 把模版加到原图像上

 令$\overline{f}(x,y)$为模糊图像
 首先我们可以得到模版
$g_{mask}(x,y)=f(x,y)-\overline{f}(x,y)$
然后在原图像加上模版的一个权重k(k>0)：
$g(x,y)=f(x,y)+k*\overline{f}(x,y)$      
当k>1时，为高提升滤波



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

print(h)
print(sum(h))
print(sum(sum(h)))

h /= np.sum(h)  # 归一化处理

print('----')
print(h)
print(sum(h))
print(sum(sum(h)))

spanImg = np.zeros((H + 4, W + 4, 3), np.uint8)  # 5*5扩充后的图像

# print('----------')
# print(spanImg)


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

# Unsharp mask
UmPix = np.zeros((H, W), np.int32)
max = 0
min = 255
for i in range(H):
    for j in range(W):
        UmPix[i, j] = int(img[i, j, 0]) - blurImg[i, j, 0]
        if UmPix[i, j] > max:
            max = UmPix[i, j]
        if UmPix[i, j] < min:
            min = UmPix[i, j]

t = 0
if max > min:
    t = 255 / (max - min)

# 归一化的掩蔽图像
UmImg = np.zeros((H, W, 3), np.uint8)
for i in range(H):
    for j in range(W):
        UmImg[i, j, 0] = round((UmPix[i, j] - min) * t)
        UmImg[i, j, 1] = UmImg[i, j, 0]
        UmImg[i, j, 2] = UmImg[i, j, 0]

# 压缩至0-255之间的加入钝化掩蔽的图像
simgk = np.zeros((H, W, 3), np.uint8)
spixk = np.zeros((H, W), np.int32)

# 加入钝化掩蔽之后的图像
imgk = np.zeros((H, W, 3), np.uint8)
pixk = np.zeros((H, W), np.int32)

# 高提升滤波后图像
imgk2 = np.zeros((H, W, 3), np.uint8)
pixk2 = np.zeros((H, W), np.int32)

# 这里用到了截断来处理，并没有用scale来处理
for i in range(H):
    for j in range(W):
        pixk[i, j] = img[i, j, 0] + UmPix[i, j]
        spixk[i, j] = pixk[i, j]
        if pixk[i, j] > 255:
            pixk[i, j] = 255
        if pixk[i, j] < 0:
            pixk[i, j] = 0
        pixk2[i, j] = round(img[i, j, 0] + 4.5 * UmPix[i, j])
        if pixk2[i, j] > 255:
            pixk2[i, j] = 255
        if pixk2[i, j] < 0:
            pixk2[i, j] = 0
        imgk[i, j, 0] = pixk[i, j]
        imgk[i, j, 1] = imgk[i, j, 0]
        imgk[i, j, 2] = imgk[i, j, 0]
        imgk2[i, j, 0] = pixk2[i, j]
        imgk2[i, j, 1] = imgk2[i, j, 0]
        imgk2[i, j, 2] = imgk2[i, j, 0]

# 下面处理压缩0-255之间的图片
max = 0
min = 255
for i in range(H):
    for j in range(W):
        if spixk[i, j] > max:
            max = spixk[i, j]
        if spixk[i, j] < min:
            min = spixk[i, j]
sk = 0
if max > min:
    sk = 255 / (max - min)
for i in range(H):
    for j in range(W):
        simgk[i, j, 0] = round((spixk[i, j] - min) * sk)
        simgk[i, j, 1] = simgk[i, j, 0]
        simgk[i, j, 2] = simgk[i, j, 0]

# 原图
plt.subplot(3, 2, 1)
plt.axis('off')
plt.title('Original Image')
plt.imshow(img)

# 高斯平滑滤波模糊化的图像
plt.subplot(3, 2, 2)
plt.axis('off')
plt.title('Gaussian Smooth Filter Blurring')
plt.imshow(blurImg)

# 像素拉伸之后的差值图像
plt.subplot(3, 2, 3)
plt.axis('off')
plt.title('Scaled Unsharped mask(original - blur)')
plt.imshow(UmImg)

# 锐化并压缩的图像
plt.subplot(3, 2, 4)
plt.axis('off')
plt.title('Sharpened Image(scaled to 0-255)')
plt.imshow(simgk)

# 加上钝化mask之后的锐化图像
plt.subplot(3, 2, 5)
plt.axis('off')
plt.title('Sharpened Image(clipped to 0-255)')
plt.imshow(imgk)

# 设置系数为4.5之后的锐化图像
plt.subplot(3, 2, 6)
plt.axis('off')
plt.title('Highboostted Image(k=4.5,clipped)')
plt.imshow(imgk2)

plt.show()

```
