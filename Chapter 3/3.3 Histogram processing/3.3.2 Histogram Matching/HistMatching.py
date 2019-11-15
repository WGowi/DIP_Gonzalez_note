import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Fig0323.tif')  # 月球表面图片
H = img.shape[0]
W = img.shape[1]

x = np.arange(256)  # 作为index使用
hr = np.zeros(256)  # 原始图片的灰度统计
pr = np.zeros(256)  # 原始图片的概率密度
rtos = np.zeros(256)  # r-s对应关系
hs = np.zeros(256)  # 直方图均衡之后的灰度值
thz = np.zeros(256)  # 目标直方图灰度值
tpz = np.zeros(256)  # 目标直方图的概率密度
rtoz = np.zeros(256)  # r-z的对应关系
Gz = np.zeros(256)  # G(z)的值
Zg = np.zeros(256)  # G(z)的反函数
hz = np.zeros(256)  # 实际得到的匹配图像的直方图

# 统计原始图片的直方图
for row in range(H):
    for col in range(W):
        hr[img[row, col]] += 1

# 直方图均衡
# 先计算概率密度
for i in range(256):
    pr[i] = hr[i] / (H * W)

for i in range(256):  # i=[0,255]
    temp = 0
    for j in range(i + 1):  # j=[0,i]
        temp += pr[j]
    rtos[i] = round(temp * 255)  # 四舍五入取整

hisImg = np.zeros((H, W, 3), np.uint8)  # 建立直方图均衡变换之后的图片
for row in range(H):
    for col in range(W):
        hisImg[row, col] = rtos[img[row, col]]

# 统计均衡后的图片的直方图
for row in range(H):
    for col in range(W):
        hs[hisImg[row, col]] += 1

# 使用类似于图3.25a的概率分布图（估计值）
# 对应的几个拐点
#   i    thz
#   0     0
#   4     400000
#   16    45600
#   185   0
#   205   34200
#   255   0
for i in range(256):
    if i < 5:
        thz[i] = i * 100000
    elif i < 17:
        thz[i] = 518133 - 29533 * i
    elif i < 186:
        thz[i] = 49950 - i * 270
    elif i < 206:
        thz[i] = 1710 * (i - 185)
    else:
        thz[i] = 684 * (255 - i)

# 统计目标直方图的概率分布
ztotal = np.sum(thz)
for i in range(256):
    tpz[i] = thz[i] / ztotal

# 计算G(z)的值，同样采用直方图均衡化方法
for i in range(256):  # i=[0,255]
    temp = 0
    for j in range(i + 1):  # j=[0,i]
        temp += tpz[j]
    Gz[i] = round(temp * 255)  # 四舍五入取整

# for i in range(256):
#    print(i, Gz[i])

# 因为给出的概率分布没有水平线，所以Gz是单调递增的（除了一个点x=185）
# 这里考虑没有单调增的情况，实现r->s->Gz->z的映射
# 求G(z)的反函数，这样求出来的反函数可能会有某个Zg[i]==0
for i in range(256):
    Zg[int(Gz[i])] = i

for i in range(1, 256):
    if Zg[i] == 0:
        for j in range(1, 256):  # 如果Zg[i]为0，则从左右开始搜索最接近i的非零值，设置为Zg[i]的值
            if i - j >= 0 and Zg[i - j] != 0:
                Zg[i] = Zg[i - j]
            elif i + j < 256 and Zg[i + j] != 0:
                Zg[i] = Zg[i + j]
            break

SpecifiedImg = np.zeros((H, W, 3), dtype=np.uint8)  # 建立规定直方图变换之后的图片

# 求出r->s->Gz->z的映射
for i in range(256):
    rtoz[i] = Zg[int(rtos[i])]

# 利用新的映射绘制新图
for row in range(H):
    for col in range(W):
        SpecifiedImg[row, col] = rtoz[img[row, col]]

# 统计真正匹配图像的直方图
for row in range(H):
    for col in range(W):
        hz[SpecifiedImg[row, col]] += 1

# print(hz)

# 原图
plt.subplot(2, 4, 1)
plt.axis('off')
plt.title('Original image')
plt.imshow(img)

# 原图的直方图
plt.subplot(2, 4, 2)
plt.bar(x, hr)
plt.title('Histogram of original image')

# 均衡后的图片
plt.subplot(2, 4, 3)
plt.axis('off')
plt.title('After equalization')
plt.imshow(hisImg)

# r->s的变化函数
# plt.subplot(3, 2, 4)
# plt.title('s=T(r)')
# plt.scatter(x, rtos)

# 均衡化后的直方图
plt.subplot(2, 4, 4)
plt.title('Histogram after equalization')
plt.bar(x, hs)

# 目标直方图
plt.subplot(2, 4, 5)
plt.title('Specified histogram')
plt.plot(x, thz)

# Gz和Zg
plt.subplot(2, 4, 6)
plt.title('G(z) and its reverse')
plt.plot(x, Gz, Zg)

# 新的图片
plt.subplot(2, 4, 7)
plt.axis('off')
plt.title('Specified Image')
plt.imshow(SpecifiedImg)

# 新的直方图
plt.subplot(2, 4, 8)
plt.title('Real specified histogram')
plt.bar(x, hz)

plt.show()
