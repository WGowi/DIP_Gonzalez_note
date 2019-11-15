import cv2
import numpy as np
import matplotlib.pyplot as plt

img_a = cv2.imread('Fig0316a.tif')
img_b = cv2.imread('Fig0316b.tif')
img_c = cv2.imread('Fig0316c.tif')
img_d = cv2.imread('Fig0316d.tif')

H = img_a.shape[0]
W = img_a.shape[1]

His = np.zeros((4, 256))  # store 4 images
pr = np.zeros((4, 256))  # 概率密度函数
S = np.zeros((4, 256))  # transformed images
newHis = np.zeros((4, 256))  # 新图的直方图

# 统计原始图片中的灰度分布
for row in range(H):
    for col in range(W):
        His[0, img_a[row, col]] += 1
        His[1, img_b[row, col]] += 1
        His[2, img_c[row, col]] += 1
        His[3, img_d[row, col]] += 1

# 计算原始概率密度函数pr
for i in range(4):
    for j in range(256):
        pr[i, j] = His[i, j] / (H * W)

# 计算转换后s的值
for i in range(4):
    for j in range(256):
        for m in range(j + 1):
            S[i, j] += pr[i, m]
        S[i, j] = (int)(S[i, j] * 255)

# for i in range(256):
#     print(pr[0, i], S[0, i])

# 创建新的四张图片
# 由于原图读进来的时候是三通道的，所以这里也要建立三通道的图片，因此第三个参数是3
newImg_a = np.zeros((H, W, 3), np.uint8)
newImg_b = np.zeros((H, W, 3), np.uint8)
newImg_c = np.zeros((H, W, 3), np.uint8)
newImg_d = np.zeros((H, W, 3), np.uint8)

for row in range(H):
    for col in range(W):
        newImg_a[row, col] = S[0, img_a[row, col]]
        newImg_b[row, col] = S[1, img_b[row, col]]
        newImg_c[row, col] = S[2, img_c[row, col]]
        newImg_d[row, col] = S[3, img_d[row, col]]

# 绘制新的图片的直方图
for row in range(H):
    for col in range(W):
        newHis[0, newImg_a[row, col]] += 1
        newHis[1, newImg_b[row, col]] += 1
        newHis[2, newImg_c[row, col]] += 1
        newHis[3, newImg_d[row, col]] += 1

# 原始图片
plt.subplot(4, 4, 1)
plt.axis('off')
plt.imshow(img_a)
plt.subplot(4, 4, 5)
plt.axis('off')
plt.imshow(img_b)
plt.subplot(4, 4, 9)
plt.axis('off')
plt.imshow(img_c)
plt.subplot(4, 4, 13)
plt.axis('off')
plt.imshow(img_d)

# 新图片
plt.subplot(4, 4, 3)
plt.axis('off')
plt.imshow(newImg_a)
plt.subplot(4, 4, 7)
plt.axis('off')
plt.imshow(newImg_b)
plt.subplot(4, 4, 11)
plt.axis('off')
plt.imshow(newImg_c)
plt.subplot(4, 4, 15)
plt.axis('off')
plt.imshow(newImg_d)

x = np.arange(0, 256, 1)  # [0,255]
# 老的直方图
plt.subplot(4, 4, 2)
plt.bar(x, His[0])
plt.subplot(4, 4, 6)
plt.bar(x, His[1])
plt.subplot(4, 4, 10)
plt.bar(x, His[2])
plt.subplot(4, 4, 14)
plt.bar(x, His[3])

# 新图片的直方图
plt.subplot(4, 4, 4)
plt.bar(x, newHis[0])
plt.subplot(4, 4, 8)
plt.bar(x, newHis[1])
plt.subplot(4, 4, 12)
plt.bar(x, newHis[2])
plt.subplot(4, 4, 16)
plt.bar(x, newHis[3])

plt.show()
