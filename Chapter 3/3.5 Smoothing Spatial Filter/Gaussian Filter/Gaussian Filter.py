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
