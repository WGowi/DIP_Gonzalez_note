import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Fig0333.tif')  # 测试图片
H = img.shape[0]
W = img.shape[1]

img3 = np.zeros((H, W, 3), np.uint8)  # 3*3邻域平滑后的图像
img5 = np.zeros((H, W, 3), np.uint8)  # 5*5邻域平滑后的图像
img9 = np.zeros((H, W, 3), np.uint8)  # 9*9邻域平滑后的图像
img15 = np.zeros((H, W, 3), np.uint8)  # 15*15邻域平滑后的图像
img35 = np.zeros((H, W, 3), np.uint8)  # 35*35邻域平滑后的图像

imgs = [img3, img5, img9, img15, img35]
size = [1, 2, 4, 7, 17]  # 邻域的一半

for id in range(5):  # 五个图片
    for i in range(H):
        for j in range(W):
            sum = 0
            count = 0
            for m in range(-1 * size[id], size[id] + 1):
                for n in range(-1 * size[id], size[id] + 1):
                    if 0 <= i + m < H and 0 <= j + n < W:  # 这个if循环避免了补0的操作
                        count += 1
                        sum += img[i + m, j + n, 0]
            imgs[id][i, j, 0] = sum // count
            imgs[id][i, j, 1] = imgs[id][i, j, 0]
            imgs[id][i, j, 2] = imgs[id][i, j, 0]

# 原图
plt.subplot(2, 3, 1)
plt.axis('off')
plt.title('Original image')
plt.imshow(img)

# 3*3邻域
plt.subplot(2, 3, 2)
plt.axis('off')
plt.title('3*3 smoothing')
plt.imshow(img3)

# 5*5邻域
plt.subplot(2, 3, 3)
plt.axis('off')
plt.title('5*5 smoothing')
plt.imshow(img5)

# 9*9邻域
plt.subplot(2, 3, 4)
plt.axis('off')
plt.title('9*9 smoothing')
plt.imshow(img9)

# 15*15邻域
plt.subplot(2, 3, 5)
plt.axis('off')
plt.title('15*15 smoothing')
plt.imshow(img15)

# 35*35邻域
plt.subplot(2, 3, 6)
plt.axis('off')
plt.title('35*35 smoothing')
plt.imshow(img35)

plt.show()
