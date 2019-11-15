import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Fig0335.tif')  # 测试图片
H = img.shape[0]
W = img.shape[1]

img3 = np.zeros((H, W, 3), np.uint8)  # 3*3邻域平滑后的图像
imgmid = np.zeros((H, W, 3), np.uint8)  # 3*3邻域内取中值的图像

tmpImg = np.zeros((H + 2, W + 2, 3), np.uint8)  # 扩充之后的图像
for i in range(H):
    for j in range(W):
        tmpImg[i + 1, j + 1] = img[i, j]

starttime = datetime.datetime.now()
for i in range(H):
    for j in range(W):
        S = []
        for x in range(3):
            for y in range(3):
                # S[x * 3 + y] = tmpImg[i + x, j + y, 0]
                S.append(tmpImg[i + x, j + y, 0])
        img3[i, j, 0] = sum(S) // 9
        img3[i, j, 1] = img3[i, j, 0]
        img3[i, j, 2] = img3[i, j, 0]
        # 冒泡排序，只要排到中间一个值，即4，因此x范围是8->3
        # for x in range(8, 3, -1):
        #     for y in range(x):
        #         if S[y + 1] > S[y]:
        #             temp = S[y]
        #             S[y] = S[y + 1]
        #             S[y + 1] = temp
        # 自带的排序，timesort，经过测试，这个比冒泡排序要快60%
        S.sort()
        print(S)
        imgmid[i, j, 0] = S[4]
        imgmid[i, j, 1] = imgmid[i, j, 0]
        imgmid[i, j, 2] = imgmid[i, j, 0]

endtime = datetime.datetime.now()
print(endtime - starttime)

# 原图
plt.subplot(1, 3, 1)
plt.axis('off')
plt.title('Original image')
plt.imshow(img)

# 3*3邻域
plt.subplot(1, 3, 2)
plt.axis('off')
plt.title('3*3 smoothing')
plt.imshow(img3)

# 邻域中值替换之后
plt.subplot(1, 3, 3)
plt.axis('off')
plt.title('Middle Value Replaced Image')
plt.imshow(imgmid)

plt.show()
