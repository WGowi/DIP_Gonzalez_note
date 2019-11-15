import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime

# 局部直方图增强 3.3.4节
# 使用3*3的邻域处理
img = cv2.imread('Fig0327.tif')  # 局部信息图片
H = img.shape[0]
W = img.shape[1]

# 一些参数的设置
E = 4.0
k0 = 0.4
k1 = 0.02
k2 = 0.4

ls = 3  # 邻域大小，但是后面没用到
MG = 0  # 全局平均灰度值
deltaG = 0  # 全局标准差
MS = 0  # 局部平均灰度值
deltaS = 0  # 局部标准差
# 统计全局平均灰度值
for i in range(H):
    for j in range(W):
        MG += img[i, j, 0]  # 灰度值的三个通道值都是一样的

MG = round(MG / (H * W))
# 计算全局标准差
for i in range(H):
    for j in range(W):
        deltaG += np.square(MG - img[i, j, 0])
deltaG = np.sqrt(deltaG / (H * W))

# 建立变换之后的图像
newImg = np.zeros((H, W, 3), np.uint8)
# 使用公示3.3-24进行变换
starttime = datetime.datetime.now()
# 先处理四个角的像素，只有4个像素
# 左上角像素
MS = (int(img[0, 0, 0]) + int(img[0, 1, 0]) + int(img[1, 0, 0]) + int(img[1, 1, 0])) / 4
deltaS = np.square(MS - img[0, 0, 0]) + np.square(MS - img[0, 1, 0]) \
         + np.square(MS - img[1, 0, 0]) + np.square(MS - img[1, 1, 0])
deltaS = np.sqrt(deltaS / 4)
if MS <= k0 * MG and k1 * deltaG <= deltaS <= k2 * deltaG:  # 这里的连写方式和数学方式一致
    newImg[0, 0] = E * img[0, 0]
else:
    newImg[0, 0] = img[0, 0]

# 左下角像素
MS = (int(img[H - 1, 0, 0]) + int(img[H - 1, 1, 0]) + int(img[H - 2, 0, 0]) + int(img[H - 2, 1, 0])) / 4
deltaS = np.square(MS - img[H - 1, 0, 0]) + np.square(MS - img[H - 1, 1, 0]) \
         + np.square(MS - img[H - 2, 0, 0]) + np.square(MS - img[H - 2, 1, 0])
deltaS = np.sqrt(deltaS / 4)
if MS <= k0 * MG and k1 * deltaG <= deltaS <= k2 * deltaG:  # 这里的连写方式和数学方式一致
    newImg[H - 1, 0] = E * img[H - 1, 0]
else:
    newImg[H - 1, 0] = img[H - 1, 0]

# 右上角像素
MS = (img[0, W - 1, 0] + img[0, W - 2, 0] + img[1, W - 1, 0] + img[1, W - 2, 0]) / 4
deltaS = np.square(MS - img[0, W - 1, 0]) + np.square(MS - img[0, W - 2, 0]) \
         + np.square(MS - img[1, W - 1, 0]) + np.square(MS - img[1, W - 2, 0])
deltaS = np.sqrt(deltaS / 4)
if MS <= k0 * MG and k1 * deltaG <= deltaS <= k2 * deltaG:  # 这里的连写方式和数学方式一致
    newImg[0, W - 1] = E * img[0, W - 1]
else:
    newImg[0, W - 1] = img[0, W - 1]

# 右下角像素
MS = (img[H - 1, W - 1, 0] + img[H - 1, W - 2, 0] + img[H - 2, W - 1, 0] + img[H - 2, W - 2, 0]) / 4
deltaS = np.square(MS - img[H - 1, W - 1, 0]) + np.square(MS - img[H - 1, W - 2, 0]) \
         + np.square(MS - img[H - 2, W - 1, 0]) + np.square(MS - img[H - 2, W - 2, 0])
deltaS = np.sqrt(deltaS / 4)
if MS <= k0 * MG and k1 * deltaG <= deltaS <= k2 * deltaG:  # 这里的连写方式和数学方式一致
    newImg[H - 1, W - 1] = E * img[H - 1, W - 1]
else:
    newImg[H - 1, W - 1] = img[H - 1, W - 1]

# 处理最上行数据，只有6个点
for i in range(1, W - 1):
    MS = (int(img[0, i, 0]) + img[0, i - 1, 0] + img[0, i + 1, 0] + img[1, i, 0] + img[1, i - 1, 0] + img[
        1, i + 1, 0]) / 6
    deltaS = np.square(MS - img[0, i, 0]) + np.square(MS - img[0, i - 1, 0]) + np.square(
        MS - img[0, i + 1, 0]) + np.square(MS - img[1, i, 0]) + np.square(MS - img[1, i - 1, 0]) + np.square(
        MS - img[1, i + 1, 0])
    deltaS = np.sqrt(deltaS / 6)
    if MS <= k0 * MG and k1 * deltaG <= deltaS <= k2 * deltaG:  # 这里的连写方式和数学方式一致
        newImg[0, i] = E * img[0, i]
    else:
        newImg[0, i] = img[0, i]

# 处理最下行数据，只有6个点
for i in range(1, W - 1):
    MS = (int(img[H - 1, i, 0]) + img[H - 1, i - 1, 0] + img[H - 1, i + 1, 0] + img[H - 2, i, 0] + img[
        H - 2, i - 1, 0] + img[H - 2, i + 1, 0]) / 6
    deltaS = np.square(MS - img[H - 1, i, 0]) + np.square(MS - img[H - 1, i - 1, 0]) + np.square(
        MS - img[H - 1, i + 1, 0]) + np.square(MS - img[H - 2, i, 0]) + np.square(
        MS - img[H - 2, i - 1, 0]) + np.square(MS - img[H - 2, i + 1, 0])
    deltaS = np.sqrt(deltaS / 6)
    if MS <= k0 * MG and k1 * deltaG <= deltaS <= k2 * deltaG:  # 这里的连写方式和数学方式一致
        newImg[H - 1, i] = E * img[H - 1, i]
    else:
        newImg[H - 1, i] = img[H - 1, i]

# 处理最左列数据，只有6个点
for i in range(1, H - 1):
    MS = (int(img[i, 0, 0]) + img[i, 1, 0] + img[i - 1, 0, 0] + img[i - 1, 1, 0] + img[i + 1, 0, 0] + img[
        i + 1, 1, 0]) / 6
    deltaS = np.square(MS - img[i, 0, 0]) + np.square(MS - img[i, 1, 0]) + np.square(MS - img[i - 1, 0, 0]) + np.square(
        MS - img[i - 1, 1, 0]) + np.square(MS - img[i + 1, 0, 0]) + np.square(MS - img[i + 1, 1, 0])
    deltaS = np.sqrt(deltaS / 6)
    if MS <= k0 * MG and k1 * deltaG <= deltaS <= k2 * deltaG:  # 这里的连写方式和数学方式一致
        newImg[i, 0] = E * img[i, 0]
    else:
        newImg[i, 0] = img[i, 0]

# 处理最右列数据，只有6个点
for i in range(1, H - 1):
    MS = (img[i, W - 1, 0] + img[i, W - 2, 0] + img[i - 1, W - 1, 0] + img[i - 1, W - 2, 0] + img[i + 1, W - 1, 0] +
          img[i + 1, W - 2, 0]) / 6
    deltaS = np.square(MS - img[i, W - 1, 0]) + np.square(MS - img[i, W - 2, 0]) + np.square(
        MS - img[i - 1, W - 1, 0]) + np.square(MS - img[i - 1, W - 2, 0]) + np.square(
        MS - img[i + 1, W - 1, 0]) + np.square(MS - img[i + 1, W - 2, 0])
    deltaS = np.sqrt(deltaS / 6)
    if MS <= k0 * MG and k1 * deltaG <= deltaS <= k2 * deltaG:  # 这里的连写方式和数学方式一致
        newImg[i, W - 1] = E * img[i, W - 1]
    else:
        newImg[i, W - 1] = img[i, W - 1]

# 处理剩下的内部矩形的点，都有9个点
for r in range(1, H - 2):
    for c in range(1, W - 2):
        # r==1是对第一列的点要重新计算
        # 剩下三行是比较灰度值相同，只要不同则重新计算，只要比较0通道，对于灰度图来说，三个通道值相同
        if r == 1 \
                or img[r, c - 2, 0] != img[r, c + 2, 0] \
                or img[r - 1, c - 2, 0] != img[r - 1, c + 2, 0] \
                or img[r + 1, c - 2, 0] != img[r + 1, c + 2, 0]:
            # 如果左右三个像素相同，则局部平均灰度值不用求
            MS = (int(img[r - 1, c - 1, 0]) + img[r - 1, c, 0] + img[r - 1, c + 1, 0] + img[r, c - 1, 0] + img[
                r, c, 0] + img[r, c + 1, 0] + img[r + 1, c - 1, 0] + img[r + 1, c, 0] + img[r + 1, c + 1, 0]) / 9
            deltaS = np.square(MS - img[r - 1, c - 1, 0]) + np.square(MS - img[r - 1, c, 0]) + np.square(
                MS - img[r - 1, c + 1, 0]) + np.square(MS - img[r, c - 1, 0]) + np.square(
                MS - img[r, c, 0]) + np.square(MS - img[r, c + 1, 0]) + np.square(
                MS - img[r + 1, c - 1, 0]) + np.square(MS - img[r + 1, c, 0]) + np.square(MS - img[r + 1, c + 1, 0])
            deltaS = np.sqrt(deltaS / 9)
        if MS <= k0 * MG and k1 * deltaG <= deltaS <= k2 * deltaG:  # 这里的连写方式和数学方式一致
            newImg[r, c] = E * img[r, c]
        else:
            newImg[r, c] = img[r, c]

endtime = datetime.datetime.now()
print(endtime - starttime)

# 原图
plt.subplot(1, 2, 1)
plt.axis('off')
plt.title('Original image')
plt.imshow(img)

# 直方图均衡后的图
plt.subplot(1, 2, 2)
plt.axis('off')
plt.title('Local Histogram Enhanced image')
plt.imshow(newImg)

plt.show()
