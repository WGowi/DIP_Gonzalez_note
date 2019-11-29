# 数字图像处理——中值滤波器【像素级别处理】（python）


# 简介
中值滤波（统计排序滤波器）是一种非线性数字滤波器技术，经常用于去除图像或者其它信号中的噪声。这个设计思想就是检查输入信号中的采样并判断它是否代表了信号，使用奇数个采样组成的观察窗实现这项功能。观察窗口中的数值进行排序，位于观察窗中间的中值作为输出。然后，丢弃最早的值，取得新的采样，重复上面的计算过程。

其中中值滤波对椒盐噪声处理非常好

# 代码实现


输入：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191114230709453.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMzA5Mjg2,size_16,color_FFFFFF,t_70)

```python
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

```



输出：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191114230557129.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMzA5Mjg2,size_16,color_FFFFFF,t_70)


```python
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Fig0335.tif')  # 测试图片
H = img.shape[0]
W = img.shape[1]

img3 = np.zeros((H, W, 3), np.uint8)  # 3*3邻域平滑后的图像
imgmid = np.zeros((H, W, 3), np.uint8)  # 3*3邻域内取中值的图像
imgmid5 = np.zeros((H, W, 3), np.uint8)

tmpImg = np.zeros((H + 2, W + 2, 3), np.uint8)  # 扩充之后的图像
tmpImg5 = np.zeros((H + 4, W + 4, 3), np.uint8)

for i in range(H):
    for j in range(W):
        tmpImg[i + 1, j + 1] = img[i, j]
        tmpImg5[i + 2, j + 2] = img[i, j]

starttime = datetime.datetime.now()
for i in range(H):
    for j in range(W):
        S = []
        T = []
        for x in range(3):
            for y in range(3):
                # S[x * 3 + y] = tmpImg[i + x, j + y, 0]
                S.append(tmpImg[i + x, j + y, 0])
        # img3[i, j, 0] = sum(S) // 9
        # img3[i, j, 1] = img3[i, j, 0]
        # img3[i, j, 2] = img3[i, j, 0]
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
        for x in range(5):
            for y in range(5):
                T.append(tmpImg5[i + x, j + y, 0])
        T.sort()
        imgmid5[i, j, 0] = T[12]
        imgmid5[i, j, 1] = T[12]
        imgmid5[i, j, 2] = T[12]

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
plt.title('5*5 smoothing')
plt.imshow(imgmid5)

# 邻域中值替换之后
plt.subplot(1, 3, 3)
plt.axis('off')
plt.title('Middle Value Replaced Image')
plt.imshow(imgmid)

plt.show()

```


![在这里插入图片描述](https://img-blog.csdnimg.cn/20191116225736458.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMzA5Mjg2,size_16,color_FFFFFF,t_70)
