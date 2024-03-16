#RGB to INV and HSV conversion
import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = "451"

# Load the image
image = cv2.imread(f"/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/images/{filename}.png")

# Convert the image from BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 調整對比度
# alpha: 控制對比度 (1.0~3.0)
# beta: 控制亮度 (0~100)
image_conver = cv2.convertScaleAbs(image, alpha=2.0, beta=0)

# gray the image
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

image_add_gray = cv2.add(image_gray, image_gray)

# Invert the image
image_inv = cv2.bitwise_not(image)

# Convert the image from RGB to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

image_mul_hsv = cv2.multiply(image, hsv_image)
image_mul_hsv_gray = cv2.cvtColor(image_mul_hsv, cv2.COLOR_RGB2GRAY)

# 低通濾波器
kernel = np.ones((5, 5), np.float32) / 25
image_mul_hsv_gray = cv2.filter2D(image_mul_hsv_gray, -1, kernel)

# 閥值127以上的變為255 其餘變為0
ret, image_mul_hsv_gray = cv2.threshold(image_mul_hsv_gray, 127, 255, cv2.THRESH_BINARY)


# # Display the original image
# plt.imshow(image)
# plt.axis('off')
# plt.title('Original image')
# #save the image
# plt.savefig(f"{filename}.png")

# # Display the inverted image
# plt.imshow(image_inv)
# plt.axis('off')
# plt.title('Inverted image')
# plt.savefig(f"{filename}_inverted.png")

# # Display the HSV image
# plt.imshow(hsv_image)
# plt.axis('off')
# plt.title('HSV image')
# plt.savefig(f"{filename}_HSV.png")

#合為一張圖
fig, ax = plt.subplots(1, 8, figsize=(15, 5))
ax[0].imshow(image)
ax[0].axis('off')
ax[0].set_title('Original image')
ax[1].imshow(image_inv)
ax[1].axis('off')
ax[1].set_title('Inverted image')
ax[2].imshow(hsv_image)
ax[2].axis('off')
ax[2].set_title('HSV image')
ax[3].imshow(image_mul_hsv)
ax[3].axis('off')
ax[3].set_title('Subtracted image')
ax[4].imshow(image_mul_hsv_gray, cmap='gray')

ax[4].axis('off')
ax[4].set_title('Subtracted image (gray)')

ax[5].imshow(image_gray, cmap='gray')
ax[5].axis('off')
ax[5].set_title('Gray image')

ax[6].imshow(image_add_gray, cmap='gray')
ax[6].axis('off')
ax[6].set_title('Subtracted image (gray)')

ax[7].imshow(image_conver)
ax[7].axis('off')
ax[7].set_title('Contrast image')


plt.savefig(f"{filename}_all.png")
plt.show()
