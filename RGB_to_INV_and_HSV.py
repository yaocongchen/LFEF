#RGB to INV and HSV conversion
import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = "986"

# Load the image
image = cv2.imread(f"/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/images/{filename}.png")

# Convert the image from BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Invert the image
image_inv = cv2.bitwise_not(image)

# Convert the image from RGB to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

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
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(image)
ax[0].axis('off')
ax[0].set_title('Original image')
ax[1].imshow(image_inv)
ax[1].axis('off')
ax[1].set_title('Inverted image')
ax[2].imshow(hsv_image)
ax[2].axis('off')
ax[2].set_title('HSV image')
plt.savefig(f"{filename}_all.png")
plt.show()
