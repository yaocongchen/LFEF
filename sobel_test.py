import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torchvision

# 讀取圖像
image_path = "/home/yaocong/Experimental/speed_smoke_segmentation/testing_multiple_result/test_output/test_output_3.jpg"
image = Image.open(image_path)

# 將圖像轉換為 PyTorch 張量
image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0)

# 將張量轉換為灰度
gray_image_tensor = 0.2989 * image_tensor[:, 0, :, :] + 0.5870 * image_tensor[:, 1, :, :] + 0.1140 * image_tensor[:, 2, :, :]
gray_image_tensor = gray_image_tensor.unsqueeze(0)

# 定義 Sobel 運算子
sobel_x = torch.tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)

#Sobel 運算子得到垂直方向的梯度
sobel_y = torch.tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)


# 計算圖像在水平和垂直方向上的梯度
gradient_x = F.conv2d(gray_image_tensor, sobel_x)
gradient_y = F.conv2d(gray_image_tensor, sobel_y)

# 計算梯度的大小
gradient_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)


torchvision.utils.save_image (gradient_x, '/home/yaocong/Experimental/speed_smoke_segmentation/testing_multiple_result/test_output/sobel_test_x.jpg')
torchvision.utils.save_image (gradient_y, '/home/yaocong/Experimental/speed_smoke_segmentation/testing_multiple_result/test_output/sobel_test_y.jpg')


torchvision.utils.save_image (gradient_magnitude, '/home/yaocong/Experimental/speed_smoke_segmentation/testing_multiple_result/test_output/sobel_test.jpg')
# 顯示原始圖像和梯度相關的信息
plt.figure(figsize=(10, 5))

plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Gradient X")
plt.imshow(gradient_x.squeeze().numpy(), cmap='gray')
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Gradient Y")
plt.imshow(gradient_y.squeeze().numpy(), cmap='gray')
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Gradient XY")
plt.imshow((gradient_magnitude).squeeze().numpy(), cmap='gray')
plt.axis("off")

plt.show()
# #save
# plt.savefig('/home/yaocong/Experimental/speed_smoke_segmentation/testing_multiple_result/test_output/sobel_test.jpg')
