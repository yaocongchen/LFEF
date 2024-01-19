import torch
import torchvision.transforms.functional as F
from PIL import Image

# 讀取圖像
image_path = "/home/yaocong/Experimental/speed_smoke_segmentation/testing_multiple_result/test_RGB_image/test_RGB_image_3.jpg"
image = Image.open(image_path)

# 將圖像轉換為 PyTorch 張量
image_tensor = F.to_tensor(image)

# 將張量轉換回 torch.uint8 類型並將像素值範圍轉換回 [0, 255]
image_tensor = (image_tensor * 255).byte()

# 進行亮度均衡
equalized_image_tensor = F.equalize(image_tensor)

# 將均衡後的張量轉換為 PIL 圖像
equalized_image = F.to_pil_image(equalized_image_tensor)

# 顯示原始圖像和均衡後的圖像
image.show()
equalized_image.show()