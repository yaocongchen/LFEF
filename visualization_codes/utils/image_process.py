from PIL import Image
import numpy as np

def process_pil_to_np(image, size=(256, 256), gray=False):
    if gray:
        image = image.convert("L")
    image = image.resize(size)
    image = np.array(image, dtype=np.int32)
    return image

def gray_to_binary(image):
    gray = image.convert("L")
    threshold = 128

    table = []
    for pixel_g in range(256):
        if pixel_g < threshold:
            table.append(0)
        else:
            table.append(1)

    binary = gray.point(table, "1")  # PIL Image.point()
    return binary


def overlap_v1(image1, image2, read_method):
    W, H = image2.size
    black_background = (0, 0, 0, 255)
    # white_mask = (255, 255, 255, 255)

    for h in range(H):
        for w in range(W):
            dot = (w, h)
            color_1 = image2.getpixel(dot)
            if color_1 == black_background:
                color_1 = color_1[:-1] + (
                    0,
                )  # Commas are used to create a (tuple) 逗號是用於創造一個(tuple)
                image2.putpixel(dot, color_1)
            else:
                if read_method == "PIL_RGBA":
                    color_1 = (
                        255,
                        0,
                        0,
                    ) + color_1[
                        3:
                    ]  # 逗號是用於創造一個(tuple) #RGBA
                elif read_method == "OpenCV_BGRA":
                    color_1 = (
                        0,
                        0,
                        255,
                    ) + color_1[
                        3:
                    ]  # 逗號是用於創造一個(tuple) #BGRA

                image2.putpixel(dot, color_1)

    # img2.show()
    # Overlay image 疊合影像
    blendImg = Image.blend(image1, image2, alpha=0.2)
    return blendImg


def overlap_v2(image1, image2, read_method):
    image = image1.copy()
    W, H = image2.size
    black_background = (0, 0, 0, 255)
    # white_mask = (255, 255, 255, 255)

    for h in range(H):
        for w in range(W):
            dot = (w, h)
            color_1 = image1.getpixel(dot)
            color_2 = image2.getpixel(dot)
            if color_2 == black_background:
                continue
            else:
                if read_method == "PIL_RGBA":
                    color_1 = ((color_1[0] + 255), (color_1[1] + 0), (color_1[2] + 0))
                elif read_method == "OpenCV_BGRA":
                    color_1 = ((color_1[0] + 0), (color_1[1] + 0), (color_1[2] + 255))
                image.putpixel(dot, color_1)

    return image

def overlap_v3(image1: np.ndarray, mask: np.ndarray, read_method):
    image = image1.copy()
    if read_method == "PIL_RGBA":
        color_fn = lambda x: np.add(x[mask_idx], np.array([255, 0, 0]))
    elif read_method == "OpenCV_BGRA":
        color_fn = lambda x: np.add(x[mask_idx], np.array([0, 0, 255]))

    mask_idx = mask == 255
    image[mask_idx] = color_fn(image) 
    image = np.clip(image, 0, 255).astype(np.uint8) 
    return image

if __name__ == "__main__":
    image1 = np.array(Image.open("/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/img/1_1.jpg"), dtype=np.int32)
    image2 = np.array(Image.open("/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/gt/1_1.png"), dtype=np.int32)
    blendImg = overlap_v3(image1, image2, "PIL_RGBA")
    Image.fromarray(blendImg).show()
    Image.fromarray(blendImg).save("blendImg_1.png")
    print("overlap_v3 done")