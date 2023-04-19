from PIL import Image

def gray_to_binary(image):

    gray = image.convert('L')
    threshold = 200

    table = []
    for pixel_g in range(256):
        if pixel_g < threshold:
            table.append(0)
        else:
            table.append(1)

    binary = gray.point(table, '1')   #PIL Image.point()
    return binary

def overlap(image1,image2,read_method):
    W,H = image2.size
    black_background = (0, 0, 0, 255)
    #white_mask = (255, 255, 255, 255)

    for h in range(H):
        for w in range(W):
            dot = (w,h)
            color_1 = image2.getpixel(dot)
            if color_1 == black_background:
                color_1 = color_1[:-1] + (0,)   # Commas are used to create a (tuple) 逗號是用於創造一個(tuple)
                image2.putpixel(dot,color_1)
            else:
                if read_method == "PIL_RGBA":
                    color_1 = (255,0,0,) + color_1[3:]  #逗號是用於創造一個(tuple) #RGBA
                elif read_method == "OpenCV_BGRA":
                    color_1 = (0,0,255,) + color_1[3:]  #逗號是用於創造一個(tuple) #BGRA

                image2.putpixel(dot,color_1)
    #img2.show()
    # Overlay image 疊合影像
    blendImg = Image.blend(image1, image2 , alpha = 0.2)
    return blendImg