from PIL import Image
img1 = Image.open('/home/yaocong/Experimental/speed_smoke_segmentation/ttt/img/1_3.jpg')
img1 = img1.convert('RGBA')
L,H = img1.size
for h in range(H):
    for l in range(L):
        color = img1.getpixel((h,l))
        color = color[:-1] + (1000, )
        img1.putpixel((h,l),color)

img1.save("trans.png")