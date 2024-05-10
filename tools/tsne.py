import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# from tsnecuda import TSNE
import argparse
from glob import glob
from PIL import Image

def load_images(image_paths):
    images = []
    for path in image_paths:
        image = Image.open(path)
        image = image.resize((64, 64))  # 縮小影像以節省記憶體
        image = np.array(image).flatten()  # 轉換為一維陣列
        images.append(image)
    return np.array(images)

def main():
    data1_paths = glob(f"{args['dataset1']}/*.jpg")
    data2_paths = glob(f"{args['dataset2']}/*.png")
    data1 = load_images(data1_paths)
    data2 = load_images(data2_paths)
    combined_data = np.concatenate((data1, data2), axis=0)

    # 設定輸出的維度數量
    n_components = 2

    # 創建 t-SNE 實例
    tsne = TSNE(n_components=n_components)

    # 擬合並轉換數據
    reduced_data = tsne.fit_transform(combined_data)

    # 將數據分離回單獨的資料集
    reduced_data1 = reduced_data[:len(data1)]
    reduced_data2 = reduced_data[len(data1):]

    # 以藍色繪製第一個資料集
    plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1], color='blue', label='Dataset1')

    # 以紅色繪製第二個資料集
    plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1], color='red', label='Dataset2')

    # 添加標籤、標題和圖例
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of Two Datasets')
    plt.legend()

    # 顯示圖表
    plt.show()
    print("Done")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument(
    #     "-td", "--test_directory", required=True, help="path to test images directory"
    # )
    ap.add_argument(
        "-data1",
        "--dataset1",
        default="/home/yaocong/Experimental/Dataset/SYN70K_dataset/training_data/blendall/",
        help="path to dataset1",
    )
    ap.add_argument(
        "-data2",
        "--dataset2",
        default="/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/img/",
        help="path to dataset2",
    )
    ap.add_argument("-bs", "--batch_size", type=int, default=1, help="set batch_size")
    ap.add_argument("-nw", "--num_workers", type=int, default=1, help="set num_workers")

    args = vars(ap.parse_args())
    main()