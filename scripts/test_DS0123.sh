#!/bin/sh

# 執行以下命令
cd ..
python val.py -ti /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/images/ -tm /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/masks/ -m /home/yaocong/Experimental/speed_smoke_segmentation/trained_models/best.pth
python val.py -ti /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS02/images/ -tm /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS02/masks/ -m /home/yaocong/Experimental/speed_smoke_segmentation/trained_models/best.pth
python val.py -ti /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS03/images/ -tm /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS03/masks/ -m /home/yaocong/Experimental/speed_smoke_segmentation/trained_models/best.pth
python val.py -ti /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/Real/images/ -tm /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/Real/masks/ -m /home/yaocong/Experimental/speed_smoke_segmentation/trained_models/best.pth

python val.py -ti /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/images/ -tm /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/masks/ -m /home/yaocong/Experimental/speed_smoke_segmentation/trained_models/mynet_70k_data/CGnet_erfnet3_1_1_3_test_dilated/best.pth
python val.py -ti /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS02/images/ -tm /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS02/masks/ -m /home/yaocong/Experimental/speed_smoke_segmentation/trained_models/mynet_70k_data/CGnet_erfnet3_1_1_3_test_dilated/best.pth
python val.py -ti /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS03/images/ -tm /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS03/masks/ -m /home/yaocong/Experimental/speed_smoke_segmentation/trained_models/mynet_70k_data/CGnet_erfnet3_1_1_3_test_dilated/best.pth
python val.py -ti /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/Real/images/ -tm /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/Real/masks/ -m /home/yaocong/Experimental/speed_smoke_segmentation/trained_models/mynet_70k_data/CGnet_erfnet3_1_1_3_test_dilated/best.pth


