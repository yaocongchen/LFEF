#!/bin/sh

function check_update(){
    local folder=$1
    local remote_folder=$2
    local remote_ip=$3
    local remote_user=$4
    local remote_port=$5
    local remote_folder_path=$remote_user@$remote_ip:$remote_folder
    local local_folder_path=$folder
    local local_folder_name=${local_folder_path##*/}
    local remote_folder_name=${remote_folder_path##*/}
    local local_file_path=$local_folder_path/log.txt
    local remote_file_path=$remote_folder/log.txt

    # 先從遠程服務器下載 log.txt 文件到臨時文件
    rsync -avz -e "ssh -p $remote_port" $remote_folder_path/log.txt /tmp/remote_log.txt

    # 檢查本地文件是否存在
    if [ ! -f $local_file_path ]; then
        echo "local file $local_file_path does not exist"
        echo "copying $remote_folder_name to $local_folder_name"
        rsync -avz -e "ssh -p $remote_port" $remote_folder_path $local_folder_path
    else
        # 使用 diff 命令比較本地和遠程的 log.txt 文件
        diff $local_file_path /tmp/remote_log.txt > /dev/null
        if [ $? -ne 0 ]; then
            echo "local file $local_file_path is different from remote file"
            echo "copying $remote_folder_name to $local_folder_name"
            rsync -avz -e "ssh -p $remote_port" $remote_folder_path $local_folder_path
        else
            echo "local file $local_file_path is the same as remote file"
            echo "do nothing"
        fi
    fi
}

check_update "/home/yaocong/Experimental/speed_smoke_segmentation/trained_models/mynet_70k_data/CGnet_erfnet3_1_1_3_test_dilated/" "~/speed_smoke_segmentation/trained_models/mynet_70k_data/CGnet_erfnet3_1_1_3_test_dilated/" "140.125.35.191" "m11013017" "22"