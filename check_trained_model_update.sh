#check inside /speed_smoke_segmentation/trained_models/mynet_70k_data/CGnet_erfnet3_1_1_3_test_3113_dilated files are updated or not
# if updated, copy the folder to yaocong@140.125.35.199:/home/yaocong/Experimental/speed_smoke_segmentation/trained_models/mynet_70k_data/
# if not, do nothing
#Usage: bash check_trained_model_update.sh
#!/bin/sh

function check_update(){
    local folder=$1
    local remote_folder=$2
    local remote_ip=$3
    local remote_user=$4
    local remote_folder_path=$remote_user@$remote_ip:$remote_folder
    local local_folder_path=$folder
    local local_folder_name=${local_folder_path##*/}
    local remote_folder_name=${remote_folder_path##*/}
    local local_folder_time=$(stat -c %Y $local_folder_path)
    local remote_folder_time=$(ssh $remote_user@$remote_ip stat -c %Y $remote_folder)
    if [ $local_folder_time -gt $remote_folder_time ]; then
        echo "local folder $local_folder_name is newer than remote folder $remote_folder_name"
        echo "copying $local_folder_name to $remote_folder_name"
        scp -r $local_folder_path $remote_folder_path
    else
        echo "local folder $local_folder_name is older than remote folder $remote_folder_name"
        echo "do nothing"
    fi
}

check_update "trained_models/mynet_70k_data/CGnet_erfnet3_1_1_3_test_3113_dilated" "/home/yaocong/Experimental/speed_smoke_segmentation/trained_models/mynet_70k_data/" "140.125.35.199" "yaocong"
