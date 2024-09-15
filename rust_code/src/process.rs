use crate::utils::{image_processing, model};
use ort::Tensor;
use std::fs;
use image::imageops::FilterType;
use std::path::Path;
use opencv::imgproc;
use opencv::{
    core, highgui,
    prelude::*,
    videoio::{self, VideoCapture, VideoWriter, CAP_ANY},
    Result,
};



pub fn single_image() -> Result<(), Box<dyn std::error::Error>> {

    let input_img =
        image::open("/home/yaocong/Dataset/SYN70K_dataset/testing_data/DS01/images/2.png").unwrap();
    let output_folder = "./results/processed_single_images";
    fs::create_dir_all(output_folder)?;

    // input_img.save("input.png")?;

    let model = model::create_model_session()?;

    let input_vec = image_processing::process_image(&input_img);
    let input_tensor = Tensor::from_array(([1, 3, 256, 256], input_vec.into_boxed_slice()))?;
    print!("input_tensor: {:?}", input_tensor);

    let outputs = model.run(ort::inputs!["input" => input_tensor]?)?;
    let predictions = outputs["output"].try_extract_tensor::<f32>()?;
    let predictions = predictions.as_slice().unwrap();

    let (output, output_threshold, output_threshold_red) =
        image_processing::process_predictions(predictions, input_img.width(), input_img.height());

    let resized_input_img = input_img.resize_exact(256, 256, FilterType::CatmullRom);
    let overlap_image =
        image_processing::create_overlap_image(&resized_input_img, &output_threshold_red, 256, 256);
    let concat_img = image_processing::concatenate_images(
        &resized_input_img,
        &output,
        &output_threshold,
        &overlap_image,
    );

    // save in putput folder
    output.save(format!("{output_folder}/output.png"))?;
    output_threshold.save(format!("{output_folder}/output_threshold.png"))?;
    output_threshold_red.save(format!("{output_folder}/output_threshold_red.png"))?;
    overlap_image.save(format!("{output_folder}/overlap.png"))?;
    concat_img.save(format!("{output_folder}/concat_img.png"))?;

    Ok(())
}

pub fn folder() -> Result<(), Box<dyn std::error::Error>> {

    // 定義資料夾路徑
    let input_folder = "/home/yaocong/Dataset/SYN70K_dataset/testing_data/DS02/images";
    let output_folder = "./results/processed_images";

    // 創建輸出資料夾如果不存在
    fs::create_dir_all(output_folder)?;

    let model = model::create_model_session()?;

    // 遍歷資料夾中的所有圖像文件
    for entry in fs::read_dir(input_folder)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().map_or(false, |ext| ext == "png") {
            let input_img = image::open(&path)?;
            let input_vec = image_processing::process_image(&input_img);
            let input_tensor = Tensor::from_array(([1, 3, 256, 256], input_vec.into_boxed_slice()))?;
        
            let outputs = model.run(ort::inputs!["input" => input_tensor]?)?;

            let predictions = outputs["output"].try_extract_tensor::<f32>()?;
            let predictions = predictions.as_slice().unwrap();

            let (output, output_threshold, output_threshold_red) =
                image_processing::process_predictions(
                    predictions,
                    input_img.width(),
                    input_img.height(),
                );

            // input image and output_threshold image overlap
            let resized_input_img = input_img.resize_exact(256, 256, FilterType::CatmullRom);

            let overlap_image = image_processing::create_overlap_image(
                &resized_input_img,
                &output_threshold_red,
                256,
                256,
            );

            let concat_img = image_processing::concatenate_images(
                &resized_input_img,
                &output,
                &output_threshold,
                &overlap_image,
            );

            // 輸出文件名
            let file_name = path.file_name().unwrap().to_str().unwrap();
            let output_path = Path::new(output_folder).join(format!("concat_{}", file_name));

            // 保存拼接後的圖像
            concat_img.save(output_path)?;

            // // 輸出文件名
            // let file_name = path.file_name().unwrap().to_str().unwrap();
            // let output_path = Path::new(output_folder).join(file_name);
            // let output_threshold_path =
            //     Path::new(output_folder).join(format!("threshold_{}", file_name));

            // // 保存輸出圖像
            // output.save(output_path)?;
            // output_threshold.save(output_threshold_path)?;
        }
    }

    Ok(())
}

pub fn video() -> Result<(), Box<dyn std::error::Error>> {
    // 設置影片來源和目的地
    let video_path = "/home/yaocong/Dataset/smoke_video_dataset/Black_smoke_517.avi";
    let output_video_path = "output_video.mp4";

    let output_folder = "./results/processed_videos";
    fs::create_dir_all(output_folder)?;

    let model = model::create_model_session()?;

    // 打開影片檔案
    let mut cap = VideoCapture::from_file(video_path, CAP_ANY)?;
    if !cap.is_opened()? {
        panic!("Cannot open the video file");
    }

    // 取得影片的幀寬和幀高
    let frame_width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let frame_height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let fps = cap.get(videoio::CAP_PROP_FPS)?;

    // 設置影片寫入器
    let fourcc = VideoWriter::fourcc('m', 'p', '4', 'v')?;
    let mut writer = VideoWriter::new(
        output_video_path,
        fourcc,
        fps,
        core::Size::new(frame_width, frame_height),
        true,
    )?;

    // 讀取影片幀並寫入新影片檔案
    let mut frame = Mat::default();
    while cap.read(&mut frame)? {
        if frame.empty() {
            break;
        }
        let mut resized_frame = Mat::default();
        imgproc::resize(
            &frame,
            &mut resized_frame,
            opencv::core::Size::new(256, 256),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        frame = resized_frame;

        // 將幀轉換為模型的輸入
        let dynamic_image = image_processing::mat_to_imagebuffer(&frame)?;
        // print!("dynamic_image_type: {:?}", dynamic_image.color());
        let input_vec = image_processing::process_image(&dynamic_image);
        let input_tensor = Tensor::from_array(([1, 3, 256, 256], input_vec.into_boxed_slice()))?;
    
        let outputs = model.run(ort::inputs!["input" => input_tensor]?)?;
        let predictions = outputs["output"].try_extract_tensor::<f32>()?;
        let predictions = predictions.as_slice().unwrap();

        let (_output, _output_threshold, output_threshold_red) =
            image_processing::process_predictions(
                predictions,
                dynamic_image.width(),
                dynamic_image.height(),
            );

        let resized_input_img = dynamic_image.resize_exact(256, 256, FilterType::CatmullRom);
        let overlap_image = image_processing::create_overlap_image(
            &resized_input_img,
            &output_threshold_red,
            256,
            256,
        );

        frame = image_processing::imagebuffer_to_mat(&overlap_image)?;

        // 顯示影片幀
        highgui::imshow("Video", &frame)?;
        if highgui::wait_key(10)? == 27 {
            // 按下 'ESC' 鍵退出
            break;
        }

        writer.write(&frame)?;
    }

    // 釋放資源
    cap.release()?;
    writer.release()?;
    highgui::destroy_all_windows()?;
    Ok(())
}
