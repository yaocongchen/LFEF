use crate::utils::image_processing;
use image::imageops::FilterType;
use opencv::{
    core, highgui, imgproc,
    prelude::*,
    videoio::{self, VideoCapture, VideoWriter, CAP_ANY},
    Result,
};
use ort::{session::Session, value::Tensor};
use std::fs;
use std::path::Path;

pub fn single_image(model: &Session, source: &str) -> Result<(), Box<dyn std::error::Error>> {
    let input_img = image::open(source).unwrap();
    let output_folder = "./results/processed_single_images";
    fs::create_dir_all(output_folder)?;

    // input_img.save("input.png")?;

    let input_vec = image_processing::process_image(&input_img);
    let input_tensor = Tensor::from_array(([1, 3, 256, 256], input_vec.into_boxed_slice()))?;
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

pub fn folder(model: &Session, source: &str) -> Result<(), Box<dyn std::error::Error>> {
    // 定義資料夾路徑
    let input_folder = source;
    let output_folder = "./results/processed_images";

    // 創建輸出資料夾如果不存在
    fs::create_dir_all(output_folder)?;

    // 遍歷資料夾中的所有圖像文件
    for entry in fs::read_dir(input_folder)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().map_or(false, |ext| ext == "png") {
            let input_img = image::open(&path)?;
            let input_vec = image_processing::process_image(&input_img);
            let input_tensor =
                Tensor::from_array(([1, 3, 256, 256], input_vec.into_boxed_slice()))?;

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

pub fn video(model: &Session, source: &str) -> Result<(), Box<dyn std::error::Error>> {
    // 設置影片來源和目的地
    let video_path = source;
    let output_video_name = "output_video.mp4";
    let output_folder = "./results/processed_videos";
    fs::create_dir_all(output_folder)?;
    let output_video_path = format!("{}/{}", output_folder, output_video_name);

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
        &output_video_path,
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

        frame = image_processing::process_frame(&model, &frame)?;

        // 顯示影片幀
        highgui::imshow("Video", &frame)?;
        if highgui::wait_key(10)? == 27 {
            // 按下 'ESC' 鍵退出
            break;
        }

        // 將 4 通道的幀轉換為 3 通道的幀
        let mut bgr_frame = Mat::default();
        imgproc::cvt_color(&frame, &mut bgr_frame, imgproc::COLOR_BGRA2BGR, 0)?;
        let mut restore_frame = Mat::default();
        imgproc::resize(
            &bgr_frame,
            &mut restore_frame,
            opencv::core::Size::new(frame_width, frame_height),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        // bgr_frame = restore_frame;
        writer.write(&restore_frame)?;
    }

    // 釋放資源
    cap.release()?;
    writer.release()?;
    highgui::destroy_all_windows()?;
    Ok(())
}

pub fn camera(model: &Session, source: &str) -> Result<(), Box<dyn std::error::Error>> {
    // 設置影片來源和目的地
    //camera_index to i32
    let camera_index = source.parse::<i32>().unwrap();
    let output_video_name = "output_video.mp4";
    let output_folder = "./results/processed_videos";
    fs::create_dir_all(output_folder)?;
    let output_video_path = format!("{}/{}", output_folder, output_video_name);

    // 打開影片檔案
    let mut camera: videoio::VideoCapture =
        videoio::VideoCapture::new(camera_index, videoio::CAP_ANY)?;
    if !camera.is_opened()? {
        panic!("Failed to open camera");
    }

    highgui::named_window("Camera Feed", highgui::WINDOW_AUTOSIZE)?;

    // 取得影片的幀寬和幀高
    let frame_width = camera.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let frame_height = camera.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let fps = camera.get(videoio::CAP_PROP_FPS)?;

    // 設置影片寫入器
    let fourcc = VideoWriter::fourcc('m', 'p', '4', 'v')?;
    let mut writer = VideoWriter::new(
        &output_video_path,
        fourcc,
        fps,
        core::Size::new(frame_width, frame_height),
        true,
    )?;

    // 讀取影片幀並寫入新影片檔案
    let mut frame = Mat::default();
    while camera.read(&mut frame)? {
        if frame.empty() {
            break;
        }

        frame = image_processing::process_frame(&model, &frame)?;

        // 顯示影片幀
        highgui::imshow("Camera Feed", &frame)?;
        if highgui::wait_key(10)? == 27 {
            // 按下 'ESC' 鍵退出
            break;
        }

        // 將 4 通道的幀轉換為 3 通道的幀
        let mut bgr_frame = Mat::default();
        imgproc::cvt_color(&frame, &mut bgr_frame, imgproc::COLOR_BGRA2BGR, 0)?;
        let mut restore_frame = Mat::default();
        imgproc::resize(
            &bgr_frame,
            &mut restore_frame,
            opencv::core::Size::new(frame_width, frame_height),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        // bgr_frame = restore_frame;
        writer.write(&restore_frame)?;
    }

    // 釋放資源
    camera.release()?;
    writer.release()?;
    highgui::destroy_all_windows()?;
    Ok(())
}
