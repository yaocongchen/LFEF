use rust_code::utils::{model, image_processing};
use image::{DynamicImage, ImageBuffer};

use opencv::{
    core,
    highgui,
    prelude::*,
    videoio::{self, VideoCapture, VideoWriter, CAP_ANY},
    Result,
};
use image::imageops::FilterType;


use std::error::Error;



fn mat_to_dynamic_image(mat: &Mat) -> Result<DynamicImage, Box<dyn Error>> {
    let mut buffer = vec![0; (mat.rows() * mat.cols() * mat.channels() as i32) as usize];
    mat.data_typed::<u8>()?.copy_to_slice(&mut buffer);
    let image = DynamicImage::ImageRgb8(ImageBuffer::from_raw(mat.cols() as u32, mat.rows() as u32, buffer).unwrap());
    Ok(image)
}

fn dynamic_image_to_mat(image: &DynamicImage) -> Result<Mat, Box<dyn Error>> {
    let (width, height) = image.dimensions();
    let mut mat = Mat::new_rows_cols_with_data(
        height as i32,
        width as i32,
        core::CV_8UC3,
        image.to_bytes().as_slice(),
    )?;
    core::cvt_color(&mut mat, &mut mat, core::COLOR_RGB2BGR, 0)?;
    Ok(mat)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 設置影片來源和目的地
    let video_path = "/home/yaocong/Experimental/Dataset/smoke_video_dataset/Black_smoke_517.avi";
    let output_video_path = "output_video.mp4";
    
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
    let fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G')?;
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

        // 將幀轉換為模型的輸入
        let dynamic_image = mat_to_dynamic_image(&frame)?;
        let input = image_processing::process_image(&dynamic_image);
        
        let outputs = model.run(ort::inputs!["input" => input.view()]?)?;
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
        
        // 顯示影片幀
        highgui::imshow("Video", &frame)?;
        if highgui::wait_key(10)? == 27 { // 按下 'ESC' 鍵退出
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