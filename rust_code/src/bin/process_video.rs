use opencv::core::{CV_8UC3, Mat};
use opencv::imgproc;
use rust_code::utils::{model, image_processing};
use image::{DynamicImage, ImageBuffer, Rgb, Rgba};

use opencv::{
    core,
    highgui,
    prelude::*,
    videoio::{self, VideoCapture, VideoWriter, CAP_ANY},
    Result,
};
use image::imageops::FilterType;
use std::fs;

fn mat_to_imagebuffer(mat: &Mat) -> Result<DynamicImage, Box<dyn std::error::Error>> {
    // 檢查 Mat 是否為 8 位 3 通道的圖像
    if mat.typ() != CV_8UC3 {
        return Err("Mat 必須是 CV_8UC3 格式".into());
    }

    // 獲取 Mat 的尺寸
    let size = mat.size()?;
    let width = size.width as u32;
    let height = size.height as u32;

    // 獲取 Mat 的數據
    let data = mat.data_bytes()?;

    // 創建 ImageBuffer
    let buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, data.to_vec())
        .ok_or("無法從 Mat 數據創建 ImageBuffer")?;

    let dynamic_image = DynamicImage::ImageRgb8(buffer);

    Ok(dynamic_image)
}

fn imagebuffer_to_mat(img: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> Result<Mat, Box<dyn std::error::Error>> {
    // 獲取 ImageBuffer 的寬度和高度
    //img ,rgba to rgb
    // 假設 img 是 ImageBuffer<Rgba<u8>, Vec<u8>>

    // 手動轉換為 ImageBuffer<Rgb<u8>, Vec<u8>>
    let rgb_img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        let pixel = img.get_pixel(x, y);
        Rgb([pixel[0], pixel[1], pixel[2]])
    });
    // ImageBuffer rgb to gray
    let gray_img = image::imageops::grayscale(&rgb_img);
    
    let (width, height) = gray_img.dimensions();

    // 獲取 ImageBuffer 的原始數據
    let mut img_data = gray_img.into_raw();

    // 創建 Mat，使用 Mat::new_rows_cols_with_data 並指定行列和類型

    let mat = Mat::new_rows_cols_with_data_mut(height as i32,width as i32, &mut img_data)?;
    let mat_clone : Mat = mat.try_clone()?;

    Ok(mat_clone)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    // 設置影片來源和目的地
    let video_path = "/home/yaocong/Experimental/Dataset/smoke_video_dataset/Black_smoke_517.avi";
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
        imgproc::resize(&frame, &mut resized_frame, opencv::core::Size::new(256, 256), 0.0, 0.0, imgproc::INTER_LINEAR)?;
        frame = resized_frame;

        // 將幀轉換為模型的輸入
        let dynamic_image = mat_to_imagebuffer(&frame)?;
        // print!("dynamic_image_type: {:?}", dynamic_image.color());
        let input = image_processing::process_image(&dynamic_image);
        
        let outputs = model.run(ort::inputs!["input" => input.view()]?)?;
        let predictions = outputs["output"].try_extract_tensor::<f32>()?;
        let predictions = predictions.as_slice().unwrap();

        let (_output, _output_threshold, output_threshold_red) =
            image_processing::process_predictions(predictions, dynamic_image.width(), dynamic_image.height());

        let resized_input_img = dynamic_image.resize_exact(256, 256, FilterType::CatmullRom);
        let overlap_image =
            image_processing::create_overlap_image(&resized_input_img, &output_threshold_red, 256, 256);

        frame = imagebuffer_to_mat(&overlap_image)?;

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