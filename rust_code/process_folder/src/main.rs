use image::{imageops::FilterType, GenericImageView};
use ndarray::Array;
use ort::{CUDAExecutionProvider, GraphOptimizationLevel, Session};
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    
    // 定義資料夾路徑
    let input_folder = "/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/images";
    let output_folder = "./results/processed_images";

    // 創建輸出資料夾如果不存在
    fs::create_dir_all(output_folder)?;

    // 建立 ONNX 模型會話
    let model = Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("../../trained_models/best.onnx")?;

    // 遍歷資料夾中的所有圖像文件
    for entry in fs::read_dir(input_folder)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().map_or(false, |ext| ext == "png") {
            let input_img = image::open(&path)?;
            let (_img_width, _img_height) = (input_img.width(), input_img.height());
            let img = input_img.resize_exact(256, 256, FilterType::CatmullRom);
            let mut input = Array::zeros((1, 3, 256, 256));
            for pixel in img.pixels() {
                let x = pixel.0 as _;
                let y = pixel.1 as _;
                let [r, g, b, _] = pixel.2 .0;
                input[[0, 0, x, y]] = r as f32 / 255.0;
                input[[0, 1, x, y]] = g as f32 / 255.0;
                input[[0, 2, x, y]] = b as f32 / 255.0;
            }

            let outputs = model.run(ort::inputs!["input" => input.view()]?)?;
            let predictions = outputs["output"].try_extract_tensor::<f32>()?;
            let predictions = predictions.as_slice().unwrap();
            let mut output = image::ImageBuffer::new(256, 256);
            let mut output_threshold = image::ImageBuffer::new(256, 256);
            for x in 0..256 {
                for y in 0..256 {
                    let idx = x * 256 + y;
                    let value = predictions[idx];
                    let value = (value * 255.0) as u8;
                    // value > 127 = 255; value <= 127 = 0
                    output.put_pixel(x as _, y as _, image::Rgb([value, value, value]));
                    let value_threshold: u8 = if value > 127 { 255 } else { 0 };
                    output_threshold.put_pixel(
                        x as _,
                        y as _,
                        image::Rgb([value_threshold, value_threshold, value_threshold]),
                    );
                }
            }

            // 輸出文件名
            let file_name = path.file_name().unwrap().to_str().unwrap();
            let output_path = Path::new(output_folder).join(file_name);
            let output_threshold_path = Path::new(output_folder).join(format!("threshold_{}", file_name));

            // 保存輸出圖像
            output.save(output_path)?;
            output_threshold.save(output_threshold_path)?;
        }
    }

    Ok(())
}
