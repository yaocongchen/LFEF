use rust_code::utils::{model, image_processing};

use image::imageops::FilterType;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    // 定義資料夾路徑
    let input_folder = "/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS02/images";
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
            let input = image_processing::process_image(&input_img);

            let outputs = model.run(ort::inputs!["input" => input.view()]?)?;
            let predictions = outputs["output"].try_extract_tensor::<f32>()?;
            let predictions = predictions.as_slice().unwrap();

            let (output, output_threshold, output_threshold_red) =
            image_processing::process_predictions(predictions, input_img.width(), input_img.height());
    
            
            // input image and output_threshold image overlap
            let resized_input_img = input_img.resize_exact(256, 256, FilterType::CatmullRom);
            
            let overlap_image =
            image_processing::create_overlap_image(&resized_input_img, &output_threshold_red, 256, 256);

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
