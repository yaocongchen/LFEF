use std::fs;
use image::imageops::FilterType;

use rust_code::utils::{model, image_processing};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let input_img = image::open(
        "/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/images/2.png",
    )
    .unwrap();
    let output_folder = "./results/processed_single_images";
    fs::create_dir_all(output_folder)?;

    input_img.save("input.png")?;

    let model = model::create_model_session()?;

    let input = image_processing::process_image(&input_img);

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

    // save in putput folder
    output.save(format!("{output_folder}/output.png"))?;
    output_threshold.save(format!("{output_folder}/output_threshold.png"))?;
    output_threshold_red.save(format!("{output_folder}/output_threshold_red.png"))?;
    overlap_image.save(format!("{output_folder}/overlap.png"))?;
    concat_img.save(format!("{output_folder}/concat_img.png"))?;

    Ok(())
}
