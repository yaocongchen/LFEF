use image::{imageops::FilterType, GenericImageView};
use ndarray::Array;
use ort::{CUDAExecutionProvider, GraphOptimizationLevel, Session};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let use_cuda = env::var("USE_CUDA").unwrap_or_else(|_| "false".to_string()) == "true";
    print!("use_cuda: {}", use_cuda);
    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    let input_img = image::open(
        "/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/images/2.png",
    )
    .unwrap();
    input_img.save("input.png")?;
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


    let model = Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("../../trained_models/best.onnx")?;


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
    output.save("output.png")?;
    output_threshold.save("output_threshold.png")?;

    Ok(())
}
