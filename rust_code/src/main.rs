use image::{imageops::FilterType, GenericImageView, RgbaImage};
use ndarray::Array;
use ort::{CUDAExecutionProvider, GraphOptimizationLevel, Session};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let input_img = image::open(
        "/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/images/2.png",
    )
    .unwrap();

    input_img.save("input.png")?;
    
    let model = Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("../trained_models/best.onnx")?;

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
    let mut output_threshold_red = image::ImageBuffer::new(256, 256);
    for x in 0..256 {
        for y in 0..256 {
            let idx = x * 256 + y;
            let value = predictions[idx];
            let value = (value * 255.0) as u8;
            // value > 127 = 255; value <= 127 = 0
            output.put_pixel(x as _, y as _, image::Rgba([value, value, value, 255]));
            let value_threshold: u8 = if value > 127 { 255 } else { 0 };
            output_threshold.put_pixel(
                x as _,
                y as _,
                image::Rgba([value_threshold, value_threshold, value_threshold, 255]),
            );
            let threshold_red: u8 = if value_threshold == 255 { 255 } else { 0 };
            output_threshold_red.put_pixel(
                x as _,
                y as _,
                image::Rgba([threshold_red, 0, 0, 255]),
            );
        }
    }



    let resized_input_img = input_img.resize_exact(256, 256, FilterType::CatmullRom);
    
    assert_eq!(resized_input_img.dimensions(), output_threshold_red.dimensions());

    let (width, height) = resized_input_img.dimensions();
    let mut overlap_image = RgbaImage::new(width, height);

    for x in 0..width {
        for y in 0..height {
            let pixel1 = resized_input_img.get_pixel(x, y);
            let pixel2 = output_threshold_red.get_pixel(x, y);

            let r = pixel1[0].saturating_add(pixel2[0]);
            let g = pixel1[1].saturating_add(pixel2[1]);
            let b = pixel1[2].saturating_add(pixel2[2]);
            let a = pixel1[3].saturating_add(pixel2[3]);

            overlap_image.put_pixel(x, y, image::Rgba([r, g, b, a]));
        }
    }

    let mut concat_img = RgbaImage::new(resized_input_img.width() + output.width() + output_threshold.width() + overlap_image.width(), resized_input_img.height());
    
    image::imageops::overlay(&mut concat_img, &resized_input_img, 0, 0);
    image::imageops::overlay(&mut concat_img, &output, resized_input_img.width() as i64, 0);
    image::imageops::overlay(
        &mut concat_img,
        &output_threshold,
        (resized_input_img.width() + output.width()) as i64,
        0,
    );
    image::imageops::overlay(
        &mut concat_img,
        &overlap_image,
        (resized_input_img.width() + output.width() + output_threshold.width()) as i64,
        0,
    );
    
    output.save("output.png")?;
    output_threshold.save("output_threshold.png")?;
    output_threshold_red.save("output_threshold_red.png")?;
    overlap_image.save("overlap.png")?;
    concat_img.save("concat_img.png")?;

    Ok(())
}
