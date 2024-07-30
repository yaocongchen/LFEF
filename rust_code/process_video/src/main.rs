use image::{imageops::FilterType, GenericImageView, ImageBuffer};
use ndarray::Array;
use ort::{CUDAExecutionProvider, GraphOptimizationLevel, Session};
use opencv::{prelude::*, videoio};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let mut cap = videoio::VideoCapture::from_file(
        "/home/yaocong/Experimental/Dataset/smoke_video_dataset/Black_smoke_517.avi",
        videoio::CAP_ANY,
    )?;
    let fps = cap.get(videoio::CAP_PROP_FPS)?;
    let frame_width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let frame_height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;

    let mut writer = videoio::VideoWriter::new(
        "output_video.avi",
        videoio::VideoWriter::fourcc(b'M', b'J', b'P', b'G')?,
        fps,
        opencv::core::Size::new(256, 256),
        true,
    )?;

    let model = Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("../../trained_models/best.onnx")?;

    while let Ok(ret) = cap.grab() {
        if !ret {
            break;
        }

        let mut frame = opencv::core::Mat::default();
        cap.retrieve(&mut frame, 0)?;

        let img = opencv::imgcodecs::imencode(".png", &frame, &opencv::core::Vector::new())?;
        let img = image::load_from_memory(&img)?;

        let img = img.resize_exact(256, 256, FilterType::CatmullRom);
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

        let mut output = ImageBuffer::new(256, 256);
        for x in 0..256 {
            for y in 0..256 {
                let idx = x * 256 + y;
                let value = predictions[idx];
                let value = (value * 255.0) as u8;
                output.put_pixel(x as _, y as _, image::Rgb([value, value, value]));
            }
        }

        let output_img = opencv::imgcodecs::imdecode(&opencv::core::Vector::from(output.as_raw()), opencv::imgcodecs::IMREAD_COLOR)?;
        writer.write(&output_img)?;
    }

    Ok(())
}
