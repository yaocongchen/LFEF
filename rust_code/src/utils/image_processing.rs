use image::{imageops::FilterType, DynamicImage, GenericImageView, ImageBuffer, Rgb, Rgba, RgbaImage};
use ndarray::Array;
use opencv::core::{Mat, CV_8UC3};
use opencv::{
    imgproc,
    prelude::*,
    Result,
};
use ort::{session::Session, value::Tensor};


pub fn process_image(input_img: &DynamicImage) -> Vec<f32> {
    let img = input_img.resize_exact(256, 256, FilterType::CatmullRom);
    let mut input = Array::zeros((1, 3, 256, 256));
    for pixel in img.pixels() {
        let x = pixel.0 as usize;
        let y = pixel.1 as usize;
        let [r, g, b, _] = pixel.2 .0;
        input[[0, 0, x, y]] = r as f32 / 255.0;
        input[[0, 1, x, y]] = g as f32 / 255.0;
        input[[0, 2, x, y]] = b as f32 / 255.0;
    }
    input.into_raw_vec_and_offset().0
}

pub fn process_predictions(
    predictions: &[f32],
    width: u32,
    height: u32,
) -> (
    ImageBuffer<Rgba<u8>, Vec<u8>>,
    ImageBuffer<Rgba<u8>, Vec<u8>>,
    ImageBuffer<Rgba<u8>, Vec<u8>>,
) {
    let mut output = ImageBuffer::new(width, height);
    let mut output_threshold = ImageBuffer::new(width, height);
    let mut output_threshold_red = ImageBuffer::new(width, height);

    for x in 0..width {
        for y in 0..height {
            let idx = (x * height + y) as usize;
            let value = predictions[idx];
            let value = (value * 255.0) as u8;
            // value > 127 = 255; value <= 127 = 0
            output.put_pixel(x, y, Rgba([value, value, value, 255]));
            let value_threshold: u8 = if value > 127 { 255 } else { 0 };
            output_threshold.put_pixel(
                x,
                y,
                Rgba([value_threshold, value_threshold, value_threshold, 255]),
            );
            let threshold_red: u8 = if value_threshold == 255 { 255 } else { 0 };
            output_threshold_red.put_pixel(x, y, Rgba([threshold_red, 0, 0, 255]));
        }
    }

    (output, output_threshold, output_threshold_red)
}

pub fn create_overlap_image(
    resized_input_img: &DynamicImage,
    output_threshold_red: &RgbaImage,
    width: u32,
    height: u32,
) -> RgbaImage {
    assert_eq!(
        resized_input_img.dimensions(),
        output_threshold_red.dimensions()
    );

    let mut overlap_image = RgbaImage::new(width, height);

    for x in 0..width {
        for y in 0..height {
            let pixel1 = resized_input_img.get_pixel(x, y);
            let pixel2 = output_threshold_red.get_pixel(x, y);

            let r = pixel1[0].saturating_add(pixel2[0]);
            let g = pixel1[1].saturating_add(pixel2[1]);
            let b = pixel1[2].saturating_add(pixel2[2]);
            let a = pixel1[3].saturating_add(pixel2[3]);

            overlap_image.put_pixel(x, y, Rgba([r, g, b, a]));
        }
    }

    overlap_image
}

pub fn concatenate_images(
    resized_input_img: &DynamicImage,
    output: &RgbaImage,
    output_threshold: &RgbaImage,
    overlap_image: &RgbaImage,
) -> RgbaImage {
    let total_width = resized_input_img.width()
        + output.width()
        + output_threshold.width()
        + overlap_image.width();
    let height = resized_input_img.height();
    let mut concat_img = RgbaImage::new(total_width, height);

    image::imageops::overlay(&mut concat_img, resized_input_img, 0, 0);
    image::imageops::overlay(&mut concat_img, output, resized_input_img.width() as i64, 0);
    image::imageops::overlay(
        &mut concat_img,
        output_threshold,
        (resized_input_img.width() + output.width()) as i64,
        0,
    );
    image::imageops::overlay(
        &mut concat_img,
        overlap_image,
        (resized_input_img.width() + output.width() + output_threshold.width()) as i64,
        0,
    );

    concat_img
}

pub fn mat_to_imagebuffer(mat: &Mat) -> Result<DynamicImage, Box<dyn std::error::Error>> {
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


pub fn imagebuffer_to_mat(buffer: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> Result<Mat, Box<dyn std::error::Error>> {
    let (_width, height) = buffer.dimensions();
    let data = buffer.as_raw();
    let mat = Mat::from_slice(data)?;
    let mat = mat.reshape(4, height as i32)?;
    let mat_clane: Mat = mat.try_clone()?;
    Ok(mat_clane)
}

pub fn process_frame(model:&Session, frame:&Mat) -> Result<Mat, Box<dyn std::error::Error>> {
    let mut resized_frame = Mat::default();
    imgproc::resize(
        &frame,
        &mut resized_frame,
        opencv::core::Size::new(256, 256),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let dynamic_image = mat_to_imagebuffer(&resized_frame)?;
    let input_vec = process_image(&dynamic_image);
    let input_tensor = Tensor::from_array(([1, 3, 256, 256], input_vec.into_boxed_slice()))?;
    let outputs = model.run(ort::inputs!["input" => input_tensor]?)?;
    let predictions = outputs["output"].try_extract_tensor::<f32>()?;
    let predictions = predictions.as_slice().unwrap();

    let (_output, _output_threshold, output_threshold_red) =
        process_predictions(
            predictions,
            dynamic_image.width(),
            dynamic_image.height(),
        );

    let resized_input_img = dynamic_image.resize_exact(256, 256, FilterType::CatmullRom);
    let overlap_image = create_overlap_image(
        &resized_input_img,
        &output_threshold_red,
        256,
        256,
    );

    let frame = imagebuffer_to_mat(&overlap_image)?;
    Ok(frame)
}
