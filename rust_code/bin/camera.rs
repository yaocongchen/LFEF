extern crate opencv;

use opencv::{
    highgui,
    prelude::*,
    videoio,
    Result,
};

fn main() -> Result<()> {
    // \u958b\u555f\u76f8\u6a5f\uff080 \u901a\u5e38\u662f\u9810\u8a2d\u76f8\u6a5f\uff09
    let mut camera: videoio::VideoCapture = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;

    // \u6aa2\u67e5\u76f8\u6a5f\u662f\u5426\u6210\u529f\u958b\u555f
    if !camera.is_opened()? {
        panic!("Failed to open camera");
    }

    // \u5275\u5efa\u7a97\u53e3
    highgui::named_window("Camera Feed", highgui::WINDOW_AUTOSIZE)?;

    loop {
        let mut frame = Mat::default();
        // \u6355\u6349\u4e00\u5e40\u5f71\u50cf
        camera.read(&mut frame)?;

        // \u5982\u679c\u6355\u6349\u5931\u6557\uff0c\u9000\u51fa\u5faa\u74b0
        if frame.empty() {
            break;
        }

        // \u986f\u793a\u5f71\u50cf
        highgui::imshow("Camera Feed", &frame)?;

        // \u6309 'q' \u9375\u9000\u51fa
        if highgui::wait_key(1)? == 113 {
            break;
        }
    }

    Ok(())
}
