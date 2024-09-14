// mod process_folder;
// mod process_single_image;
mod process_video;
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    // // 調用 process_single_image 模組的函數
    // process_single_image::process_single_image()?;

    // // 調用 process_folder 模組的函數
    // process_folder::process_folder()?;

    // 調用 process_video 模組的函數
    process_video::process_video()?;

    Ok(())
}
