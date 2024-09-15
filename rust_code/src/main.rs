mod process;
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    // 調用 process_single_image 模組的函數
    process::single_image()?;

    // 調用 process_folder 模組的函數
    process::folder()?;

    // 調用 process_video 模組的函數
    process::video()?;

    Ok(())
}
