mod process;
mod utils;
use utils::model;

use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: {} <model_path>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let file_path = &args[2];

    let model = model::create_model_session(model_path)?;

    if file_path.ends_with(".png") || file_path.ends_with(".jpg") {
        process::single_image(&model, file_path)?;
    } else if file_path.ends_with(".mp4") || file_path.ends_with(".avi") {
        process::video(&model, file_path)?;
    } else {
        process::folder(&model, file_path)?;
    }

    Ok(())
}
