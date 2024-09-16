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
    let source = &args[2];

    let model = model::create_model_session(model_path)?;

    if source.ends_with(".png") || source.ends_with(".jpg") {
        process::single_image(&model, source)?;
    } else if source.ends_with(".mp4") || source.ends_with(".avi") {
        process::video(&model, source)?;
    } else if source.ends_with("0") {
        process::camera(&model, source)?;
    } else {
        process::folder(&model, source)?;
    }

    Ok(())
}
