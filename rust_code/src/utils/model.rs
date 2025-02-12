use ort::{execution_providers::{CPUExecutionProvider, CUDAExecutionProvider}, session::builder::GraphOptimizationLevel, session::Session};
use std::process::Command;
use std::str;

fn is_cuda_available() -> bool {
    // 使用 `nvidia-smi` 命令來檢查 CUDA 是否可用
    if let Ok(output) = Command::new("nvidia-smi").output() {
        if let Ok(stdout) = str::from_utf8(&output.stdout) {
            if !stdout.contains("No devices were found") && !stdout.is_empty() {
                return true;
            }
        }
    }

    // 默認返回 false，表示沒有可用的 GPU
    false
}

pub fn create_model_session(model_path: &str) -> Result<Session, Box<dyn std::error::Error>> {
    let session_builder = Session::builder()?;
    
    // 根據是否有 CUDA GPU 來決定使用 GPU 或 CPU
    let use_cuda = is_cuda_available();
    
    let session_builder = if use_cuda {
        println!("CUDA provider is available.");
        session_builder.with_execution_providers([CUDAExecutionProvider::default().build()])?
    } else {
        println!("CUDA provider is not available. Falling back to CPU.");
        session_builder.with_execution_providers([CPUExecutionProvider::default().build()])?
    };

    let model = session_builder
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    Ok(model)
}
