use ort::{CUDAExecutionProvider, GraphOptimizationLevel, Session};

pub fn create_model_session() -> Result<Session, Box<dyn std::error::Error>> {
    let model = Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("../trained_models/best.onnx")?;
    Ok(model)
}
