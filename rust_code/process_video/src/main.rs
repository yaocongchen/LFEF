use opencv::{
    core,
    prelude::*,
    videoio::{self, VideoCapture, VideoWriter, CAP_ANY},
    Result,
};

fn main() -> Result<()> {
    // 設置影片來源和目的地
    let video_path = "/home/yaocong/Experimental/Dataset/smoke_video_dataset/Black_smoke_517.avi";
    let output_video_path = "output_video.mp4";

    // 打開影片檔案
    let mut cap = VideoCapture::from_file(video_path, CAP_ANY)?;
    if !cap.is_opened()? {
        panic!("Cannot open the video file");
    }

    // 取得影片的幀寬和幀高
    let frame_width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let frame_height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let fps = cap.get(videoio::CAP_PROP_FPS)?;

    // 設置影片寫入器
    let fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G')?;
    let mut writer = VideoWriter::new(
        output_video_path,
        fourcc,
        fps,
        core::Size::new(frame_width, frame_height),
        true,
    )?;

    // 讀取影片幀並寫入新影片檔案
    let mut frame = Mat::default();
    while cap.read(&mut frame)? {
        if frame.empty() {
            break;
        }
        writer.write(&frame)?;
    }

    // 釋放資源
    cap.release()?;
    writer.release()?;
    Ok(())
}