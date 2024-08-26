use std::process::Command;

fn main() {
    // 呼叫 process_single_image 二進位檔案
    let output = Command::new("cargo")
        .args(&["run", "--bin", "process_single_image"])
        .output()
        .expect("Failed to execute process_single_image");

    // 打印輸出
    println!("Status: {}", output.status);
    println!("Output: {}", String::from_utf8_lossy(&output.stdout));
    println!("Error: {}", String::from_utf8_lossy(&output.stderr));
}