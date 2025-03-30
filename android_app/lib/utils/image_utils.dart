import 'dart:io';
import 'package:image/image.dart' as img;
import 'dart:typed_data';

Future<File> image_processing(File input_image, File model_output_image) async {
  // 讀取圖片
  final bytes = await input_image.readAsBytes();
  final originalImage = img.decodeImage(Uint8List.fromList(bytes));

  if (originalImage == null) {
    throw Exception("無法解碼圖片");
  }
  // 讀取模型輸出圖片
  final modelOutputBytes = await model_output_image.readAsBytes();
  final modelOutputImage = img.decodeImage(
    Uint8List.fromList(modelOutputBytes),
  );
  if (modelOutputImage == null) {
    throw Exception("無法解碼模型輸出圖片");
  }

  //將運算出的圖片與原始圖片結合
  // 創建一個空的圖片
  final resizeImage = img.copyResize(originalImage, width: 256, height: 256);

  final overlap_image = img.Image(width: 256, height: 256); // 假設模型需要 RGB 格式

  for (int y = 0; y < 256; y++) {
    for (int x = 0; x < 256; x++) {
      final oriPixel = resizeImage.getPixel(x, y);
      final modelOutputPixel = modelOutputImage.getPixel(x, y);

      // 根據模型輸出設置像素值
      if (modelOutputPixel.r == 255) {
        overlap_image.setPixelRgba(
          x,
          y,
          255,
          oriPixel.g,
          oriPixel.b,
          255,
        ); // 設置為紅色
      } else {
        overlap_image.setPixelRgba(
          x,
          y,
          oriPixel.r,
          oriPixel.g,
          oriPixel.b,
          255,
        ); // 保留原始像素值
      }
    }
  }
  // 將處理後的圖片放大回原始圖片大小
  final restoredImage = img.copyResize(
    overlap_image,
    width: originalImage.width,
    height: originalImage.height,
  );

  // 儲存處理後的圖片
  final tempDir = Directory.systemTemp;
  final timestamp = DateTime.now().millisecondsSinceEpoch;
  final resultFile = File('${tempDir.path}/processed_image_$timestamp.png');
  await resultFile.writeAsBytes(img.encodePng(restoredImage));

  return resultFile;
}
