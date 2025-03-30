import 'dart:io';
import 'dart:core';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart' as ort;
import 'package:image/image.dart' as img;

Future<ort.OrtSession> loadModel() async {
  final sessionOptions = ort.OrtSessionOptions();
  try {
    sessionOptions.appendNnapiProvider(ort.NnapiFlags.cpuOnly);
    print("NNAPI 已啟用");
  } catch (e) {
    print("NNAPI 無法啟用，回退到默認 CPU 提供者: $e");
  }

  // 使用 rootBundle 加載資產
  final rawAssetFile = await rootBundle.load('assets/models/best.onnx');
  final bytes = rawAssetFile.buffer.asUint8List();
  final session = ort.OrtSession.fromBuffer(bytes, sessionOptions);
  return session;
}

Future<File> runInference(ort.OrtSession session, File imageFile) async {
  // 確保臨時目錄中沒有舊的推論結果文件
  final tempDir = Directory.systemTemp;
  final oldFiles = tempDir.listSync().where((file) {
    return file.path.contains('processed_image_') && file is File;
  });

  for (var file in oldFiles) {
    try {
      await file.delete(); // 刪除舊文件
      print('已刪除舊文件: ${file.path}');
    } catch (e) {
      print('刪除舊文件失敗: ${file.path}, 錯誤: $e');
    }
  }

  // 步驟 1: 讀取並預處理圖片
  final bytes = await imageFile.readAsBytes();
  final originalImage = img.decodeImage(Uint8List.fromList(bytes));

  if (originalImage == null) {
    throw Exception("無法解碼圖片");
  }

  // 步驟 2: 調整圖片大小為模型所需尺寸 (例如 256x256)
  final resizedImage = img.copyResize(
    originalImage,
    width: 256,
    height: 256,
    interpolation: img.Interpolation.cubic,
  );

  // 步驟 3: 準備模型輸入張量
  // final inputData = Float32List(256 * 256 * 3); // 假設模型需要 RGB 格式
  // int pixelIndex = 0;

  // for (int y = 0; y < resizedImage.height; y++) {
  //   for (int x = 0; x < resizedImage.width; x++) {
  //     final pixel = resizedImage.getPixel(x, y);
  //     // 提取 RGB 值並正規化到 0-1
  //     inputData[pixelIndex] = pixel.r / 255.0;
  //     inputData[pixelIndex + 1] = pixel.g / 255.0;
  //     inputData[pixelIndex + 2] = pixel.b / 255.0;
  //     pixelIndex += 3;
  //   }
  // }
  // 初始化一個三維數組來存儲通道

  List<List<List<double>>> inputData = List.generate(
    3,
    (_) => List.generate(256, (_) => List.filled(256, 0.0)),
  );

  // 填充每個通道
  for (int y = 0; y < resizedImage.height; y++) {
    for (int x = 0; x < resizedImage.width; x++) {
      final pixel = resizedImage.getPixel(x, y);
      inputData[0][x][y] = pixel.r / 255.0;
      inputData[1][x][y] = pixel.g / 255.0;
      inputData[2][x][y] = pixel.b / 255.0;
    }
  }

  // 將三維數組展平為一維數組
  final flatInputData = inputData.expand((e) => e).expand((e) => e).toList();
  // 將一維數組轉換為 Float32List
  final inputDataFloat32 = Float32List.fromList(flatInputData);
  // print("輸入數據類型: ${inputDataFloat32.runtimeType}");
  // print(inputDataFloat32.sublist(0, 10)); // 打印前 10 個數據

  // 步驟 4: 建立 ONNX 輸入張量
  final inputShape = [1, 3, 256, 256]; // 批次大小, 通道數, 高度, 寬度
  final inputTensor = ort.OrtValueTensor.createTensorWithDataList(
    inputDataFloat32,
    inputShape,
  );

  // 步驟 5: 執行推論
  final inputs = {'input': inputTensor}; // 假設模型輸入名稱為 'input'
  final OrtRunOptions = ort.OrtRunOptions();
  final outputs = await session.run(OrtRunOptions, inputs);

  inputTensor.release(); // 釋放輸入張量
  OrtRunOptions.release(); // 釋放運行選項

  // 步驟 6: 處理輸出
  final outputTensor = outputs.first; // 獲取第一個輸出張量
  if (outputTensor == null) {
    throw Exception("輸出張量為 null");
  }

  final outputData = outputTensor.value;
  // print("輸出數據: $outputData");
  // print("輸出數據: $outputData");
  print("輸出數據類型: ${outputData.runtimeType}");
  if (outputData is List) {
    final batchSize = outputData.length;
    final channelSize = (outputData[0] as List).length;
    final height = (outputData[0][0] as List).length;
    final width = (outputData[0][0][0] as List).length;

    print("輸出張量形狀: [$batchSize, $channelSize, $height, $width]");
  }

  // 確保輸出數據是多維數據
  if (outputData is List<List<List<List<dynamic>>>>) {
    // 假設輸出形狀為 [1, 1, 256, 256]
    final batch = outputData[0]; // 取出第一個批次
    // 如果只有一個通道，將其複製到三個通道
    final grayChannel = batch[0]; // 灰度通道

    // 創建一個空的圖片
    final outputImage = img.Image(width: 256, height: 256);

    for (int y = 0; y < 256; y++) {
      for (int x = 0; x < 256; x++) {
        // 提取灰度值，並將其複製到 RGB 通道
        final gray = (grayChannel[y][x].toDouble() * 255).toInt().clamp(0, 255);

        // 閥值處理：將灰度直設置為 0 或 255
        final threshold = 127; // 閥值
        final binaryGray = gray > threshold ? 255 : 0;

        outputImage.setPixelRgba(x, y, binaryGray, binaryGray, binaryGray, 255);
      }
    }

    final tempDir = Directory.systemTemp;
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final resultFile = File('${tempDir.path}/processed_image_$timestamp.png');
    await resultFile.writeAsBytes(img.encodePng(outputImage));

    print('處理後的圖片已保存: ${resultFile.path}');
    return resultFile;
  } else {
    throw Exception("無法處理的輸出格式: ${outputData.runtimeType}");
  }
}
