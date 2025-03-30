import 'dart:io';
import 'dart:core';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:onnxruntime/onnxruntime.dart' as ort;

class LFEF_App extends StatefulWidget {
  @override
  _LfefSegmentaion createState() => _LfefSegmentaion();
}

class _LfefSegmentaion extends State<LFEF_App> {
  File? _image; // 用於存儲選擇的圖片
  File? _modeloutputimage; // 用於存儲負片圖片
  final ImagePicker _picker = ImagePicker();
  int _currentIndex = 0; // 當前選中的索引

  Future<ort.OrtSession> loadModel() async {
    // 載入模型
    final sessionOptions = ort.OrtSessionOptions();
    // 啟用 NNAPI
    try {
      sessionOptions.appendNnapiProvider(ort.NnapiFlags.cpuOnly);
      print("NNAPI 已啟用");
    } catch (e) {
      print("NNAPI 無法啟用，回退到默認 CPU 提供者: $e");
    }

    final assetFileName = 'assets/models/best.onnx'; // 模型文件名稱
    final rawAssetFile = await DefaultAssetBundle.of(
      context,
    ).load(assetFileName); // 載入模型文件
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
    print(inputDataFloat32.sublist(0, 10)); // 打印前 10 個數據

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
          final gray = (grayChannel[y][x].toDouble() * 255).toInt().clamp(
            0,
            255,
          );

          // 閥值處理：將灰度直設置為 0 或 255
          final threshold = 127; // 閥值
          final binaryGray = gray > threshold ? 255 : 0;

          outputImage.setPixelRgba(
            x,
            y,
            binaryGray,
            binaryGray,
            binaryGray,
            255,
          );
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

  Future<File> image_processing(
    File input_image,
    File model_output_image,
  ) async {
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

  // 從相簿選擇圖片
  Future<void> pickImage() async {
    final pickedFile = await _picker.pickImage(
      source: ImageSource.gallery, // 使用相簿
    );

    if (pickedFile != null) {
      final originalImage = File(pickedFile.path); // 選擇的圖片
      print("選擇的圖片路徑: ${originalImage.path}");
      // final negativeImage = await _convertToNegative(originalImage); // 負片圖片

      setState(() {
        _image = originalImage; // 更新選擇的圖片
        // _modeloutputimage = negativeImage; // 更新負片圖片
      });

      // 載入模型並進行推論
      try {
        final session = await loadModel();
        final result = await runInference(session, _image!);
        session.release(); // 釋放模型會話
        ort.OrtEnv.instance.release(); // 釋放環境
        print("釋放模型會話和環境");
        final overlap_result = await image_processing(_image!, result); // 負片圖片

        setState(() {
          _modeloutputimage = overlap_result; // 更新處理後的圖片
        });
      } catch (e) {
        print('推論錯誤: $e');
        // 顯示錯誤提示
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('模型處理失敗: ${e.toString()}')));
      }
    }
  }

  // 從相機拍攝圖片
  Future<void> pickImageFromCamera() async {
    final pickedFile = await _picker.pickImage(
      source: ImageSource.camera, // 使用相機
    );

    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path); // 更新選擇的圖片
      });

      // 載入模型並進行推論
      try {
        final session = await loadModel();
        final result = await runInference(session, _image!);
        final overlap_result = await image_processing(_image!, result); // 負片圖片

        setState(() {
          _modeloutputimage = overlap_result; // 更新處理後的圖片
        });
      } catch (e) {
        print('推論錯誤: $e');
        // 顯示錯誤提示
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('模型處理失敗: ${e.toString()}')));
      }
    }
  }

  // // // 讀取影片並播放
  // Future<void> pickVideo() async {
  //   final pickedFile = await _picker.pickVideo(
  //     source: ImageSource.gallery, // 使用相簿
  //   );

  //   if (pickedFile != null) {
  //     setState(() {
  //       _image = File(pickedFile.path); // 更新選擇的圖片
  //     });
  //   }
  // }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("LFEF APP"), backgroundColor: Colors.blue),
      bottomNavigationBar: BottomNavigationBar(
        type: BottomNavigationBarType.fixed,
        currentIndex: _currentIndex, // 當前選中的索引
        onTap: (index) {
          setState(() {
            _currentIndex = index; // 更新當前選中的索引
          });
        },
        items: [
          BottomNavigationBarItem(icon: Icon(Icons.image), label: "圖片"),
          BottomNavigationBarItem(icon: Icon(Icons.video_call), label: "影片"),
          BottomNavigationBarItem(icon: Icon(Icons.camera), label: "相機"),
        ],
      ),
      body: IndexedStack(
        index: _currentIndex,
        children: [
          // 圖片頁面
          ListView(
            padding: EdgeInsets.all(16),
            children: [
              Text(
                "圖片推論",
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                textAlign: TextAlign.center,
              ),
              SizedBox(height: 20),

              // 顯示圖片或提示文字
              _image != null
                  ? Column(
                    children: [
                      Text("選擇圖片"),
                      Image.file(
                        _image!, // 顯示選擇的圖片
                        width: 256,
                        height: 256,
                        fit: BoxFit.cover,
                      ),
                    ],
                  )
                  : Container(
                    alignment: Alignment.center,
                    height: 100,
                    child: Text("尚未選擇圖片"),
                  ),

              SizedBox(height: 20),

              // 顯示模型輸出圖片或提示文字
              _modeloutputimage != null
                  ? Column(
                    children: [
                      Text("模型輸出圖片"),
                      Image.file(
                        _modeloutputimage!,
                        width: 256,
                        height: 256,
                        fit: BoxFit.cover,
                      ),
                    ],
                  )
                  : Container(),

              SizedBox(height: 20),

              // 按鈕區域
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  // 選擇圖片按鈕
                  Expanded(
                    child: ElevatedButton(
                      onPressed: pickImage,
                      child: Text("從相簿選擇圖片"),
                      style: ElevatedButton.styleFrom(
                        padding: EdgeInsets.symmetric(vertical: 12),
                      ),
                    ),
                  ),
                  SizedBox(width: 10),
                  // 拍攝圖片按鈕
                  Expanded(
                    child: ElevatedButton(
                      onPressed: pickImageFromCamera,
                      child: Text("從相機拍攝圖片"),
                      style: ElevatedButton.styleFrom(
                        padding: EdgeInsets.symmetric(vertical: 12),
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),

          // 影片頁面
          Center(
            child: Text(
              "影片功能尚未實現",
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
          ),

          // 相機頁面
          Center(
            child: Text(
              "相機功能尚未實現",
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
          ),
        ],
      ),
    );
  }
}

void main() {
  ort.OrtEnv.instance.init();
  runApp(
    MaterialApp(
      home: LFEF_App(),
      debugShowCheckedModeBanner: false, // 隱藏調試標籤
    ),
  );
}
