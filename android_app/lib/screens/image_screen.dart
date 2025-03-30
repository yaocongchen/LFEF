import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:onnxruntime/onnxruntime.dart' as ort;
import '../utils/model_utils.dart';
import '../utils/image_utils.dart';

class ImageScreen extends StatefulWidget {
  @override
  _ImageScreenState createState() => _ImageScreenState();
}

class _ImageScreenState extends State<ImageScreen> {
  File? _image; // 用於存儲選擇的圖片
  File? _modelOutputImage; // 用於存儲模型輸出圖片
  final ImagePicker _picker = ImagePicker();

  Future<void> pickImage() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      final originalImage = File(pickedFile.path);
      setState(() {
        _image = originalImage;
      });

      try {
        final session = await loadModel();
        final result = await runInference(session, _image!);
        session.release(); // 釋放會話
        ort.OrtEnv.instance.release(); // 釋放環境
        final overlapResult = await image_processing(_image!, result);

        setState(() {
          _modelOutputImage = overlapResult;
        });
      } catch (e) {
        print('推論錯誤: $e');
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
        session.release(); // 釋放會話
        ort.OrtEnv.instance.release(); // 釋放環境
        final overlap_result = await image_processing(_image!, result); // 負片圖片

        setState(() {
          _modelOutputImage = overlap_result; // 更新處理後的圖片
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

  @override
  Widget build(BuildContext context) {
    return ListView(
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
        _modelOutputImage != null
            ? Column(
              children: [
                Text("模型輸出圖片"),
                Image.file(
                  _modelOutputImage!,
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
    );
  }
}
