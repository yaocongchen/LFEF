import 'package:flutter/material.dart';
import 'package:onnxruntime/onnxruntime.dart' as ort;
import 'screens/image_screen.dart';
import 'screens/video_screen.dart';
import 'screens/camera_screen.dart';

void main() {
  ort.OrtEnv.instance.init();
  runApp(
    MaterialApp(
      home: LFEF_App(),
      debugShowCheckedModeBanner: false, // 隱藏調試標籤
    ),
  );
}

class LFEF_App extends StatefulWidget {
  @override
  _LfefSegmentaion createState() => _LfefSegmentaion();
}

class _LfefSegmentaion extends State<LFEF_App> {
  int _currentIndex = 0; // 當前選中的索引

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
          ImageScreen(), // 圖片頁面
          VideoScreen(), // 影片頁面
          CameraScreen(), // 相機頁面
        ],
      ),
    );
  }
}
