import onnx

# 載入模型
model = onnx.load("./trained_models/best.onnx")

# 獲取模型的圖
graph = model.graph

# 列出所有輸入名稱
input_names = [input.name for input in graph.input]
print("Model input names:", input_names)