from ultralytics import YOLO
if __name__ == '__main__':
    # 加载模型
    model = YOLO("yolov8-C2f_GhostModule-DynamicConv-CARAFE-RFAHead-SPPELAN-AKConv.yaml")  # 从头开始构建新模型
    results = model.train(data='F:/deep-learning-model/YOLOv8-Steel Detection/ultralytics/data/NEU-DET.yaml',
                          epochs=1000, patience=100,
                          name='yolov8-auto-n')  # 训练模型