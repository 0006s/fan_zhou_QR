import cv2
import numpy as np
import time
import os

class QRScanner:
    def __init__(self, cam_index=0, model_dir=None):
        # 1. 初始化摄像头
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 (Index: {cam_index})，请检查硬件连接！")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 2. 初始化 WeChatQRCode（支持多码识别）
        # 自动检测模型路径
        detector = None
        try:
            if model_dir is None:
                # 自动使用无参版本（模型已内置于 opencv-contrib-python）
                detector = cv2.wechat_qrcode.WeChatQRCode()
                print("WeChatQRCode 初始化成功（内置模型）")
            else:
                # 指定模型路径
                pbtxt = os.path.join(model_dir, "detect.prototxt")
                caffemodel = os.path.join(model_dir, "detect.caffemodel")
                sr_pbtxt = os.path.join(model_dir, "sr.prototxt")
                sr_caffemodel = os.path.join(model_dir, "sr.caffemodel")
                detector = cv2.wechat_qrcode.WeChatQRCode(pbtxt, caffemodel, sr_pbtxt, sr_caffemodel)
                print(f"WeChatQRCode 初始化成功（模型目录: {model_dir}）")
        except Exception as e:
            print(f"WeChatQRCode 初始化失败: {e}")
            raise
        self.detector = detector
        
        # 3. 结果缓存
        self.results_display = []  # 支持多码同时显示
        self.last_scan_time = 0 
        self.timeout_seconds = 2.0  # 2秒未扫到二维码，UI恢复等待状态
        
        # 4. 初始化 PIL 与字体预加载
        self.use_pil = False
        self.font = None
        try:
            from PIL import Image, ImageDraw, ImageFont
            font_path = "C:\\Windows\\Fonts\\simhei.ttf" if os.name == 'nt' else "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
            if os.path.exists(font_path):
                self.font = ImageFont.truetype(font_path, 28)
                self.use_pil = True
                print("检测到 PIL 和字体资源，已开启中文显示支持")
            else:
                print("未找到中文字体路径，将降级使用 OpenCV 默认字体")
        except ImportError:
            print("未检测到 PIL 库，建议执行: pip install pillow")

    def draw_text(self, img, text, position, color):
        """支持中文的文字绘制函数"""
        if not self.use_pil or self.font is None:
            return cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        from PIL import Image, ImageDraw
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=self.font, fill=color[::-1]) 
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def scan(self):
        print("多码二维码识别任务启动（WeChatQRCode）... 按下 'q' 键退出")
        print("提示: 画面中所有二维码都会被识别并显示")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            
            # --- WeChatQRCode 多码识别 ---
            (decoded_data, points) = self.detector.detectAndDecode(frame)

            if decoded_data:
                self.results_display = list(decoded_data)  # 存储所有识别结果
                self.last_scan_time = time.time()
            else:
                # 超过超时时间，清空结果
                if time.time() - self.last_scan_time > self.timeout_seconds:
                    self.results_display = []

            # --- 多码可视化（绘制边框） ---
            # points 是每个二维码的四个角点
            for pts in points:
                pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            
            # --- UI 绘制 ---
            roi_y_start = h - 60
            roi = frame[roi_y_start:h, 0:w]
            black_rect = np.zeros_like(roi)
            frame[roi_y_start:h, 0:w] = cv2.addWeighted(roi, 0.6, black_rect, 0.4, 0)
            
            # 显示识别结果（多码）
            if self.results_display:
                status_text = " | ".join([f"{i+1}:{r}" for i, r in enumerate(self.results_display[:4])])
                status_color = (0, 255, 0)
            else:
                status_text = "Waiting..."
                status_color = (0, 255, 255)
            
            frame = self.draw_text(frame, status_text, (20, h-45), status_color)
            
            cv2.imshow("Raspberry Pi QR Scanner", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # 模型目录（不填则使用内置模型）
        MODEL_DIR = None  # 或 "C:/qr_models"
        
        scanner = QRScanner(cam_index=1, model_dir=MODEL_DIR) 
        scanner.scan()
    except Exception as e:
        print(f"程序出错: {e}")
