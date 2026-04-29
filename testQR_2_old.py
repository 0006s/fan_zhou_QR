import cv2
import numpy as np
import time
import os
from collections import defaultdict

class QRScanner:
    def __init__(self, cam_index=0, model_dir=None):
        # 1. 初始化摄像头
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 (Index: {cam_index})，请检查硬件连接！")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 2. 初始化 WeChatQRCode（支持多码识别）
        detector = None
        try:
            if model_dir is None:
                detector = cv2.wechat_qrcode.WeChatQRCode()
                print("WeChatQRCode 初始化成功（内置模型）")
            else:
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
        
        # 3. 时间平滑参数（核心改进）
        self.qr_buffer = defaultdict(lambda: {'count': 0, 'points': None, 'last_seen': 0})  # 缓冲池
        self.confirm_thresh = 3      # 连续检测到3帧才确认
        self.drop_thresh = 5         # 连续丢失5帧才丢弃
        self.max_count = 10          # 计数上限，防止无限增长
        self.frame_count = 0         # 帧计数器
        
        # 4. 图像预处理参数
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # 5. 初始化 PIL 与字体预加载
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

    def preprocess_frame(self, frame):
        """图像预处理：CLAHE对比度增强"""
        # 转为LAB色彩空间，对L通道做CLAHE
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return enhanced

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
        print("多码二维码识别任务启动（WeChatQRCode + 时间平滑）... 按下 'q' 键退出")
        print(f"时间平滑参数: 确认阈值={self.confirm_thresh}帧, 丢弃阈值={self.drop_thresh}帧")
        print("提示: 画面中所有二维码都会被识别并显示（带时间平滑防抖）")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            self.frame_count += 1
            
            # --- 图像预处理 ---
            processed_frame = self.preprocess_frame(frame)
            
            # --- WeChatQRCode 多码识别 ---
            (decoded_data, points) = self.detector.detectAndDecode(processed_frame)
            
            # --- 时间平滑：更新缓冲池 ---
            # 当前帧检测到的所有码
            current_qrs = {}
            if decoded_data:
                for i, content in enumerate(decoded_data):
                    current_qrs[content] = points[i] if i < len(points) else None
            
            # 更新缓冲池：所有存在的码计数+1，不存在的码计数-1
            for content in list(self.qr_buffer.keys()):
                if content in current_qrs:
                    # 检测到了：计数+1，更新角点
                    self.qr_buffer[content]['count'] = min(self.qr_buffer[content]['count'] + 1, self.max_count)
                    self.qr_buffer[content]['points'] = current_qrs[content]
                    self.qr_buffer[content]['last_seen'] = self.frame_count
                else:
                    # 没检测到：计数-1
                    self.qr_buffer[content]['count'] -= 1
            
            # 新检测到的码加入缓冲池
            for content, pts in current_qrs.items():
                if content not in self.qr_buffer:
                    self.qr_buffer[content] = {'count': 1, 'points': pts, 'last_seen': self.frame_count}
            
            # 清理长期未检测到的码
            to_remove = []
            for content, data in self.qr_buffer.items():
                if data['count'] <= -self.drop_thresh:
                    to_remove.append(content)
            for content in to_remove:
                del self.qr_buffer[content]
            
            # --- 生成稳定结果：只显示计数 >= confirm_thresh 的码 ---
            stable_results = []
            stable_points = []
            for content, data in self.qr_buffer.items():
                if data['count'] >= self.confirm_thresh:
                    stable_results.append(content)
                    if data['points'] is not None:
                        stable_points.append(data['points'])
            
            # --- 多码可视化（绘制边框） ---
            for pts in stable_points:
                pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            
            # 绘制未稳定但正在确认的码（黄色，半透明）
            for content, data in self.qr_buffer.items():
                if 0 < data['count'] < self.confirm_thresh and data['points'] is not None:
                    pts = np.array(data['points'], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, (0, 255, 255), 1)  # 黄色细线
            
            # --- UI 绘制 ---
            roi_y_start = h - 80
            roi = frame[roi_y_start:h, 0:w]
            black_rect = np.zeros_like(roi)
            frame[roi_y_start:h, 0:w] = cv2.addWeighted(roi, 0.6, black_rect, 0.4, 0)
            
            # 显示稳定识别结果
            if stable_results:
                status_text = " | ".join([f"{i+1}:{r}" for i, r in enumerate(stable_results[:4])])
                status_color = (0, 255, 0)
                # 显示稳定码数量
                count_text = f"稳定: {len(stable_results)} 个"
                frame = self.draw_text(frame, count_text, (20, h-75), (0, 255, 0))
            else:
                status_text = "Waiting..."
                status_color = (0, 255, 255)
            
            frame = self.draw_text(frame, status_text, (20, h-45), status_color)
            
            # 调试信息：显示缓冲池状态
            debug_text = f"缓冲: {len(self.qr_buffer)}  帧: {self.frame_count}"
            frame = self.draw_text(frame, debug_text, (20, 30), (128, 128, 128))
            
            cv2.imshow("Raspberry Pi QR Scanner (Temporal Smoothing)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        MODEL_DIR = None
        scanner = QRScanner(cam_index=1, model_dir=MODEL_DIR) 
        scanner.scan()
    except Exception as e:
        print(f"程序出错: {e}")
