import cv2
import time
import os
import threading
import queue
import serial
import re
from collections import defaultdict

# ==========================================
# 核心配置区 (比赛前请务必确认这里的值)
# ==========================================
DEBUG_MODE = True            # 调试模式开关：True显示画面，False关闭画面全力计算
SERIAL_PORT = 'COM3'  #'/dev/ttyUSB0' # 树莓派上的串口设备号 (Windows测试时改为 'COM3')
SERIAL_BAUD = 115200         # 串口波特率
CAM_INDEX = 1                # 摄像头序号

class CompetitionVisionNode:
    def __init__(self):
        print("[初始化] 正在启动视觉节点...")
        
        # 1. 初始化线程队列
        self.q_frame = queue.Queue(maxsize=1)  # 保证永远只处理最新一帧
        self.q_serial = queue.Queue(maxsize=5) # 存放准备发送的有效数据

        # 2. 初始化摄像头
        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # 建议提高分辨率应对2x2矩阵
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 3. 初始化微信二维码引擎
        model_dir = os.path.join(os.path.expanduser("~"), ".qclaw", "workspace", "wechat_qrcode_models")
        # 如果模型不在上述目录，请手动修改为你电脑上的实际路径！
        self.detector = cv2.wechat_qrcode.WeChatQRCode(
            os.path.join(model_dir, "detect.prototxt"),
            os.path.join(model_dir, "detect.caffemodel"),
            os.path.join(model_dir, "sr.prototxt"),
            os.path.join(model_dir, "sr.caffemodel")
        )
        
        # 4. 初始化图像预处理与防抖算法 (继承自你的优秀设计)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.qr_buffer = defaultdict(lambda: {'count': 0, 'points': None, 'last_seen': 0})
        self.confirm_thresh = 3
        self.drop_thresh = 5
        self.max_count = 10
        self.frame_count = 0

        # 5. 启动子线程
        self.running = True
        self.t_cam = threading.Thread(target=self._camera_thread, daemon=True)
        self.t_ser = threading.Thread(target=self._serial_thread, daemon=True)
        self.t_cam.start()
        self.t_ser.start()

    def preprocess_frame(self, frame):
        """CLAHE 对比度增强"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def is_valid_format(self, text):
        """
        验证规则：必须包含 颜色+专属编码+核心编码。
        假设赛场标准格式为 "Red,1,3" 或 "蓝,2,3"。这里用正则表达式匹配。
        你可以根据最终和组委会确认的格式修改这里的正则。
        """
        # 匹配：任意字母或中文字符 + 逗号 + 数字 + 逗号 + 数字
        pattern = r'^([a-zA-Z\u4e00-\u9fa5]+),(\d+),(\d+)$'
        return bool(re.match(pattern, text.strip()))

    def _camera_thread(self):
        """摄像头读取线程：只管疯狂抓图，保证画面零延迟"""
        print("[子线程] 摄像头线程已启动")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # 队列满了就扔掉旧画面，永远只塞最新的
            if self.q_frame.full():
                try: self.q_frame.get_nowait()
                except queue.Empty: pass
            self.q_frame.put(frame)

    def _serial_thread(self):
        """串口通信线程：自带断线重连机制，防止实车震动掉线"""
        print("[子线程] 串口线程已启动")
        ser = None
        while self.running:
            # 尝试连接串口
            if ser is None or not ser.is_open:
                try:
                    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
                    print(f"[串口] 成功连接 {SERIAL_PORT}")
                except Exception as e:
                    print(f"[串口报错] 无法连接串口，1秒后重试... ({e})")
                    time.sleep(1)
                    continue

            # 获取主线程发来的数据并发送
            try:
                # 阻塞等待数据，0.5秒超时以便能够检查 self.running
                data_to_send = self.q_serial.get(timeout=0.5) 
                
                # 加上换行符作为帧尾发送给 STM32
                packet = f"{data_to_send}\r\n".encode('utf-8')
                ser.write(packet)
                print(f"[串口发送] -> {data_to_send}")
                
            except queue.Empty:
                pass 
            except Exception as e:
                print(f"[串口报错] 发送失败，准备重连... ({e})")
                ser.close()

    def run(self):
        """主线程：专注图像算法处理"""
        print("\n=== 视觉节点已全面就绪 ===")
        print(f"当前模式: {'调试模式 (开启画面)' if DEBUG_MODE else '比赛模式 (极限性能, 无画面)'}")
        
        while True:
            if self.q_frame.empty():
                time.sleep(0.005)
                continue
                
            frame = self.q_frame.get()
            self.frame_count += 1
            
            # 1. 预处理
            processed = self.preprocess_frame(frame)
            
            # 2. 识别
            decoded_data, points = self.detector.detectAndDecode(processed)
            
            # 3. 时间平滑算法 (你的核心逻辑)
            current_qrs = {}
            if decoded_data:
                for i, content in enumerate(decoded_data):
                    if content:
                        current_qrs[content] = points[i] if i < len(points) else None

            # 更新缓冲池
            for content in list(self.qr_buffer.keys()):
                if content in current_qrs:
                    self.qr_buffer[content]['count'] = min(self.qr_buffer[content]['count'] + 1, self.max_count)
                    self.qr_buffer[content]['points'] = current_qrs[content]
                    self.qr_buffer[content]['last_seen'] = self.frame_count
                else:
                    self.qr_buffer[content]['count'] -= 1

            for content, pts in current_qrs.items():
                if content not in self.qr_buffer:
                    self.qr_buffer[content] = {'count': 1, 'points': pts, 'last_seen': self.frame_count}

            # 清理超时码
            to_remove = [c for c, d in self.qr_buffer.items() if d['count'] <= -self.drop_thresh]
            for c in to_remove:
                del self.qr_buffer[c]

            # 4. 提取稳定码并过滤有效信息
            stable = [(c, d) for c, d in self.qr_buffer.items() if d['count'] >= self.confirm_thresh]
            
            for content, data in stable:
                # 只将符合比赛 "颜色,专属编码,核心编码" 格式的数据发给单片机
                if self.is_valid_format(content):
                    if not self.q_serial.full():
                        self.q_serial.put(content)
                        # 如果只需要发送一次，这里可以加一段逻辑防止疯狂重复发送

            # 5. 画面渲染 (仅在 DEBUG 模式下执行)
            if DEBUG_MODE:
                # 简单画个绿框，确认识别位置，舍弃 PIL 中文渲染
                for content, data in stable:
                    if data['points'] is not None:
                        import numpy as np
                        pts = np.array(data['points'], dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
                        # 用原生 cv2 画一段英文提示
                        cv2.putText(frame, "TARGET LOCKED", (pts[0][0][0], pts[0][0][1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Vision Node (DEBUG)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break

        # 优雅退出
        self.cap.release()
        cv2.destroyAllWindows()
        print("程序已退出。")

if __name__ == "__main__":
    node = CompetitionVisionNode()
    try:
        node.run()
    except KeyboardInterrupt:
        node.running = False
        print("\n被用户中断。")