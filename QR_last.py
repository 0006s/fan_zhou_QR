import cv2
import numpy as np
import time
import os
from collections import defaultdict


class MultiQRScanner:
    """
    多码同时识别扫描器
    基于 WeChatQRCode + 时间平滑，支持画面中同时识别多个二维码
    """

    def __init__(self, cam_index=0, model_dir=None):
        # 默认模型路径
        if model_dir is None:
            model_dir = os.path.join(
                os.path.expanduser("~"),
                ".qclaw", "workspace", "wechat_qrcode_models"
            )

        # 1. 初始化摄像头
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 (Index: {cam_index})，请检查硬件连接！")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 2. 初始化 WeChatQRCode
        pbtxt = os.path.join(model_dir, "detect.prototxt")
        caffemodel = os.path.join(model_dir, "detect.caffemodel")
        sr_pbtxt = os.path.join(model_dir, "sr.prototxt")
        sr_caffemodel = os.path.join(model_dir, "sr.caffemodel")

        if not all(os.path.exists(f) for f in [pbtxt, caffemodel, sr_pbtxt, sr_caffemodel]):
            missing = [f for f in [pbtxt, caffemodel, sr_pbtxt, sr_caffemodel] if not os.path.exists(f)]
            raise FileNotFoundError(f"缺少模型文件: {missing}")

        self.detector = cv2.wechat_qrcode.WeChatQRCode(pbtxt, caffemodel, sr_pbtxt, sr_caffemodel)
        print(f"WeChatQRCode 初始化成功（模型目录: {model_dir}）")

        # 3. 时间平滑参数
        self.qr_buffer = defaultdict(lambda: {'count': 0, 'points': None, 'last_seen': 0})
        self.confirm_thresh = 3      # 连续检测到3帧才确认
        self.drop_thresh = 5         # 连续丢失5帧才丢弃
        self.max_count = 10          # 计数上限
        self.frame_count = 0

        # 4. 图像预处理
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 5. PIL 中文绘制支持（加载字体，Image/ImageDraw 用到时再导入）
        self.use_pil = False
        self.font_large = None
        self.font_small = None
        try:
            from PIL import ImageFont
            font_path = "C:\\Windows\\Fonts\\simhei.ttf"
            if os.path.exists(font_path):
                self.font_large = ImageFont.truetype(font_path, 28)
                self.font_small = ImageFont.truetype(font_path, 18)
                self.use_pil = True
                print("已开启中文显示支持")
        except ImportError:
            print("未检测到 PIL，执行: pip install pillow")

    def preprocess_frame(self, frame):
        """CLAHE 对比度增强，提升模糊/低光照二维码识别率"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def draw_qr_overlay(self, frame, stable, pending):
        """
        在帧上绘制所有二维码相关标注（边框 + 文字）。
        内部使用单次 PIL 转换批量绘制所有文字（避免每段文字各自转换导致卡顿）。

        参数:
            frame:      BGR numpy 数组（原始摄像头帧）
            stable:     已确认的码列表 [(content, data), ...]
            pending:    确认中的码列表 [(content, data), ...]

        返回:
            绘制完成后的 BGR numpy 数组
        """
        h, w = frame.shape[:2]
        texts = []          # 统一收集所有文字：(text, (x,y), (b,g,r), font_size)
        pending_pts = []     # pending 边框点
        stable_pts = []     # stable 边框点

        # ---------- 收集 pending 码 ----------
        for content, data in pending:
            if data['points'] is not None:
                pts = np.array(data['points'], dtype=np.int32).reshape((-1, 1, 2))
                pending_pts.append(pts)
                cx = int(pts[0][0][0])
                cy = int(pts[0][0][1])
                texts.append(("?", (cx, cy - 24), (0, 255, 255), "small"))

        # ---------- 收集 stable 码 ----------
        for i, (content, data) in enumerate(stable):
            if data['points'] is not None:
                pts = np.array(data['points'], dtype=np.int32).reshape((-1, 1, 2))
                stable_pts.append(pts)

                # 左上角序号标签
                cx = int(pts[0][0][0])
                cy = int(pts[0][0][1])
                texts.append((f"#{i+1}", (cx, cy - 24), (0, 0, 0), "small"))

                # 中心内容（截断过长文本）
                cx_center = int(pts[:, 0, 0].mean())
                cy_center = int(pts[:, 0, 1].mean())
                display_text = content if len(content) <= 12 else content[:11] + ".."
                texts.append((display_text, (cx_center - 30, cy_center - 10),
                              (255, 255, 255), "small"))

        # ---------- 底部状态栏 ----------
        roi_y = h - 90
        roi = frame[roi_y:h, 0:w]
        black = np.zeros_like(roi)
        frame[roi_y:h, 0:w] = cv2.addWeighted(roi, 0.6, black, 0.4, 0)

        texts.append((f"已识别: {len(stable)} 个码", (20, roi_y + 5), (0, 255, 0), "large"))
        if stable:
            display = " | ".join([f"#{i+1}:{c[:10]}" for i, (c, _) in enumerate(stable[:5])])
            texts.append((display, (20, roi_y + 38), (255, 255, 255), "large"))
        texts.append((f"缓冲:{len(self.qr_buffer)} 帧:{self.frame_count}",
                      (w - 280, roi_y + 5), (128, 128, 128), "large"))

        # ---------- 统一绘制 ----------
        if self.use_pil:
            # PIL 模式：cv2 画边框 → 单次 PIL 转换 → 批量文字 → cv2 转回
            from PIL import Image, ImageDraw

            for pts in pending_pts:
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
            for pts in stable_pts:
                cv2.polylines(frame, [pts], True, (0, 255, 0), 3)

            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            for text, pos, color, size in texts:
                font = self.font_large if size == "large" else self.font_small
                # cv2 BGR → PIL RGB
                pil_color = (color[2], color[1], color[0])
                draw.text(pos, text, font=font, fill=pil_color)

            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        else:
            # 无 PIL 降级：cv2 画边框 + cv2 画文字（中文显示为 ?）
            for pts in pending_pts:
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
            for pts in stable_pts:
                cv2.polylines(frame, [pts], True, (0, 255, 0), 3)

            for text, pos, color, size in texts:
                scale = 0.7 if size == "large" else 0.45
                cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
            return frame

    def scan(self):
        print("多码同时识别已启动... 按 'q' 退出")
        print(f"参数: 确认={self.confirm_thresh}帧, 丢弃={self.drop_thresh}帧")
        print("提示: 绿框=已确认 | 黄框=确认中")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1

            # --- 预处理（仅用于识别，不影响显示帧） ---
            processed = self.preprocess_frame(frame)

            # --- WeChatQRCode 多码识别 ---
            (decoded_data, points) = self.detector.detectAndDecode(processed)

            # --- 时间平滑缓冲池更新 ---
            current_qrs = {}
            if decoded_data:
                for i, content in enumerate(decoded_data):
                    if content:
                        current_qrs[content] = points[i] if i < len(points) else None

            for content in list(self.qr_buffer.keys()):
                if content in current_qrs:
                    self.qr_buffer[content]['count'] = min(
                        self.qr_buffer[content]['count'] + 1, self.max_count)
                    self.qr_buffer[content]['points'] = current_qrs[content]
                    self.qr_buffer[content]['last_seen'] = self.frame_count
                else:
                    self.qr_buffer[content]['count'] -= 1

            for content, pts in current_qrs.items():
                if content not in self.qr_buffer:
                    self.qr_buffer[content] = {
                        'count': 1, 'points': pts, 'last_seen': self.frame_count
                    }

            # 清理超时码
            to_remove = [c for c, d in self.qr_buffer.items() if d['count'] <= -self.drop_thresh]
            for c in to_remove:
                del self.qr_buffer[c]

            # --- 分离稳定/确认中的码 ---
            stable = [(c, d) for c, d in self.qr_buffer.items() if d['count'] >= self.confirm_thresh]
            pending = [(c, d) for c, d in self.qr_buffer.items()
                       if 0 < d['count'] < self.confirm_thresh]

            # --- 统一绘制：边框 + 全部文字（单次 PIL 转换） ---
            frame = self.draw_qr_overlay(frame, stable, pending)

            cv2.imshow("Multi-QR Scanner", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        scanner = MultiQRScanner(cam_index=1)
        scanner.scan()
    except Exception as e:
        print(f"程序出错: {e}")
