import cv2
import numpy as np

def find_max_blobs(frame, lab_color):
    """
    在图像帧中寻找最大的彩色块并返回它的中心点和轮廓。
    
    参数：
    frame: ndarray，BGR颜色空间的图像帧
    lab_color: tuple，LAB颜色空间的色彩范围
    
    返回：
    center_x, center_y(, cnt): 图像帧中最大彩色块的中心坐标(和轮廓)
    如果没有找到符合条件的彩色块，则返回None。
    """
    Lmin, Amin, Bmin, Lmax, Amax, Bmax = lab_color
    
    # 将帧转换为LAB颜色空间
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # 创建掩码，根据预定义的LAB颜色范围过滤帧中的颜色
    mask = cv2.inRange(lab_frame, (Lmin, Amin, Bmin), (Lmax, Amax, Bmax))

    # 执行形态学操作，如腐蚀和膨胀，以消除噪声和平滑掩码
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 查找掩码中的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < max_area and area > 50:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 计算外接矩形的中心坐标
        center_x = x + int(w / 2)
        center_y = y + int(h / 2)

        max_area = area
        max_cnt = cnt
    if max_area == 0:
        return None
    
    # 判断宽高比
    peri = 0.02*cv2.arcLength(max_cnt,True)
    approx = cv2.approxPolyDP(max_cnt, peri, True)

    rect = cv2.minAreaRect(approx)
    w, h = rect[1]

    if h == 0:
        return None
    if w / h >= 3 or w / h <= 0.3:
        return None
    
    return center_x, center_y
    # return center_x, center_y, max_cnt

# def draw_bounding_rect_and_put_on_info(cnt, img, center_point):
#     """
#     在给定图像上绘制给定轮廓的边界矩形，并添加有关其位置和大小的信息。

#     参数：
#     cnt：array，轮廓。
#     img：ndarray，要在其上绘制的图像。
#     center_point：tuple，矩形的中心点。
#     """
    
#     # 色块的外接矩形颜色
#     rectangle_color = (0, 255, 0)  # 绿色
    
#     center_x, center_y = center_point
    
#     # 轮廓的边界矩形
#     x, y, w, h = cv2.boundingRect(cnt)
    
#     # 绘制外接矩形
#     cv2.rectangle(img, (x, y), (x + w, y + h), rectangle_color, 2)
    
#     # 在外接矩形上绘制中心点
#     cv2.circle(img, center_point, 2, rectangle_color, 2)

#     # 显示外接矩形的中心坐标和宽高
#     cv2.putText(img, f"Center: ({center_x}, {center_y})", (x, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, rectangle_color, 2)
#     cv2.putText(img, f"Width: {w}, Height: {h}", (x, y + h + 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, rectangle_color, 2)
    
#     cv2.imshow("Target",img)
    
def find_treasure(frame, lab_domino, lab_pattern, id):
    """
    在给定的帧中找到宝藏(骨牌)。

    参数：
    frame: ndarray，BGR颜色空间的图像帧
    lab_domino: tuple，骨牌的LAB颜色空间的色彩范围
    lab_pattern: tuple，图案的LAB颜色空间的色彩范围
    
    返回：
    id: 找到骨牌类型标识
    如果骨牌和图案的中心点距离大于200，或者没有找到骨牌或图案，则返回None。
    """
    tup1 = find_max_blobs(frame, lab_domino) # tup[0:2]存储色块中心点，tup[2]是轮廓
    if not tup1:
        return None
    
    tup2 = find_max_blobs(frame, lab_pattern)
    if not tup2:
        return None

    domino_center = tup1[0:2]
    pattern_center = tup2[0:2]
    # 通过图案和骨牌中心是否接近判断是否成功识别
    center_dist = np.sqrt((domino_center[0] - pattern_center[0])**2
                            +(domino_center[1] - pattern_center[1])**2)
    if center_dist > 200:
        return None
    
    # draw_bounding_rect_and_put_on_info(tup1[2], frame, pattern_center)
    return id

