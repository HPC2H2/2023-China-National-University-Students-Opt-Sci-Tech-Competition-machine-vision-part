import cv2
import numpy as np

def draw_bounding_rect_and_put_on_info(cnt, img):
    """画出定位框。"""
    # 色块的外接矩形颜色
    rectangle_color = (0, 255, 0)  # 绿色
    
    # 轮廓的边界矩形
    x, y, w, h = cv2.boundingRect(cnt)
    
    # 绘制外接矩形
    cv2.rectangle(img, (x, y), (x + w, y + h), rectangle_color, 2)
    
    cv2.putText(img, f"Width: {w}, Height: {h}", (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, rectangle_color, 2)
    
def thresholded_prep_for_map_op(map):
    """
    对给定的地图进行预处理，包括转换为灰度图像，应用高斯滤波器进行模糊处理，以及应用自适应阈值处理。

    首先将地图转换为灰度图像。然后应用高斯模糊以减少图像中的噪声。
    接着使用自适应阈值处理方法对模糊后的图像进行二值化，该方法使用高斯窗口为每个像素计算阈值。
    最后，反转阈值处理后的图像。

    Args:
    map (numpy.ndarray): 输入的地图图像，必须为3通道的BGR图像。

    Returns:
    numpy.ndarray: 预处理后的地图图像。该图像是一个二值图像，其中原始地图的结构为白色，背景为黑色。
    """
    gray = cv2.cvtColor(map,cv2.COLOR_BGR2GRAY)
    guassian = cv2.GaussianBlur(gray,(5,5),0)
    
    thresholded = cv2.adaptiveThreshold(guassian, 255, 
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
    
    thresholded = cv2.bitwise_not(thresholded)    
    
    return thresholded

def is_contour_square(contour):
    """
    判断给定的轮廓是否为正方形。

    通过比较轮廓的拟合多边形顶点数以及其宽度和高度的相对差异其是否为正方形。

    Args:
    contour (numpy.ndarray): 需要进行判断的轮廓

    Returns:
    bool: 如果轮廓是正方形，则返回True；否则，返回False
    """
    # 判断轮廓的多边形拟合边数是否为4
    peri = 0.02*cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour, peri, True)
    if len(approx) != 4:
        return False

    # 判断宽高比
    rect = cv2.minAreaRect(approx)
    w, h = rect[1]

    if 0.8 <= w / h <= 1.2:
        return True

    return False

def calculate_contour_centroid(contour):
    """计算给定轮廓的质心坐标。"""
    moments = cv2.moments(contour)
    centroid_x = int(moments["m10"] / moments["m00"])
    centroid_y = int(moments["m01"] / moments["m00"])
    return centroid_x, centroid_y

def is_point_inside_contour(contour, point):
    """判断点是否在给定轮廓内部。"""
    return cv2.pointPolygonTest(contour, point, False) > 0

def distance(point1, point2):
    """计算两点间欧式距离。"""
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) 

def get_outer_squares(thresholded):
    """
    从二值图中返回包裹着小正方形的外部正方形(定位框)。

    对于图像中的轮廓,通过面积和是否为正方形的条件筛选。
    符合条件的轮廓被留下,两两计算质心并判断是否被彼此的轮廓包含。
    如果包含,再比较两质心间的距离,若小于5,则轮廓面积大者为外部正方形。

    对于已被选择的外部正方形,两两判断他们的质心距离,如果在15以内,则去除面积小的正方形。
    
    Args:
    thresholded : 预处理后的摄像头缓冲帧，为二值图像

    Returns:
    outer_squares : 表示外部正方形的轮廓列表
    """

    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    outer_squares = []
    centroids = []
    areas = []
    for i, cnt1 in enumerate(contours):
        area1 = cv2.contourArea(cnt1)
        if 20 <= area1 <= 6000 and is_contour_square(cnt1):
            centroid1 = calculate_contour_centroid(cnt1)
            for j, cnt2 in enumerate(contours):
                if i < j:
                    area_2 = cv2.contourArea(cnt2)
                    if 20 <= area_2 <= 6000 and is_contour_square(cnt2):
                        centroid2 = calculate_contour_centroid(cnt2)
                        if is_point_inside_contour(cnt1, centroid2) and \
                            is_point_inside_contour(cnt2, centroid1):
                            if distance(centroid1, centroid2) < 15:
                                centroids.append(centroid1)
                                areas.append(area1)
                                outer_squares.append(cnt1)
                                break
    
    
    vis = [False]*len(outer_squares)
    for i,cnt1 in enumerate(outer_squares):
        if vis[i]:
            continue
        for j, cnt2 in enumerate(outer_squares):
                if i < j:
                    if vis[j]:
                        continue
                    if distance(centroids[i],centroids[j]) < 10:
                        if areas[i] > areas[j]:
                            vis[j] = True
                        else:
                            vis[i] = True
    outer_squares = [element for vis, element in zip(vis, outer_squares) if not vis]
    return outer_squares

 
def sort_4_points(pts): 
    """
    对给定的四个点进行排序。

    这个函数将输入的四个点按照以下顺序排序：左上，右上，左下，右下。
    首先，根据y坐标进行排序。
    然后，检查上面两个点和下面两个点的x坐标，
    如果第一个点的x坐标大于第二个点的x坐标，那么就交换它们的位置。

    Args:
        pts (List[List[int, int]]): 输入的四个点，每个点都是一个包含两个整数（x，y坐标）的列表。

    Returns:
        List[List[int, int]]: 排序后的四个点。
    """
    pts.sort(key = lambda x:x[1]) 
    if pts[0][0] > pts[1][0]:
        pts[0],pts[1] = pts[1],pts[0]
    if pts[2][0] > pts[3][0]:
        pts[2],pts[3] = pts[3],pts[2]
    return pts

def get_and_sort_squares_center(outer_squares):
    """
    计算并排序给定正方形的中心点。

    这个函数首先计算输入的每个正方形的中心点，然后使用sort_4_points函数进行排序。中心点是通过OpenCV的cv2.moments函数来计算的。

    Args:
        outer_squares (List[np.array]): 输入的正方形，每个正方形都是由四个点的坐标构成的numpy数组。

    Returns:
        List[List[int, int]]: 排序后的四个中心点。
    """
    center_points = []
    for sq in outer_squares:
        M = cv2.moments(sq)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        center_points.append([center_x, center_y])
    
    return sort_4_points(center_points)

def perform_perspective_transform(img, pts):
    """
    对给定的图像进行透视变换。

    这个函数将一个源点集映射到目标点集，目标点集的位置固定在[(75,75), (725,75), (75,725), (725,725)]。
    首先，构造从源点集到目标点集的透视变换矩阵，然后对图像进行透视变换。最后返回变换后的图像和透视变换矩阵。

    Args:
        img (numpy.ndarray): 需要进行透视变换的源图像。
        pts (List[List[float, float]]): 源图像上的四个点，每个点都是一个包含两个浮点数（x，y坐标）的列表。

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: 一个元组，包含两个元素。第一个元素是透视变换后的图像，第二个元素是透视变换矩阵。
    """
    src = np.array(pts, dtype="float32")
    
    target = [(75,75), (725,75), (75,725), (725,725)]
    dst = np.array(target, dtype="float32")
    
    # 构造从src到dst的仿射矩阵
    M = cv2.getPerspectiveTransform(src, dst)
    # 完成从src到dst的透视变换
    warped = cv2.warpPerspective(img, M,(800, 800))
    # 返回透视变换的结果
    return warped

def transform_treasure_coordinates(circle): 
    """
    用于将原始坐标转换成相对坐标的函数。

    Args:
        circle (tuple): 包含圆心的x, y坐标和半径r的元组。

    Returns:
        tuple: 包含新的x和y坐标的元组。
    """
    x, y, r = circle
    new_x = round((x - 125)/50) - 1
    new_y = 9 - (round((y - 125)/50) - 1)

    return new_x, new_y

def pick_valid_point(pts):
    """筛选出坐标值在合理范围内的藏宝点。"""
    pts = [(x, y) for (x, y) in pts if 0 <= x <= 9 and 0 <= y <= 9]
    return pts
            
def find_tresuares_and_transform_coordinates(warped):
    """
    用于在图片中找到并转换藏宝点坐标的函数。

    Args:
        warped (np.ndarray): 输入的图像。

    Returns:
        list: 包含新的x和y坐标的列表，如果未找到则返回None。
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # 非藏宝图区域不进行圆检测
    gray[:125, :] = 0
    gray[675:, :] = 0
    gray[:, :125] = 0
    gray[:, 675:] = 0
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=8, param1=50,
                               param2=22, minRadius=8, 
                               maxRadius=18)

    if circles is None or len(circles) == 0:
            print("No circle found.")
    else:
        circles = circles[0]
        if len(circles) > 8: 
            print("Too much circles.")
        if len(circles) < 8:
            print("Circles are not enough.",str(len(circles)))
        
        treasure_coordinates = []
        # 循环检测到的圆并绘制出来
        for circle in circles:
            new_x, new_y = transform_treasure_coordinates(circle)
            x, y, r = circle
            cv2.circle(warped, (x, y), r, (0, 0, 255), 6)
            cv2.putText(warped, f"({new_x},{new_y})", (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            treasure_coordinates.append((new_x,new_y))
        cv2.imshow("Circles",warped)
        if len(circles) == 8:
            print("Find 8 circles but check the image.")
            
        pts = pick_valid_point(treasure_coordinates)
        cv2.waitKey(0)
        if len(pts) == 8:
            return pts
        else:
            return None
    
def map_opreation(img):
    """
    图像处理主流程，用于找到并显示藏宝图的坐标。while True:
    _, frame = cap.read()
    print(map_opreation(frame))

    Args:
        img (np.ndarray): 输入的图像。

    Returns:
        list: 包含两个元素，分别是宝藏x和y坐标的列表。如果图像中未找到四个定位框则返回None。
    """
    thresholded = thresholded_prep_for_map_op(img)
    cv2.imshow('Local Camera', img)
    cv2.imshow('Thresholded', thresholded)
    cv2.waitKey(1)
    outer_squares =  get_outer_squares(thresholded)

    for cnt in outer_squares:
        draw_bounding_rect_and_put_on_info(cnt, img)
    
    if len(outer_squares) != 4:
        return None
    for cnt in outer_squares:
        draw_bounding_rect_and_put_on_info(cnt, img)
    cv2.imshow('Outer Squares', img)
    sorted_points = get_and_sort_squares_center(outer_squares)
    warped = perform_perspective_transform(img, sorted_points)
    
    cv2.imshow('Warped', warped)
    return find_tresuares_and_transform_coordinates(warped)  

# 测试代码
# img = cv2.imread('map.jpg')
# print(map_opreation(img))  

# cap = cv2.VideoCapture(2)
# while True:
#     _, frame = cap.read()
#     print(map_opreation(frame))

