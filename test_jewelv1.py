import cv2
from dominoes_operation import find_treasure

cap = cv2.VideoCapture(2)
    
def jewel():
    
    lab_red = (0, 148, 0, 255, 194, 255)
    lab_blue = (0, 0, 0, 255, 128, 134)
    lab_green = (0, 0, 0, 89, 118, 255)
    lab_yellow = (0, 118, 166, 255, 255, 255)
    
    skip_frames = 5
    count_frames = 0
    while count_frames < skip_frames:
        _, frame = cap.read()
        count_frames += 1
        
    while True:
        _, frame = cap.read()
        # cv2.imshow("", frame)
        # cv2.waitKey(1)
        
        # 1:蓝真;2:蓝假;3:红假;4:红真
        id = find_treasure(frame, lab_blue, lab_yellow, 1)
        if id: return id
        id = find_treasure(frame, lab_blue, lab_green, 2)
        if id: return id
        id = find_treasure(frame, lab_red, lab_yellow, 3)
        if id: return id
        id = find_treasure(frame, lab_red, lab_green, 4)
        if id: return id

# print(jewel())

