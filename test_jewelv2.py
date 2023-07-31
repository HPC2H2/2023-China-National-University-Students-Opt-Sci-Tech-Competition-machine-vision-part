import cv2
from dominoes_operation import find_max_blobs

cap = cv2.VideoCapture(2)
    
def jewel():
    
    lab_red = (0, 148, 0, 255, 194, 255)
    # lab_blue = (0, 0, 0, 255, 128, 134)
    lab_green = (0, 0, 0, 89, 118, 255)
    # lab_yellow = (0, 118, 166, 255, 255, 255)
    
    skip_frames = 5
    count_frames = 0
    while count_frames < skip_frames:
        _, frame = cap.read()
        count_frames += 1
        
    while True:
        _, frame = cap.read()
        # cv2.imshow("", frame)
        # cv2.waitKey(1)
        
        is_red = 0
        is_green = 0
        tup_red_center = find_max_blobs(frame, lab_red)
        if tup_red_center:
            is_red = 1
        tup_green_center = find_max_blobs(frame, lab_green)
        if tup_green_center:
            is_green = 1
        if is_red:
            if is_green:
                return 4
            else:
                return 3
        else:
            if is_green:
                return 2
            else:
                return 1

# print(jewel())

