import numpy as np
import cv2
import pyautogui

def mode_pps(a):
    return abs((abs(a[1][0] - a[0][0]) + abs(a[3][1] - a[0][1])) / 2)

#Filters all horizontal and vertical lines out of image - then combines the 2
def mk_lines_img(img):
    #Filter all edges out of image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200, apertureSize = 3)
    gaus_edges = cv2.GaussianBlur(edges, (3,1), 0)

    #Get all horizontal edges from filtered result
    horiz_img = np.copy(gaus_edges)
    cols = horiz_img.shape[1]
    horizontal_size = cols // 45
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horiz_img, horizontalStructure) #add stroke size to edges
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    #Get all vertical edges from filtered result
    vert_img = np.copy(gaus_edges)
    cols = vert_img.shape[0]
    vertical_size = cols // 45
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vert_img, verticalStructure) #add stroke size to edges
    vertical = cv2.dilate(vertical, verticalStructure)

    #Combine vertical and horizontal edges into one
    res = np.maximum(horizontal, vertical)

    return res

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))

def find_squares(lines_img):
    sqrs = []
    #Split lines image into separate single-channel arrays
    for gray in cv2.split(lines_img):
        for thrs in range(0, 255, 26): # Use many thresholds to find all possible contours in IMG
            if thrs == 0:
                binary = cv2.Canny(gray, 0, 50, apertureSize=5)
                binary = cv2.dilate(binary, None)
            else:
                _retval, binary = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)

            contours, _hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            #Verify contour and append to sqrs if valid
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < 0.1:
                        sqrs.append(cnt)
    return sqrs

#Return coords of leftmost/rightmost or upmost/downmost squares in sqr list
def mk_coord(all_squares, num1, num2, reverse=False):
    all_squares_loc = sorted(all_squares, key=lambda x:x[num1][num2], reverse=reverse)

    count = 0
    for sqr in all_squares_loc:
        coord = sqr[num1][num2]

        count = 0
        for sqr2 in all_squares_loc:
            if sqr2[num1][num2] - 2 < coord and sqr2[num1][num2] + 2 > coord:
                count += 1
            if count >= 2:
                count = True
                break
        if bool(count):
            return coord

def locate(img):
    dimy, dimx = 0, 100
    #Verify located chessboard is square
    while abs(dimy - dimx) > 30:
        #Extract all vertical/horizontal lines from image
        lines_img = mk_lines_img(img)

        #Get all coords of square corners in image
        squares = find_squares(lines_img)
        unfiltered_squares_list = [sql_item.tolist() for sql_item in squares if abs((sql_item.tolist()[0][0] - sql_item.tolist()[3][0]) - (sql_item.tolist()[0][1] - sql_item.tolist()[1][1])) < 10]

        sqr_w_lst = [mode_pps(sqr) for sqr in unfiltered_squares_list]
        w_mode = max(set(sqr_w_lst), key=sqr_w_lst.count)

        #Filter through all squares and get only coords of the main 64
        all_squares = []
        for square_item in unfiltered_squares_list:
            pps_of_sqr = mode_pps(square_item)
            if pps_of_sqr > w_mode - 3 and pps_of_sqr < w_mode + 3:
                all_squares.append(square_item)

        #Make coordinates of board and find dimensions of board
        x = mk_coord(all_squares, 0, 0) - 3
        x2 = mk_coord(all_squares, 1, 0, True) + 3
        y = mk_coord(all_squares, 0, 1) - 3
        y2 = mk_coord(all_squares, 3, 1, True) + 3
        dimx = x2 - x + 4
        dimy = y2 - y + 4

    return x, x2, y, y2