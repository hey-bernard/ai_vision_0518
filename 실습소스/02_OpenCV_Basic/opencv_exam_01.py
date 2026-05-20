# opencv_exam_01.py
# 이미지 변환 처리 파이썬 모듈 

import cv2
import numpy as np

def cvt_color(img):   # 색상변경
    # 색상 변경 : gray
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray_image.jpg',gray_img)
    cv2.imshow("gray_image",gray_img)
    print(cv2.COLOR_BGR2GRAY) # 6

    # 색상 변경 : hsv, hue(색조),saturation(채도),Value(명도)
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    cv2.imwrite('hsv_image.jpg',hsv_img)
    cv2.imshow("hsv_image",hsv_img)
    print(cv2.COLOR_BGR2HSV) # 40

    # 색상 변경 : YUV = YCbCr
    yuv_img = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    cv2.imwrite('yuv_image.jpg',yuv_img)
    cv2.imshow("yuv_image",yuv_img)
    print(cv2.COLOR_BGR2YUV) # 82

# 크기 조정 : cv2.resize()
def resize(img):
    # img = cv2.imread('image.jpg')  # 512*512*3,(512,512,3)
    r,c = img.shape[:2]
    print('r:',r,'c:',c) # r: 512 c: 512

    # r,c,n = img.shape[:3]
    # print('r:',r,'c:',c,'n:',n) # r: 512 c: 512 n : 3

    # 이미지 2배 확대
    new_image = cv2.resize(img,(2*r,2*c),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('resize2_image.jpg', new_image)
    cv2.imshow("resize_image", new_image)

    # # 이미지 1/2 축소
    new_image = cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
    cv2.imwrite('resize1_2_image.jpg', new_image)
    cv2.imshow("resize1_2_image", new_image)

    # 확대(1<fx) or 축소(1>fx) 모두 가능
    new_image = cv2.resize(img,None,fx=0.1,fy=0.1) # cv2.INTER_LINEAR
    cv2.imwrite('resize_linear_image.jpg', new_image)
    cv2.imshow("resize_linear_image", new_image)


# 이미지 자르기 : crop
def crop(img):
    # img = cv2.imread('image.jpg')  # 512*512*3,(512,512,3)
    img_crop = img[0:200,150:350]
    # cv2.imwrite('crop_image.jpg', img_crop)
    cv2.imshow("crop_image", img_crop)

# 이동
def tranlsation(img):
    # img = cv2.imread('image.jpg')  # 512*512*3,(512,512,3)
    r,c = img.shape[:2]
    M = np.float32([[1,0,100],[0,1,100]])  # (2,3) Matrix
    new_img = cv2.warpAffine(img,M,(c,r))
    cv2.imshow("translation_image", new_img)

# 회전
def rotation(img):
    # img = cv2.imread('image.jpg')  # 512*512*3,(512,512,3)
    r,c = img.shape[:2]
    M = cv2.getRotationMatrix2D((c/2,r/2),90,1) # 중심,회전각(CCW),크기비율
    new_img = cv2.warpAffine(img,M,(c,r))
    cv2.imshow("rotation_image", new_img)

# 상하 반전
def vertical_flop(img):
    # img = cv2.imread('image.jpg')  # 512*512*3,(512,512,3)
    vertical_flipped_img = cv2.flip(img, 0)
    cv2.imshow("Flipped Image (Vertical Flip)", vertical_flipped_img)

# 좌우 반전
def horizontal_flop(img):
    # img = cv2.imread('image.jpg')  # 512*512*3,(512,512,3)
    horiz_flipped_img = cv2.flip(img, 1)
    cv2.imshow("Flipped Image (Horizontal Flip)", horiz_flipped_img)

# 임계값 처리 : threshold
def threshold(img):
    # img = cv2.imread('image.jpg')  # 512*512*3,(512,512,3)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_image = cv2.threshold(gray_img,120,255,cv2.THRESH_BINARY)
    # 120 보다 큰값은 흰색으로 그외에는 검은색으로 출력
    # 값이 작을수록 흰색이 많이 출력
    # 값이 클수록 검은색이 많이 출력
    cv2.imshow("threshold_image", new_image[1])

# 필터 : filter,합성곱
def filter(img):
    # img = cv2.imread('image.jpg')  # 512*512*3,(512,512,3)
    ker = np.array([[1,1,1],[1,1,1],[1,1,1]]) # (3,3)

    new_img = cv2.filter2D(img,-1,ker) # -1
    cv2.imshow("threshold_image", new_img)

# 가우시안 블러
def GaussianBlur(img):
    # img = cv2.imread('image.jpg')  # 512*512*3,(512,512,3)
    # cv2.imshow("original_image", img)
    new_img = cv2.GaussianBlur(img,(5,5),0) # (5,5) : 반드시 홀수,클수록 흐림
    cv2.imshow("gaussian_image", new_img)

# 중간값(median) 블러:
def medianBlur(img):
    # img = cv2.imread('image.jpg')  # 512*512*3,(512,512,3)
    # cv2.imshow("original_image", img)
    new_img = cv2.medianBlur(img,11) #  5:반드시 홀수,클수록 흐림
    cv2.imshow("median_image", new_img)

# 침식: erosion
def erosion(img):
# 두 물체를  축소시켜서 두 물체의 차이가 명확해지게 하거나, 노이즈를 제거
#     img = cv2.imread('image.jpg')  # 512*512*3,(512,512,3)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("original_image", gray_img)
    ker = np.ones((5,5),np.uint8)
    new_img = cv2.erode(gray_img,ker,iterations = 1)
    cv2.imshow("erosion_image", new_img)

# 팽창 : diation
def dilation(img):
# 작은부분을 확대하려는 경우, 원하지 않는 틈/구멍을 메우고 싶을때
#     img = cv2.imread('image.jpg')  # 512*512*3,(512,512,3)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("original_image", gray_img)
    ker = np.ones((5, 5), np.uint8)
    new_img = cv2.dilate(gray_img, ker, iterations=1)
    cv2.imshow("dilation_image", new_img)

# 소벨 에지검출 : sobel
def sobel(img):
    # img = cv2.imread('image.jpg')  # 512*512*3,(512,512,3)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("original_image", gray_img)
    x_edges = cv2.Sobel(gray_img,-1,1,0,ksize = 5) # 1,0 : x축방향
    y_edges = cv2.Sobel(gray_img,-1,0,1,ksize = 5) # 0,1 : y축방향
    cv2.imshow("x_edges_image", x_edges)
    cv2.imshow("y_edges_image", y_edges)

# 캐니 에지검출: canny
def canny(img):
    # img = cv2.imread('image.jpg')  # 512*512*3,(512,512,3)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("original_image", gray_img)
    edges = cv2.Canny(gray_img, 100,200,5) # min,max,aperture_size(=kernel_size)
    cv2.imshow("canny_image", edges)

# 윤곽선 검출 : contour
def contour(img):
    # img = cv2.imread('image.jpg')  # 512*512*3,(512,512,3)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_image = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh_image[1],cv2.RETR_TREE,\
                                           cv2.CHAIN_APPROX_SIMPLE  )
    cv2.drawContours(img,contours,-1,(255,0,0),3) # (B,G,R):윤곽선색상 ,  3: thickness
    cv2.imshow("contour_image", img)

# 템플릿 매칭
def template_match(img):
    # img = cv2.imread("image.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_temp = cv2.imread("template.jpg")
    gray_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
    w, h = gray_temp.shape[::-1] # BGR --> RGB  , w:187, h: 193
    output = cv2.matchTemplate(gray, gray_temp, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(output)
    top = max_loc
    bottom =(top[0] + w, top[1] + h)
    cv2.rectangle(img, top, bottom, 255, 2) # 255: Blue, 2:thickness
    cv2.imshow("image", img)
