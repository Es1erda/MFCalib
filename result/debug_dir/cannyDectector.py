import cv2
import numpy as np
import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 过滤轮廓，删除长度小于阈值的轮廓，以及那些首尾距离与轮廓长度比值过小的轮廓
def contour_filter(contour, len_threshold=400):
    invalid_cam = 0
    for i in range(len(contour) - invalid_cam):
        dist = np.size(contour[i - invalid_cam])  # 轮廓长度
        end_dist = np.sum(np.abs(contour[i - invalid_cam][0, 0, :] - contour[i - invalid_cam][-1, 0, :]))
        if (dist < len_threshold and 1.5 * end_dist < dist) or (dist < len_threshold / 2):
            contour = contour[:(i - invalid_cam)] + contour[(i - invalid_cam) + 1:]
            invalid_cam += 1
        if i - invalid_cam == len(contour) - 2:
            break
    return contour
    
# 定义边缘检测函数
def edge_detector(canny_threshold, edge_threshold, src_img):
    # 高斯模糊参数
    gaussian_size = 5
    
    # 对输入图像进行高斯模糊
    src_img = cv2.GaussianBlur(src_img, (gaussian_size, gaussian_size), 0, 0)
    
    # 使用Canny方法提取边缘
    canny_result = cv2.Canny(src_img, canny_threshold, canny_threshold * 2, 3, L2gradient=1)
    
    # 寻找轮廓
    major_version = int(cv2.__version__.split('.')[0])
    if major_version >= 4:
        contours, hierarchy = cv2.findContours(canny_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, hierarchy = cv2.findContours(canny_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 使用contour_filter函数过滤轮廓
    contours = contour_filter(contours)
    
    # 获取输入图像的尺寸
    height, width = src_img.shape[:2]
    
    # 创建空白边缘图像
    edge_img = np.zeros((height, width), dtype=np.uint8)

    # 遍历轮廓
    for contour in contours:
        if len(contour) > edge_threshold:
            for point in contour:
                x, y = point[0,0], point[0,1]
                edge_img[y, x] = 255

    # 保存结果到脚本所在目录
    cv2.imwrite(os.path.join(script_dir, "canny_result.png"), canny_result)
    cv2.imwrite(os.path.join(script_dir, "edge_img.png"), edge_img)
    return edge_img

if __name__ == "__main__":
    # 载入图像，使用完整路径
    input_path = os.path.join(script_dir, "intensity_img.png")
    src_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if src_img is None:
        print(f"Error: Cannot read input image from {input_path}")
        exit(1)
        
    # 设置参数
    canny_threshold = 10
    edge_threshold = 100

    # 运行边缘检测（现在只需要3个参数）
    edge_img = edge_detector(canny_threshold, edge_threshold, src_img)
