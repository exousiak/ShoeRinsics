import cv2, os
import numpy as np

"""
Returns True if image is an RGB image and False otherwise.
"""
def is_RGB(image):
    if len(image.shape) == 3:
        return True
    else:
        return False

"""
Read an image given the image_path. If image is a mask, it is eroded to remove edges from consideration.
"""
# def read_image(image_path, is_mask=False, gray=False, verbose=True): #, scale=1, , LAB=False, from_cum=None, to_cum=None, gamma_correct=False, gamma_reverse=False):
#     image = cv2.imread(image_path)
#     if image is None:
#         extensions = [".png", ".PNG"]
#         for extension in extensions:
#             image = cv2.imread(os.path.splitext(image_path)[0] + extension)
#             if image is not None:
#                 break
#     if image is None:
#         if verbose:
#             print("No image found at " + image_path)
#         return None
#     # if scale != 1:
#     #     image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

#     if gray:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
#     # elif gamma_correct:
#     #     image = (np.power(image/255.0, 1/2.2)*255).astype(np.uint8)
#     # elif gamma_reverse:
#     #     image = (np.power(image/255.0, 2.2)*255).astype(np.uint8)

#     # if from_cum is not None and to_cum is not None:
#     #     image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     #     image[:,:,0] = to_cum[(from_cum[image[:,:,0]]*255).astype(np.uint8)]*255
#     #     image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
#     #
#     # if LAB:
#     #     image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

#     if image.dtype == np.uint8:
#         image = image.astype(np.float32) / 255.0

#     if is_mask:
#         image = image[..., 0:1]
#         image = image > np.max(image) / 2
#     return image


def read_image(image_path, is_mask=False, gray=False, verbose=True, resize_shape=False, resize_ratio = True
, compress_quality=95):
    image = cv2.imread(image_path)
    if image is None:
        extensions = [".png", ".PNG"]
        for extension in extensions:
            image = cv2.imread(os.path.splitext(image_path)[0] + extension)
            if image is not None:
                break
    if image is None:
        if verbose:
            print("No image found at " + image_path)
        return None
    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=2)
 
    # 가로 300 기준으로 비율 계산 
    # 소수점 차이로 인해 오류 발생(ex-> image = 718, mask = 717)
    # mask를 image 해상도와 같도록 조정 
    # 오류 발생 x, 신발 원본 이미지의 비율 유지 
    # -> resize_ratio = True
    if resize_ratio:
        resize_width = 300
        height = int(np.round((image.shape[0] * resize_width / image.shape[1])))
        image = cv2.resize(image, (resize_width, height), interpolation=cv2.INTER_AREA)

    # 성인 남자 신발(270) 기준 가로 세로 비율 1:2.5 (데이터셋 평균 비율 1:2.51)
    # -> 288 x 720으로 임의 선정 
    # 오류 X But 실제 이미지 비율과 다르기 때문에 후에 예측 과정에서 문제 발생 가능성 있음 
    # -> resize_shape = False 
    if resize_shape:
        resize_width = 288
        resize_height = 720
        image = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    # 마스크 해상도 변경 후 처리 예정
    # -> is_mask = False 
    if is_mask:
        image = image[..., 0:1]
        image = image > np.max(image) / 2
        


    return image


"""
Given an image in HxWxC format, change it to CxHxW format so that it can be fed into a neural network.
"""
def image_to_channels(image):
    return image.transpose(2, 0, 1).astype(np.float32)


