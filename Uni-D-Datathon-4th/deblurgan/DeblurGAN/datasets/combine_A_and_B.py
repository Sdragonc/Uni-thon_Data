import os
import numpy as np
import cv2
import argparse

# 인자 설정
parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for clean images', type=str, default='Training/clean')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for noisy images', type=str, default='Training/noisy')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory for combined images', type=str, default='combined')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
args = parser.parse_args()

# 입력 디렉토리 설정
img_fold_A = args.fold_A
img_fold_B = args.fold_B

# 출력 디렉토리 설정
img_fold_AB = args.fold_AB
if not os.path.isdir(img_fold_AB):
    os.makedirs(img_fold_AB)

# noisy 이미지 파일 목록 가져오기
img_list_B = os.listdir(img_fold_B)

# 이미지 매칭 및 저장
num_imgs = min(args.num_imgs, len(img_list_B))
print(f'Use {num_imgs} images for combination.')

for n in range(num_imgs):
    name_B = img_list_B[n]
    path_B = os.path.join(img_fold_B, name_B)

    # noisy 파일 이름의 '_FF.jpg'를 '_GT.jpg'로 변경하여 clean 파일 이름 생성
    name_A = name_B.replace('_FF.jpg', '_GT.jpg')
    path_A = os.path.join(img_fold_A, name_A)

    # clean과 noisy 이미지가 모두 존재하는 경우에만 결합 수행
    if os.path.isfile(path_A) and os.path.isfile(path_B):
        # 이미지 읽기
        im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
        im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)

        # 이미지 좌우로 결합
        im_AB = np.concatenate([im_A, im_B], axis=1)

        # 결과 파일 경로 설정
        path_AB = os.path.join(img_fold_AB, name_A)  # 이름은 clean 이미지 기준으로 저장

        # 결합된 이미지 저장
        cv2.imwrite(path_AB, im_AB)
        #print(f'Saved combined image: {path_AB}')

print("All matching images have been successfully combined and saved in the 'combined' directory.")
