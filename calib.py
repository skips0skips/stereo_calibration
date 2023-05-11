import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import yaml


BOARD_SIZE = [7,7] #размер шахматной доски

def _load_images(filepath):
        """
        Функция создает массив изображений из видеозаписи
        Параметры:
        filepath (str): путь к каталогу с видео
        Возвращает:
        images (list): изображения
        """
        images = list()
        cap = cv2.VideoCapture(filepath)
        i=0 #переменная отслеживает каждый третий кадр
        while cap.isOpened():
            succeed, frame = cap.read()
            if succeed:
                if i == 5: #в массив изображений добавляется лишь каждый пятый кадр
                    images.append(frame)
                    i=0
                else: i+=1
                images.append(frame)
            else:
                cap.release()       
        return np.array(images)

def read_yaml():

    # cam_plg_left = str(Path('calib','cam_plg_left.yml'))
    # fs = cv2.FileStorage(cam_plg_left, cv2.FILE_STORAGE_READ)
    # cam_plg_right = str(Path('calib','cam_plg_right.yml'))
    # extrinsics = str(Path('calib','extrinsics.yml'))    
    # # Чтение данных из intrinsics.yml (левая камера)
    # with open(cam_plg_left) as file:
    #     intrinsics_data = yaml.safe_load(file, Loader=yaml.FullLoader)
    #     left_camera_matrix = intrinsics_data['camera_matrix']
    #     left_distortion_coeffs = intrinsics_data['distortion_coefficients']

    # # Чтение данных из intrinsics.yml (правая камера)
    # with open(cam_plg_right, 'r') as file:
    #     intrinsics_data = yaml.safe_load(file, Loader=yaml.Loader)
    #     right_camera_matrix = intrinsics_data['camera_matrix']
    #     right_distortion_coeffs = intrinsics_data['distortion_coefficients']

    # # Чтение данных из extrinsics.yml
    # with open(extrinsics, 'r') as file:
    #     extrinsics_data = yaml.safe_load(file, Loader=yaml.Loader)
    #     R = extrinsics_data['R']
    #     T = extrinsics_data['T']
    #     E = extrinsics_data['E']
    #     F = extrinsics_data['F']
    pass



def main(pathleft,pathright):
    '''
    Выполняем калибровку левой и правой камеры, 
    а затем выполняет калибровку стерео-камеры.
    '''
    read_yaml()

    # Путь к видеофайлам
    pathL = str(Path('data',pathleft))
    pathR = str(Path('data',pathright))

    imagesL = _load_images(pathL)
    imagesR = _load_images(pathR)

    objp = np.zeros((BOARD_SIZE[0]*BOARD_SIZE[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:BOARD_SIZE[0],0:BOARD_SIZE[1]].T.reshape(-1,2)

    #Критерии завершения для уточнения обнаруженных углов
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    img_ptsL = []
    img_ptsR = []
    obj_pts = []
    
    for i in  tqdm(range(0,len(imagesL)),desc='Calib process'):  
        imgL = imagesL[i]
        imgR = imagesR[i]
        imgL_gray = cv2.cvtColor(imagesL[i], cv2.COLOR_BGR2GRAY)
        imgR_gray = cv2.cvtColor(imagesR[i], cv2.COLOR_BGR2GRAY)
        
        outputL = imgL.copy()
        outputR = imgR.copy()

        retR, cornersR =  cv2.findChessboardCorners(imgR_gray,BOARD_SIZE,None)
        retL, cornersL = cv2.findChessboardCorners(imgL_gray,BOARD_SIZE,None)
        
        if retR and retL:
            obj_pts.append(objp)
            cv2.cornerSubPix(imgR_gray,cornersR,(11,11),(-1,-1),criteria)
            cv2.cornerSubPix(imgL_gray,cornersL,(11,11),(-1,-1),criteria)
            cv2.drawChessboardCorners(outputR,BOARD_SIZE,cornersR,retR)
            cv2.drawChessboardCorners(outputL,BOARD_SIZE,cornersL,retL)
            # cv2.imshow('cornersR',outputR)
            # cv2.imshow('cornersL',outputL)
            # cv2.waitKey()    
            img_ptsL.append(cornersL)
            img_ptsR.append(cornersR)
     
    # Калибровка левой камеры
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts,img_ptsL,imgL_gray.shape[::-1],None,None)
    hL,wL= imgL_gray.shape[:2]
    new_mtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))
    
    # Калибровка правой камеры
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts,img_ptsR,imgR_gray.shape[::-1],None,None)
    hR,wR= imgR_gray.shape[:2]
    new_mtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # Здесь мы исправляем встроенную матрицу камеры таким образом, чтобы вычислялись только Root, Trans, Mat и Feet.
    # Следовательно, внутренние параметры одинаковы 
    
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
      
    # Этот шаг выполняется для преобразования между двумя камерами и вычисления существенной и фундаментальной матрицы
    ret, left_camera_matrix, left_distortion_coeffs, right_camera_matrix, right_distortion_coeffs, R, T, E, F = cv2.stereoCalibrate(obj_pts, img_ptsL, img_ptsR, new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::-1], criteria_stereo, flags)

    return ret, left_camera_matrix, left_distortion_coeffs, right_camera_matrix, right_distortion_coeffs, R, T, E, F
    


if __name__ == '__main__':
    ret, left_camera_matrix, left_distortion_coeffs, right_camera_matrix, right_distortion_coeffs, R, T, E, F = main(pathleft='kem.001.003.left.avi',pathright='kem.001.003.right.avi')

    # Выводим результаты калибровки
    print("Матрица левой камеры:")
    print(left_camera_matrix)
    print("Коэффициенты дисторсии левой камеры:")
    print(left_distortion_coeffs)
    print("Матрица правой камеры:")
    print(right_camera_matrix)
    print("Коэффициенты дисторсии правой камеры:")
    print(right_distortion_coeffs)
    print("Матрица поворота:")
    print(R)
    print("Вектор переноса:")
    print(T)
    print("Сущность фундаментальной матрицы:")
    print(F)
    pass