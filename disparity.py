import numpy as np
import cv2
from pathlib import Path
import time

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
                # if i == 2: #в массив изображений добавляется лишь каждый третий кадр
                #     images.append(frame)
                #     i=0
                # else: i+=1
                images.append(frame)
            else:
                cap.release()       
        return np.array(images)

def calculate_disparity(imagesL, imagesR):
    num_frames = len(imagesL)

    # Создаем объект StereoBM для расчета диспаритета
    # Чем больше значение numDisparities, тем больше диапазон глубин будет рассматриваться
    # blockSize  размер окна, используемого для сопоставления блоков между левым и правым изображениями
    stereo_bm = cv2.StereoBM_create(numDisparities=256, blockSize=5)

    disparities = []

    for i in range(num_frames):
        # Загружаем изображения с левой и правой камеры
        imgL = imagesL[i]
        imgR = imagesR[i]

        # Переводим изображения в оттенки серого
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Вычисляем диспаритет
        disparity = stereo_bm.compute(grayL,grayR)

        # Нормализуем диспаритет для визуализации
        normalized_disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        disparities.append(normalized_disparity)

    return disparities

def get_images():
    # Загружаем видео с левой и правой камеры
    pathL = str(Path('data','kem.011.001.left.avi'))
    pathR = str(Path('data','kem.011.001.right.avi'))
    imagesL = _load_images(pathL)
    imagesR = _load_images(pathR)
    return imagesL, imagesR

def imshow_disparities(disparities):
    # Визуализируем диспаритет
    for disparity in disparities:
        cv2.imshow('Disparity', disparity)
        if cv2.waitKey(200) & 0xFF == ord('q'):  # Ожидаем 200 миллисекунд (5 кадров в секунду)
            break
        time.sleep(0.2)  # Задержка между кадрами

    cv2.destroyAllWindows()

def calib_load():
    calibration_dir = "./calibration_results"
    left_npzfile = np.load("{}/calibration_left.npz".format(calibration_dir))
    right_npzfile = np.load("{}/calibration_right.npz".format(calibration_dir))
    left_map_x_undistort = left_npzfile["left_map"]
    right_map_x_undistort = right_npzfile["left_map"]
    left_map_y_undistort = left_npzfile["right_map"]
    right_map_y_undistort = right_npzfile["right_map"]
    return left_map_x_undistort,right_map_x_undistort,left_map_y_undistort,right_map_y_undistort

def calib_image(imagesL, imagesR):

    resized_width = 1280 
    resized_height = 480
    calibL = []
    calibR = []
    left_map_x_undistort,right_map_x_undistort,left_map_y_undistort,right_map_y_undistort = calib_load()
    for i in range(len(imagesL)):

        undistorted_left = cv2.remap(imagesL[i], left_map_x_undistort, left_map_y_undistort, interpolation=cv2.INTER_LINEAR)
        undistorted_right = cv2.remap(imagesR[i], right_map_x_undistort, right_map_y_undistort, interpolation=cv2.INTER_LINEAR)
        #joined_undistort = np.concatenate([undistorted_left, undistorted_right], axis=1)
        #joined_undistorted_small = cv2.resize(joined_undistort, (resized_width, resized_height))
        # cv2.imshow('',joined_undistorted_small)
        # cv2.waitKey()

        calibL.append(undistorted_left)
        calibR.append(undistorted_right)

    return np.array(calibL),np.array(calibR)


def main():

    imagesL, imagesR = get_images()

    imagesL, imagesR = calib_image(imagesL, imagesR)

    # Рассчитываем диспаритет
    disparities = calculate_disparity(imagesL, imagesR)
    
    imshow_disparities(disparities)


if __name__ == '__main__':
    main()
    pass
