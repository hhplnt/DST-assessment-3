import cv2
import os
import time
from multiprocessing import Pool


start_time = time.time()

path = "/Users/xin/Library/CloudStorage/OneDrive-UniversityofBristol/DST/DST-assessment-3/Data/PlantVillage/Tomato_healthy"
save_path = "./parallelly_processed_images1"
os.makedirs(save_path, exist_ok=True)

def image_process(image_path):
    img = cv2.imread(os.path.join(path, image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (224, 224), interpolation = cv2.INTER_AREA)
    cv2.imwrite(os.path.join(save_path, image_path), resized)
    return

def main():
    list_image = os.listdir(path)
    workers = os.cpu_count()
    # number of processors used will be equal to workers
    with Pool(workers) as p:
        p.map(image_process, list_image)

    print('Processing time: {0} [sec]'.format(time.time() - start_time))


if __name__ == '__main__':
    main()

