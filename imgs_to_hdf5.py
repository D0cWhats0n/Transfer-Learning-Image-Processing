import cv2
from os.path import join
from os import listdir, walk
from glob import glob
import h5py
import numpy as np

def read_images_from_folder(folder_path: str, width: int = 128, height: int = 128):
    '''Reads all jpg images from a folder, resizes them using 'width' 
    and 'height' (which default to 128 respectively), changes 
    their color code to RGB and returns them in a list'''  
    
    img_list = np.ndarray(shape=(len(listdir(folder_path)), width, height, 3))
    try:
        for idx, file in enumerate(listdir(folder_path)):
            if file.endswith(".jpg"):
                img = cv2.imread(join(folder_path, file))
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_list[idx, :] = img
    except OSError:
        print(f"Error: Unable to read images from path {folder_path}, "
              "which does not exist")
        raise
    return img_list


def get_file_number(path: str):
    '''Returns the number of files in the directory 'path' and 
    all subdirectories'''
    try:
        return sum([len(files) for r, d, files in walk(path)])
    except OSError:
        print("Error when calculating numbe of files")
        raise


def create_hdf5_from_images(path: str, width: int = 128, height: int = 128):
    '''TODO: Create hdf5 file with folder names as classes and parsed images'''
    with h5py.File('test.h5', 'w') as img_archive:
        try:
            file_num = get_file_number(path)

            # Instantiating numpy arrays
            num_classes = len(listdir(path))
            imgs_classes = np.ndarray(shape=(file_num, num_classes))
            imgs = np.ndarray(shape=(file_num, width, height, 3))
            imgs_class_names = listdir(path)
            file_cnt = 0

            for idx, folder in enumerate(imgs_class_names):
                one_hot_class = np.zeros(num_classes, dtype=int)
                one_hot_class[idx] = 1
                folder_imgs = read_images_from_folder(join(path, folder))
                imgs_classes[file_cnt:len(folder_imgs) + 1, :] = len(folder_imgs)*[one_hot_class]
                imgs[file_cnt:len(folder_imgs) + 1, :] = folder_imgs
                file_cnt += len(folder_imgs)
                print(f"{len(folder_imgs)} images of class {imgs_class_names[idx]}")
        except:
            raise
        #img_archive.create_dataset('images', imgs.shape, np.int8)
        img_archive['images'] = imgs
        
        print(img_archive['images'])
        print("Done")
    # #img_archive['classes'] = imgs_classes
    # #img_archive['class_names'] = imgs_class_names
    # img_archive.close()
def main():
    create_hdf5_from_images("C:\\Users\\maerkleJ\\code\\toyProjects\\Transfer-Learning-Image-Processing\\input")

if __name__ == '__main__':
    # environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()