import cv2
from os.path import join
from os import listdir
from glob import glob


def read_images_from_folder(folder_path: str, width: int = 128, height: int = 128):
    '''Reads all jpg images from a folder, resizes them using 'width' 
    and 'height' (which default to 128 respectively), changes 
    their color code to RGB and returns them in a list'''  
    
    print("Files found in folder: ")
    img_list = list()
    try:
        for file in listdir(folder_path):
            print(f"{file}")
            if file.endswith(".jpg"):
                img = cv2.imread(join(folder_path, file))
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_list.append(img_list)
    except OSError:
        print(f"Error: Unable to read images from path {folder_path}, "
              "which does not exist")
        raise
    return img_list

def create_hdf5_from_images(path: str, width: int = 128, height: int = 128):
    '''TODO: Create hdf5 file with folder names as classes and parsed images'''
    pass

def main():
    read_images_from_folder("C:\\Users\\maerkleJ\\code\\toyProjects\\transferTraining\\images\\test_img")

if __name__ == '__main__':
    # environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()