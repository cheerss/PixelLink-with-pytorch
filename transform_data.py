import numpy as np
import codecs
import cv2
import matplotlib.pylab as plt
import read_data
import config

train_images_dir = config.train_images_dir
train_labels_dir = config.train_labels_dir
save_images_dir = "train_images/resize_images/"
save_labels_dir = "train_images/resize_ground_truth/"

def test_main():
    image = cv2.imread(train_images_dir + "img_1.jpg")
    shape = image.shape
    image_resized = cv2.resize(image, (512, 512))
    cv2.imwrite(save_images_dir + "img_1.jpg", image_resized)
    import IPython

    labels = read_data.read_ground_truth(train_labels_dir, 1)
    j = 0
    IPython.embed()
    for box in labels[0]:
        img_crop = image[box[1]: box[5], box[0]:box[4]]
        cv2.imwrite(save_images_dir + str(j) + "img.jpg", img_crop)
        j += 1
    for box in labels[0]:
        for i in range(0, 8, 2):
            box[i] = (int)(box[i] * 512 / shape[1])
        for i in range(1, 8, 2):
            box[i] = (int)(box[i] * 512 / shape[0])
        img_crop = image_resized[box[1]: box[5], box[0]:box[4]]
        cv2.imwrite(save_images_dir + str(j) + "img.jpg", img_crop)
        j += 1

def resize_images():
    all_images = read_data.read_datasets(train_images_dir, config.all_trains, order="hwc")
    i = 1
    for image in all_images:
        image = cv2.resize(image, (512, 512))
        cv2.imwrite(save_images_dir + "img_" + str(i) + ".jpg", image)
        if i % 100 == 0:
            print(i)
        i += 1

def resize_labels():
    all_labels = read_data.read_ground_truth(train_labels_dir, config.all_trains)
    image_width = config.image_width
    image_height = config.image_height
    j = 1
    for boxes in all_labels:
        with open(save_labels_dir + "gt_img_" + str(j) + ".txt", "w") as f:
            for box in boxes:
                for i in range(0, 8, 2):
                    box[i] = (int)(box[i] * 512 / image_width )
                for i in range(1, 8, 2):
                    box[i] = (int)(box[i] * 512 / image_height)
                for num in box:
                    f.write(str(num) + ", ")
                f.write("\n")
        j += 1

def main():
    resize_labels()

if __name__ == "__main__":
    main()
