import ImgTransform
import ImgShow
import cv2
import codecs

def testTransform():
    filename = "img_1.jpg"
    img = ImgTransform.ReadImage(filename)
    img, angle = ImgTransform.RotateImage(data=img, angle=90)
    img, img_range = ImgTransform.CropImage(data=img, scale=0.8)
    img, size = ImgTransform.ResizeImage((512, 512), data=img)
    output_name = filename[:-4] + "_transform" + filename[-4:]
    img = img[..., [2, 1, 0]]
    cv2.imwrite(output_name, img)

def ReadLabels():
    label = {}
    label["coor"] = []
    label["word"] = []
    label["ignore"] = []
    with codecs.open("gt_img_1.txt", encoding="utf-8_sig") as file:
        data = file.readlines()
        coor = []
        word = []
        ignore = []
        for line in data:
            content = line.split(",")
            coor.append([int(n) for n in content[:8]])
            content[8] = content[8].strip("\r\n")
            # print(content[8])
            word.append(content[8])
            if content[8] == "###":
                ignore.append(True)
            else:
                ignore.append(False)
        label["coor"] = coor
        label["word"] = word
        label["ignore"] = ignore
    return label

def testTransformWithLabel():
    filename = "img_1.jpg"
    output_name = filename[:-4] + "_drawlabel" + filename[-4:]
    img = ImgTransform.ReadImage(filename)
    labels = ReadLabels()
    labels, img, angle = ImgTransform.RotateImageWithLabel(labels, data=img)
    labels, img, img_range = ImgTransform.CropImageWithLabel(labels, data=img, scale=0.8)
    labels, img, size = ImgTransform.ResizeImageWithLabel(labels, (512, 512), data=img)
    img = ImgShow.DrawLabels(img, labels)
    cv2.imwrite(output_name, img)


def main():
    testTransform()
    # testTransformWithLabel()

if __name__ == '__main__':
    main()