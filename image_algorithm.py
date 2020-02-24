import cv2 as cv
from sklearn.cluster import KMeans
import numpy as np
from detecto import core, utils, visualize

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

from detecto.utils import reverse_normalize,  _is_iterable
from torchvision import transforms

def image_recognition(image_path,save_path):
    image = utils.read_image(image_path)
    model = core.Model()

    labels, boxes, scores = model.predict_top(image)

    fig, ax = plt.subplots(1)
    # If the image is already a tensor, convert it back to a PILImage
    # and reverse normalize it
    if isinstance(image, torch.Tensor):
        image = reverse_normalize(image)
        image = transforms.ToPILImage()(image)
    ax.imshow(image)

    # Show a single box or multiple if provided
    if boxes.ndim == 1:
        boxes = boxes.view(1, 4)

    if labels is not None and not _is_iterable(labels):
        labels = [labels]

    # Plot each box
    for i in range(boxes.shape[0]):
        box = boxes[i]
        width, height = (box[2] - box[0]).item(), (box[3] - box[1]).item()
        initial_pos = (box[0].item(), box[1].item())
        rect = patches.Rectangle(initial_pos, width, height, linewidth=1,
                                 edgecolor='r', facecolor='none')
        if labels:
            ax.text(box[0] + 5, box[1] - 5, '{}'.format(labels[i]), color='red')

        ax.add_patch(rect)

    plt.savefig(save_path)

def is_img(ext):
    if ext.endswith('.jpg'):
        return True
    elif ext.endswith('.png'):
        return True
    elif ext.endswith('.jpeg'):
        return True
    elif ext.endswith('.bmp'):
        return True
    else:
        return False

def loadData(filepath):
    data = []
    image = cv.imread(filepath,cv.IMREAD_GRAYSCALE)
    img = np.array(image, dtype=float)
    imageshape = img.shape
    for i in range(imageshape[0]):
        for j in range(imageshape[1]):
                img[i][j] = img[i][j] / 255.0
                data.append(img[i][j])
    return img,imageshape[0],imageshape[1]

def kmean_img(filepath,fileName_choose,n_clusters):
    imgData ,x ,y= loadData(filepath)
    km = KMeans(n_clusters=n_clusters)
    label = km.fit_predict(imgData.reshape(x*y,1))
    label = np.array(label).reshape(x,y)
    labelshape = label.shape
    if  n_clusters == 2:
        for i in range(labelshape[0]):
            for j in range(labelshape[1]):
                if label[i][j] == 0:
                    imgData[i][j] = 0
                elif label[i][j] == 1:
                    imgData[i][j] = 255
        cv.imwrite(fileName_choose + '/segmentation1.png', imgData)
    elif n_clusters == 3:
        img1 = np.random.rand(x * y).reshape(x, y)
        img2 = np.random.rand(x * y).reshape(x, y)
        img3 = np.random.rand(x * y).reshape(x, y)
        for i in range(labelshape[0]):
            for j in range(labelshape[1]):
                if label[i][j] == 1:
                    imgData[i][j] = 0
                    img1[i][j] = 0
                    img2[i][j] = 255
                    img3[i][j] = 255
                elif label[i][j] == 2:
                    imgData[i][j] = 128
                    img1[i][j] = 255
                    img2[i][j] = 0
                    img3[i][j] = 255
                else:
                    imgData[i][j] = 255
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 0
        cv.imwrite(fileName_choose+'/segmentation1.png', img1)
        cv.imwrite(fileName_choose+'/segmentation2.png', img2)
        cv.imwrite(fileName_choose+'/segmentation3.png', img3)
    elif n_clusters == 4:
        img1 = np.random.rand(x * y).reshape(x, y)
        img2 = np.random.rand(x * y).reshape(x, y)
        img3 = np.random.rand(x * y).reshape(x, y)
        img4 = np.random.rand(x * y).reshape(x, y)
        for i in range(labelshape[0]):
            for j in range(labelshape[1]):
                if label[i][j] == 0:
                    imgData[i][j] = 0
                    img1[i][j] = 0
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                elif label[i][j] == 1:
                    imgData[i][j] = 85
                    img1[i][j] = 255
                    img2[i][j] = 0
                    img3[i][j] = 255
                    img4[i][j] = 255
                elif label[i][j] == 2:
                    imgData[i][j] = 170
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 0
                    img4[i][j] = 255
                else:
                    imgData[i][j] = 255
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 0
        cv.imwrite(fileName_choose+'/segmentation1.png', img1)
        cv.imwrite(fileName_choose+'/segmentation2.png', img2)
        cv.imwrite(fileName_choose+'/segmentation3.png', img3)
        cv.imwrite(fileName_choose+'/segmentation4.png', img4)
    elif n_clusters == 5:
        img1 = np.random.rand(x * y).reshape(x, y)
        img2 = np.random.rand(x * y).reshape(x, y)
        img3 = np.random.rand(x * y).reshape(x, y)
        img4 = np.random.rand(x * y).reshape(x, y)
        img5 = np.random.rand(x * y).reshape(x, y)
        for i in range(labelshape[0]):
            for j in range(labelshape[1]):
                if label[i][j] == 0:
                    imgData[i][j] = 0
                    img1[i][j] = 0
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                elif label[i][j] == 1:
                    imgData[i][j] = 64
                    img1[i][j] = 255
                    img2[i][j] = 0
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                elif label[i][j] == 2:
                    imgData[i][j] = 128
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 0
                    img4[i][j] = 255
                    img5[i][j] = 255
                elif label[i][j] == 3:
                    imgData[i][j] = 172
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 0
                    img5[i][j] = 255
                else:
                    imgData[i][j] = 255
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 0
        cv.imwrite(fileName_choose+'/segmentation1.png', img1)
        cv.imwrite(fileName_choose+'/segmentation2.png', img2)
        cv.imwrite(fileName_choose+'/segmentation3.png', img3)
        cv.imwrite(fileName_choose+'/segmentation4.png', img4)
        cv.imwrite(fileName_choose+'/segmentation5.png', img5)
    elif n_clusters == 6:
        img1 = np.random.rand(x * y).reshape(x, y)
        img2 = np.random.rand(x * y).reshape(x, y)
        img3 = np.random.rand(x * y).reshape(x, y)
        img4 = np.random.rand(x * y).reshape(x, y)
        img5 = np.random.rand(x * y).reshape(x, y)
        img6 = np.random.rand(x * y).reshape(x, y)
        for i in range(labelshape[0]):
            for j in range(labelshape[1]):
                if label[i][j] == 0:
                    imgData[i][j] = 0
                    img1[i][j] = 0
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 255
                elif label[i][j] == 1:
                    imgData[i][j] = 51
                    img1[i][j] = 255
                    img2[i][j] = 0
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 255
                elif label[i][j] == 2:
                    imgData[i][j] = 102
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 0
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 255
                elif label[i][j] == 3:
                    imgData[i][j] = 153
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 0
                    img5[i][j] = 255
                    img6[i][j] = 255
                elif label[i][j] == 4:
                    imgData[i][j] = 204
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 0
                    img6[i][j] = 255
                else:
                    imgData[i][j] = 255
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 0
        cv.imwrite(fileName_choose+'/segmentation1.png', img1)
        cv.imwrite(fileName_choose+'/segmentation2.png', img2)
        cv.imwrite(fileName_choose+'/segmentation3.png', img3)
        cv.imwrite(fileName_choose+'/segmentation4.png', img4)
        cv.imwrite(fileName_choose+'/segmentation5.png', img5)
        cv.imwrite(fileName_choose + '/segmentation6.png', img6)
    elif n_clusters == 7:
        img1 = np.random.rand(x * y).reshape(x, y)
        img2 = np.random.rand(x * y).reshape(x, y)
        img3 = np.random.rand(x * y).reshape(x, y)
        img4 = np.random.rand(x * y).reshape(x, y)
        img5 = np.random.rand(x * y).reshape(x, y)
        img6 = np.random.rand(x * y).reshape(x, y)
        img7 = np.random.rand(x * y).reshape(x, y)
        for i in range(labelshape[0]):
            for j in range(labelshape[1]):
                if label[i][j] == 0:
                    imgData[i][j] = 0
                    img1[i][j] = 0
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 255
                    img7[i][j] = 255
                elif label[i][j] == 1:
                    imgData[i][j] = 42
                    img1[i][j] = 255
                    img2[i][j] = 0
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 255
                    img7[i][j] = 255
                elif label[i][j] == 2:
                    imgData[i][j] = 84
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 0
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 255
                    img7[i][j] = 255
                elif label[i][j] == 3:
                    imgData[i][j] = 126
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 0
                    img5[i][j] = 255
                    img6[i][j] = 255
                    img7[i][j] = 255
                elif label[i][j] == 4:
                    imgData[i][j] = 168
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 0
                    img6[i][j] = 255
                    img7[i][j] = 255
                elif label[i][j] == 5:
                    imgData[i][j] = 210
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 0
                    img7[i][j] = 255
                else:
                    imgData[i][j] = 255
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 255
                    img7[i][j] = 0
        cv.imwrite(fileName_choose+'/segmentation1.png', img1)
        cv.imwrite(fileName_choose+'/segmentation2.png', img2)
        cv.imwrite(fileName_choose+'/segmentation3.png', img3)
        cv.imwrite(fileName_choose+'/segmentation4.png', img4)
        cv.imwrite(fileName_choose+'/segmentation5.png', img5)
        cv.imwrite(fileName_choose + '/segmentation6.png', img6)
        cv.imwrite(fileName_choose + '/segmentation7.png', img7)
    elif n_clusters == 8:
        img1 = np.random.rand(x * y).reshape(x, y)
        img2 = np.random.rand(x * y).reshape(x, y)
        img3 = np.random.rand(x * y).reshape(x, y)
        img4 = np.random.rand(x * y).reshape(x, y)
        img5 = np.random.rand(x * y).reshape(x, y)
        img6 = np.random.rand(x * y).reshape(x, y)
        img7 = np.random.rand(x * y).reshape(x, y)
        img8 = np.random.rand(x * y).reshape(x, y)
        for i in range(labelshape[0]):
            for j in range(labelshape[1]):
                if label[i][j] == 0:
                    imgData[i][j] = 0
                    img1[i][j] = 0
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 255
                    img7[i][j] = 255
                    img8[i][j] = 255
                elif label[i][j] == 1:
                    imgData[i][j] = 36
                    img1[i][j] = 255
                    img2[i][j] = 0
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 255
                    img7[i][j] = 255
                    img8[i][j] = 255
                elif label[i][j] == 2:
                    imgData[i][j] = 72
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 0
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 255
                    img7[i][j] = 255
                    img8[i][j] = 255
                elif label[i][j] == 3:
                    imgData[i][j] = 108
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 0
                    img5[i][j] = 255
                    img6[i][j] = 255
                    img7[i][j] = 255
                    img8[i][j] = 255
                elif label[i][j] == 4:
                    imgData[i][j] = 144
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 0
                    img6[i][j] = 255
                    img7[i][j] = 255
                    img8[i][j] = 255
                elif label[i][j] == 5:
                    imgData[i][j] = 180
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 0
                    img7[i][j] = 255
                    img8[i][j] = 255
                elif label[i][j] == 6:
                    imgData[i][j] = 216
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 255
                    img7[i][j] = 0
                    img8[i][j] = 255
                else:
                    imgData[i][j] = 255
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 255
                    img7[i][j] = 255
                    img8[i][j] = 0
        cv.imwrite(fileName_choose+'/segmentation1.png', img1)
        cv.imwrite(fileName_choose+'/segmentation2.png', img2)
        cv.imwrite(fileName_choose+'/segmentation3.png', img3)
        cv.imwrite(fileName_choose+'/segmentation4.png', img4)
        cv.imwrite(fileName_choose+'/segmentation5.png', img5)
        cv.imwrite(fileName_choose + '/segmentation6.png', img6)
        cv.imwrite(fileName_choose + '/segmentation7.png', img7)
        cv.imwrite(fileName_choose + '/segmentation8.png', img8)
    else:
        img1 = np.random.rand(x * y).reshape(x, y)
        img2 = np.random.rand(x * y).reshape(x, y)
        img3 = np.random.rand(x * y).reshape(x, y)
        img4 = np.random.rand(x * y).reshape(x, y)
        img5 = np.random.rand(x * y).reshape(x, y)
        img6 = np.random.rand(x * y).reshape(x, y)
        img7 = np.random.rand(x * y).reshape(x, y)
        img8 = np.random.rand(x * y).reshape(x, y)
        img9 = np.random.rand(x * y).reshape(x, y)
        for i in range(labelshape[0]):
            for j in range(labelshape[1]):
                if label[i][j] == 0:
                    imgData[i][j] = 0
                    img1[i][j] = 0
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 255
                    img7[i][j] = 255
                    img8[i][j] = 255
                    img9[i][j] = 255
                elif label[i][j] == 1:
                    imgData[i][j] = 32
                    img1[i][j] = 255
                    img2[i][j] = 0
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 255
                    img7[i][j] = 255
                    img8[i][j] = 255
                    img9[i][j] = 255
                elif label[i][j] == 2:
                    imgData[i][j] = 64
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 0
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 255
                    img7[i][j] = 255
                    img8[i][j] = 255
                    img9[i][j] = 255
                elif label[i][j] == 3:
                    imgData[i][j] = 96
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 0
                    img5[i][j] = 255
                    img6[i][j] = 255
                    img7[i][j] = 255
                    img8[i][j] = 255
                    img9[i][j] = 255
                elif label[i][j] == 4:
                    imgData[i][j] = 128
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 0
                    img6[i][j] = 255
                    img7[i][j] = 255
                    img8[i][j] = 255
                    img9[i][j] = 255
                elif label[i][j] == 5:
                    imgData[i][j] = 160
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 0
                    img7[i][j] = 255
                    img8[i][j] = 255
                    img9[i][j] = 255
                elif label[i][j] == 6:
                    imgData[i][j] = 192
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 255
                    img7[i][j] = 0
                    img8[i][j] = 255
                    img9[i][j] = 255
                elif label[i][j] == 7:
                    imgData[i][j] = 224
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 255
                    img7[i][j] = 255
                    img8[i][j] = 0
                    img9[i][j] = 255
                else:
                    imgData[i][j] = 255
                    img1[i][j] = 255
                    img2[i][j] = 255
                    img3[i][j] = 255
                    img4[i][j] = 255
                    img5[i][j] = 255
                    img6[i][j] = 255
                    img7[i][j] = 255
                    img8[i][j] = 255
                    img9[i][j] = 0
        cv.imwrite(fileName_choose+'/segmentation1.png', img1)
        cv.imwrite(fileName_choose+'/segmentation2.png', img2)
        cv.imwrite(fileName_choose+'/segmentation3.png', img3)
        cv.imwrite(fileName_choose+'/segmentation4.png', img4)
        cv.imwrite(fileName_choose+'/segmentation5.png', img5)
        cv.imwrite(fileName_choose + '/segmentation6.png', img6)
        cv.imwrite(fileName_choose + '/segmentation7.png', img7)
        cv.imwrite(fileName_choose + '/segmentation8.png', img8)
        cv.imwrite(fileName_choose + '/segmentation9.png', img9)
    return fileName_choose+'/segmentation1.png'
    # cv.imshow('input_image2', imgData)
    # cv.imshow('input_image1', image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

def color_segmentation(state_list,save_path,image_path):
    rgb_img = cv.imread(image_path)
    a = np.shape(rgb_img)
    img = np.random.rand(a[0]*a[1]).reshape(a[0], a[1])
    HSV = cv.cvtColor(rgb_img, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(HSV)
    image_shape = np.shape(HSV)
    x = image_shape[0]
    y = image_shape[1]

    for i in range(x):
        for j in range(y):
            if state_list[0] == 2:
                if 0 <= HSV[i][j][0] <= 180 and 0 <= HSV[i][j][1] <= 255 and 0 <= HSV[i][j][2] <= 46:
                    img[i][j] = 1
            if state_list[1] == 2:
                if 0 <= HSV[i][j][0] <= 180 and 0 <= HSV[i][j][1] <= 43 and 46 <= HSV[i][j][2] <= 220:
                    img[i][j] = 1
            if state_list[2] == 2:
                if 0 <= HSV[i][j][0] <= 180 and 0 <= HSV[i][j][1] <= 30 and 221 <= HSV[i][j][2] <= 255:
                    img[i][j] = 1
            if state_list[3] == 2:
                if (0 <= HSV[i][j][0] <= 10 or 156 <= HSV[i][j][0] <= 180) and 43 <= HSV[i][j][1] <= 255 and 46 <= HSV[i][j][2] <= 255:
                    img[i][j] = 1
            if state_list[4] == 2:
                if 11 <= HSV[i][j][0] <= 25 and 43 <= HSV[i][j][1] <= 255 and 46 <= HSV[i][j][2] <= 255:
                    img[i][j] = 1
            if state_list[5] == 2:
                if 26 <= HSV[i][j][0] <= 34 and 43 <= HSV[i][j][1] <= 255 and 46 <= HSV[i][j][2] <= 255:
                    img[i][j] = 1
            if state_list[6] == 2:
                if 35 <= HSV[i][j][0] <= 77 and 43 <= HSV[i][j][1] <= 255 and 46 <= HSV[i][j][2] <= 255:
                    img[i][j] = 1
            if state_list[7] == 2:
                if 78 <= HSV[i][j][0] <= 99 and 43 <= HSV[i][j][1] <= 255 and 46 <= HSV[i][j][2] <= 255:
                    img[i][j] = 1
            if state_list[8] == 2:
                if 100 <= HSV[i][j][0] <= 124 and 43 <= HSV[i][j][1] <= 255 and 46 <= HSV[i][j][2] <= 255:
                    img[i][j] = 1
            if state_list[9] == 2:
                if 125 <= HSV[i][j][0] <= 155 and 43 <= HSV[i][j][1] <= 255 and 46 <= HSV[i][j][2] <= 255:
                    img[i][j] = 1
    for p in range(x):
        for q in range(y):
            if img[p][q] != 1:
                rgb_img[p][q][0] = 0
                rgb_img[p][q][1] = 0
                rgb_img[p][q][2] = 0
    cv.imwrite(save_path, rgb_img)




