import numpy as np
import pyvirtualcam
import argparse
import cv2
import sys
import os
import mxnet as mx
import datetime
from skimage import transform as trans
import insightface
import configparser

def square_crop(im, S):
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[:resized_im.shape[0], :resized_im.shape[1], :] = resized_im
    return det_im, scale


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    # translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)
    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    # print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)


class Handler:
    def __init__(self, prefix, epoch, im_size=192, det_size=224, ctx_id=0):
        print('loading', prefix, epoch)
        if ctx_id >= 0:
            ctx = mx.gpu(ctx_id)
        else:
            ctx = mx.cpu()
        image_size = (im_size, im_size)
        self.detector = insightface.model_zoo.get_model(
            'retinaface_mnet025_v2')  # can replace with your own face detector
        # self.detector = insightface.model_zoo.get_model('retinaface_r50_v1')
        self.detector.prepare(ctx_id=ctx_id)
        self.det_size = det_size
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        self.image_size = image_size
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model
        self.image_size = image_size

    def get(self, img, get_all=False):
        out = []
        det_im, det_scale = square_crop(img, self.det_size)
        bboxes, _ = self.detector.detect(det_im)
        if bboxes.shape[0] == 0:
            return out
        bboxes /= det_scale
        if not get_all:
            areas = []
            for i in range(bboxes.shape[0]):
                x = bboxes[i]
                area = (x[2] - x[0]) * (x[3] - x[1])
                areas.append(area)
            m = np.argsort(areas)[-1]
            bboxes = bboxes[m:m + 1]
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            input_blob = np.zeros((1, 3) + self.image_size, dtype=np.float32)
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = self.image_size[0] * 2 / 3.0 / max(w, h)
            rimg, M = transform(img, center, self.image_size[0], _scale, rotate)
            rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
            rimg = np.transpose(rimg, (2, 0, 1))  # 3*112*112, RGB
            input_blob[0] = rimg
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data,))
            self.model.forward(db, is_train=False)
            pred = self.model.get_outputs()[-1].asnumpy()[0]
            if pred.shape[0] >= 3000:
                pred = pred.reshape((-1, 3))
            else:
                pred = pred.reshape((-1, 2))
            pred[:, 0:2] += 1
            pred[:, 0:2] *= (self.image_size[0] // 2)
            if pred.shape[1] == 3:
                pred[:, 2] *= (self.image_size[0] // 2)

            IM = cv2.invertAffineTransform(M)
            pred = trans_points(pred, IM)
            out.append(pred)
        return out


def readPoints(path):
    # Create an array of points.
    points = []

    # Read points
    with open(path) as file:
        for line in file:
            x, y = line.split()
            points.append((int(x), int(y)))

    return points


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Check if a point is inside a rectangle
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


# calculate delanauy triangle
def calculateDelaunayTriangles(rect, points, points_dict):
    # create subdiv
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    triangleList = subdiv.getTriangleList()

    delaunayTri = []

    pt = []

    for index, t in enumerate(triangleList):
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            # ind.append(pt1)
            # ind.append(pt2)
            # ind.append(pt3)

            # Get index of the face-points by coordinates
            # for j in range(0, 3):
            #     for k in range(0, len(points)):
            #         if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
            #             ind.append(k)
            #             break
            ind.append(points_dict[pt1])
            ind.append(points_dict[pt2])
            ind.append(points_dict[pt3])

            # ind.append(index * 3)
            # ind.append(index * 3 + 1)
            # ind.append(index * 3 + 2)
            # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

        pt = []

    return delaunayTri


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
            (1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect

    # return img2

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    handler = Handler('new_model/2d106det', 0, ctx_id=-1,
                      det_size=120)  # чем меньше размер картинки тем быстрее инференс, но точность ниже, норм при 120..

    # Make sure OpenCV is version 3.0 or above
    # (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # Read images
    config = configparser.ConfigParser()
    config.read('settings.ini')
    filename1 = config["Settings"]["filepath"]
    print(filename1)
    prev_frame = None
    img1 = cv2.imread(filename1)
    preds_source = handler.get(img1, get_all=False)
    img1 = cv2.resize(img1, (640, 480))
    print(img1.shape)
    points1 = []
    for pred in preds_source:
        pred = np.round(pred).astype(np.int)
        for i in range(pred.shape[0]):
            p = tuple(pred[i])
            points1.append(p)

    with pyvirtualcam.Camera(width=640, height=480, fps=30, print_fps=True) as cam:
        while True:
            _, img2 = cap.read()
            # img1Warped = np.copy(img2)
            img1Warped = np.zeros_like(img2)
            img_copy = img2.copy()
            # Read array of corresponding points
            # points1 = readPoints(filename1 + '.txt')
            # points2 = readPoints(filename2 + '.txt')

            points2 = []

            preds_target = handler.get(img2, get_all=False)

            color = (200, 160, 75)

            for pred in preds_target:
                pred = np.round(pred).astype(np.int)
                for i in range(pred.shape[0]):
                    p = tuple(pred[i])
                    cv2.circle(img_copy, tuple(pred[i]), radius=1, color=color)
                    # print(pred[i])
                    points2.append(p)
            # Find convex hull
            hull1 = []
            hull2 = []
            try:
                # hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)
                hullIndex = [i for i in range(len(points2))]

            except:
                if prev_frame is not None:
                    cv2.imshow('video', output)
                    # tim = cv2.cvtColor(output, cv2.COLOR_RGB2RGBA)
                    cam.send(tim)
                    continue
                else:
                    continue
            for i in range(0, len(hullIndex)):
                hull1.append(points1[int(hullIndex[i])])
                hull2.append(points2[int(hullIndex[i])])
            # print(len(hullIndex))
            if len(hullIndex) == 0:
                continue
            # Find delanauy traingulation for convex hull points
            sizeImg2 = img2.shape
            rect = (0, 0, sizeImg2[1], sizeImg2[0])
            points_dict = dict()
            for index, curPoint in enumerate(points2):
                # print(curPoint)
                points_dict[curPoint] = index

            dt = calculateDelaunayTriangles(rect, hull2, points_dict)

            if len(dt) == 0:
                quit()

            # Apply affine transformation to Delaunay triangles
            for i in range(0, len(dt)):
                t1 = []
                t2 = []

                # get points for img1, img2 corresponding to the triangles
                for j in range(0, 3):
                    t1.append(hull1[dt[i][j]])
                    t2.append(hull2[dt[i][j]])

                # img1Warped = warpTriangle(img1, img1Warped.copy(), t1, t2)
                warpTriangle(img1, img1Warped, t1, t2)

            # Calculate Mask
            # hull8U = []
            # for i in range(0, len(hull2)):
            #     hull8U.append((hull2[i][0], hull2[i][1]))
            #
            # mask = np.zeros(img2.shape, dtype=img2.dtype)
            #
            # cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
            #
            # r = cv2.boundingRect(np.float32([hull2]))
            #
            # center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
            #
            # # Clone seamlessly.
            # output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
            # tim = cv2.cvtColor(output, cv2.COLOR_BGR2RGBA)
            cv2.imshow('video', np.uint8(img1Warped))
            # cv2.imshow('video1', img_copy)
            output = np.uint8(img1Warped)

            tim = cv2.cvtColor(output, cv2.COLOR_BGR2RGBA)

            prev_frame = True

            # output = tim
            cam.send(tim)
            cv2.waitKey(1)
            cam.sleep_until_next_frame()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # cv2.imshow("Face Swapped", output)
            # cv2.waitKey(0)

    cv2.destroyAllWindows()
