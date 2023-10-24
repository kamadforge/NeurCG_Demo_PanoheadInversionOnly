import cv2
import argparse
from rembg import remove

import cv2
import math
import numpy as np
import face_alignment
from skimage import io


class FaceDetect:
    def __init__(self, device, detector):
        # landmarks will be detected by face_alignment library. Set device = 'cuda' if use GPU.
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    def align(self, image):
        landmarks = self.__get_max_face_landmarks(image)

        if landmarks is None:
            return None

        else:
            return self.__rotate(image, landmarks)

    def __get_max_face_landmarks(self, image):
        preds = self.fa.get_landmarks(image)
        if preds is None:
            return None

        elif len(preds) == 1:
            return preds[0]

        else:
            # find max face
            areas = []
            for pred in preds:
                landmarks_top = np.min(pred[:, 1])
                landmarks_bottom = np.max(pred[:, 1])
                landmarks_left = np.min(pred[:, 0])
                landmarks_right = np.max(pred[:, 0])
                areas.append((landmarks_bottom - landmarks_top) * (landmarks_right - landmarks_left))
            max_face_index = np.argmax(areas)
            return preds[max_face_index]

    @staticmethod
    def __rotate(image, landmarks):
        # rotation angle
        left_eye_corner = landmarks[36]
        right_eye_corner = landmarks[45]
        radian = np.arctan((left_eye_corner[1] - right_eye_corner[1]) / (left_eye_corner[0] - right_eye_corner[0]))

        # image size after rotating
        height, width, _ = image.shape
        cos = math.cos(radian)
        sin = math.sin(radian)
        new_w = int(width * abs(cos) + height * abs(sin))
        new_h = int(width * abs(sin) + height * abs(cos))

        # translation
        Tx = new_w // 2 - width // 2
        Ty = new_h // 2 - height // 2

        # affine matrix
        M = np.array([[cos, sin, (1 - cos) * width / 2. - sin * height / 2. + Tx],
                      [-sin, cos, sin * width / 2. + (1 - cos) * height / 2. + Ty]])

        image_rotate = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))

        landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
        landmarks_rotate = np.dot(M, landmarks.T).T
        return image_rotate, landmarks_rotate


class Preprocess:
    def __init__(self, device='cpu', detector='dlib'):
        self.detect = FaceDetect(device, detector)  # device = 'cpu' or 'cuda', detector = 'dlib' or 'sfd'
        # self.segment = FaceSeg()

    def process(self, image):
        face_info = self.detect.align(image)
        if face_info is None:
            return None
        image_align, landmarks_align = face_info

        face = self.__crop(image_align, landmarks_align)
        return face

    @staticmethod
    def __crop(image, landmarks):
        landmarks_top = np.min(landmarks[:, 1])
        landmarks_bottom = np.max(landmarks[:, 1])
        landmarks_left = np.min(landmarks[:, 0])
        landmarks_right = np.max(landmarks[:, 0])

        # expand bbox
        top = int(landmarks_top - 0.8 * (landmarks_bottom - landmarks_top))
        bottom = int(landmarks_bottom + 0.3 * (landmarks_bottom - landmarks_top))
        left = int(landmarks_left - 0.3 * (landmarks_right - landmarks_left))
        right = int(landmarks_right + 0.3 * (landmarks_right - landmarks_left))

        if bottom - top > right - left:
            left -= ((bottom - top) - (right - left)) // 2
            right = left + (bottom - top)
        else:
            top -= ((right - left) - (bottom - top)) // 2
            bottom = top + (right - left)

        image_crop = np.ones((bottom - top + 1, right - left + 1, 3), np.uint8) * 255

        h, w = image.shape[:2]
        left_white = max(0, -left)
        left = max(0, left)
        right = min(right, w-1)
        right_white = left_white + (right-left)
        top_white = max(0, -top)
        top = max(0, top)
        bottom = min(bottom, h-1)
        bottom_white = top_white + (bottom - top)

        image_crop[top_white:bottom_white+1, left_white:right_white+1] = image[top:bottom+1, left:right+1].copy()
        return image_crop


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', help='Path to the file with background', type=str)
    args = parser.parse_args()
    image = io.imread(args.img_path)
    processor = Preprocess()
    image = processor.process(image)

    image_nobg = remove(image)
    image_nobg = image_nobg.astype(np.float32) / 255
    image_nobg = image_nobg[:, :, :3] * image_nobg[:, :, 3:]
    image_nobg = np.clip(image_nobg * 255, 0, 255).astype(np.uint8)
    # import ipdb; ipdb.set_trace()
    io.imsave(f'{args.img_path[:-3]}_nbg.png', image_nobg)