# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import re
import numpy as np
import imgui
import PIL.Image
from gui_utils import imgui_utils
from . import renderer
import cv2

#----------------------------------------------------------------------------

class WebcamWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.capturing = False
        self.disabled_time = 0
        self.cap = self.cam = None
        self.cam_port = 5
        self.working_ports = []
        self.cap_frame = True


    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):

        def list_camera_ports():
            """
            Test the ports and returns a tuple with the available ports and the ones that are working.
            """
            non_working_ports = []
            dev_port = 0
            working_ports = []
            available_ports = []
            while len(non_working_ports) < 6:  # if there are more than 5 non working ports stop the testing.
                camera = cv2.VideoCapture(dev_port)
                if not camera.isOpened():
                    non_working_ports.append(dev_port)
                    print("Port %s is not working." % dev_port)
                else:
                    is_reading, img = camera.read()
                    w = camera.get(3)
                    h = camera.get(4)
                    if is_reading:
                        print("Port %s is working and reads images (%s x %s)" % (dev_port, h, w))
                        working_ports.append(dev_port)
                    else:
                        print("Port %s for camera ( %s x %s) is present but does not reads." % (dev_port, h, w))
                        available_ports.append(dev_port)
                camera.release()
                dev_port += 1
            return available_ports, working_ports, non_working_ports

        viz = self.viz
        if show:
            with imgui_utils.grayed_out(self.disabled_time != 0):
                imgui.text('Webcam Input')
                imgui.same_line(viz.label_w)

                _changed, self.cam = imgui_utils.input_text('##path', str(self.cam_port), 1024,
                    flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                    width=(viz.button_w * 2 ),  help_text='Webcam ID')

                if _changed:
                    self.cam_port = int(self.cam)

                imgui.same_line()
                if imgui_utils.button('Refresh Cams', width=viz.button_w, enabled=True):
                    _, self.working_ports, _ = list_camera_ports()
                    imgui.open_popup('available_cameras')

                if imgui.begin_popup('available_cameras'):
                    for cam_id in self.working_ports:
                        clicked, _state = imgui.menu_item(str(cam_id))
                        if clicked:
                            self.cam_port = int(cam_id)
                    imgui.end_popup()

                imgui.same_line()
                if imgui_utils.button(
                        'Start Webcam', width=viz.button_w,
                        enabled=(self.disabled_time==0 and not self.capturing and not self.cam_port is None)):
                    print(f'Starting webcam')
                    self.capturing = True
                    viz.result.message = f'Running webcam...'
                    self.disabled_time = 0.5
                    # Open the default camera (usually the built-in webcam)
                    self.cap = cv2.VideoCapture(self.cam_port)
                    delattr(viz.result, "message")

                imgui.same_line()
                if imgui_utils.button('Stop Webcam', width=viz.button_w,
                                      enabled=(self.disabled_time==0 and self.capturing)):
                    print(f'stopping webcam')
                    self.cap.release()
                    self.disabled_time = 0.5
                    self.capturing = False

                imgui.same_line()
                if imgui_utils.button('Toggle Cap Frame', width=viz.button_w,
                                      enabled=(self.disabled_time==0 and self.capturing)):
                    self.cap_frame = not self.cap_frame

            if self.capturing:
                ret, frame = self.cap.read()
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                shape = img.shape
                h = w = min(shape[:2])
                x = shape[1] / 2 - w / 2
                y = shape[0] / 2 - h / 2

                crop_img = img[int(y):int(y + h), int(x):int(x + w)]
                if self.cap_frame:
                    axesLength = (141, 188)
                    # color in BGR
                    color = (128, 0, 255)
                    # draw ellipses
                    crop_img = cv2.ellipse(crop_img, (int(h/2), int(w/2)), axesLength, 0, 0, 360, color, 5)
                viz.result.image = crop_img

        self.disabled_time = max(self.disabled_time - viz.frame_delta, 0)

#-----------------------------------------------------------------------------
