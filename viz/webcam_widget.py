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
        self.cap = None


    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            with imgui_utils.grayed_out(self.disabled_time != 0):
                imgui.text('Webcam Input')
                imgui.same_line(viz.label_w)
                # _changed, self.path = imgui_utils.input_text('##path', self.path, 1024,
                #     flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                #     width=(-1 - viz.button_w * 2 - viz.spacing * 2),
                #     help_text='PATH')
                # if imgui.is_item_hovered() and not imgui.is_item_active() and self.path != '':
                #     imgui.set_tooltip(self.path)
                # imgui.same_line()
                if imgui_utils.button('Start Webcam', width=viz.button_w, enabled=(self.disabled_time==0 and not self.capturing)):
                    print(f'Starting webcam')
                    self.capturing = True
                    viz.result.message = f'Running webcam...'
                    self.disabled_time = 0.5
                    # Open the default camera (usually the built-in webcam)
                    self.cap = cv2.VideoCapture(0)
                    delattr(viz.result, "message")

                imgui.same_line()
                if imgui_utils.button('Stop Webcam', width=viz.button_w,  enabled=(self.disabled_time==0 and self.capturing)):
                    print(f'stopping webcam')
                    self.cap.release()
                    self.disabled_time = 0.5
                    self.capturing = False

            if self.capturing:
                ret, frame = self.cap.read()
                viz.result.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        self.disabled_time = max(self.disabled_time - viz.frame_delta, 0)

#-----------------------------------------------------------------------------
