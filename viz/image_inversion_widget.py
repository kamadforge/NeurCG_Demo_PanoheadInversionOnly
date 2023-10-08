# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import sys
sys.path.append('../')
import glob
import os
import re

import dnnlib
import imgui
import numpy as np
from gui_utils import imgui_utils
import time

from . import renderer
import projector_withseg
import multiprocessing as mp
import queue


#----------------------------------------------------------------------------

def _locate_results(pattern):
    return pattern

#----------------------------------------------------------------------------

def invert(inv_q, network, img):
    inv_q.put(True)
    projector_withseg.run_projection(target_img=os.path.dirname(img),
                                     network_pkl=network,
                                     outdir=os.path.dirname(img),
                                     target_seg=None,
                                     idx=0,
                                     save_video=True,
                                     seed=666,
                                     num_steps=500,
                                     num_steps_pti=500,
                                     fps=5,
                                     shapes=False,)
    while not inv_q.empty():
        try:
            _ = inv_q.get_nowait()
        except queue.Empty:
            pass

class InvImgWidget:
    def __init__(self, viz, set_pkl_path):
        self._path_init = '/home/kamil/Dropbox/Current_research/t3D/NeurCG_Live_Demo/_screenshots'
        self.viz            = viz
        self.cur_img        = ''
        self.path_sofar = self._path_init
        self.png_aquired = False
        self.png_used = False
        self.inverting = mp.Queue()
        self.set_pkl_path = set_pkl_path
        self.path_changed = False

    def is_image_file(self, file_path):
        # List of common image file extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

        # Get the file extension from the file path
        file_extension = os.path.splitext(file_path)[1].lower()


        # Check if the file extension is in the list of image extensions
        return file_extension in image_extensions

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            imgui.text('Model Image')
            imgui.same_line(viz.label_w)
            changed, self.cur_img = imgui_utils.input_text('##png', self.cur_img, 1024,
                flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                width=(-1 - viz.button_w * 2 - viz.spacing * 2),
                help_text='<PATH> | <URL> | <RUN_DIR> | <RUN_ID> | <RUN_ID>/<KIMG>.png')
            if changed and self.is_image_file(self.cur_img):
                raise NotImplementedError("This function is not implemented yet.")
            imgui.same_line()
            try:
                inversion_in_progress = self.inverting.get(block=False)
                self.inverting.put(inversion_in_progress)
            except queue.Empty:
                inversion_in_progress = False

            if imgui_utils.button('Browse...', enabled=not inversion_in_progress, width=-1):
                imgui.open_popup('browsing_png_popup')
                # print(f'opening brows pop up')
                self.browse_refocus = True

            if inversion_in_progress:
                viz.result.message = 'Inversion in progress ...'
                # print(f'self.png_aquired : {self.png_aquired} and self.png_used{self.png_used} and self.path_changed{self.path_changed}')
            elif self.png_aquired and self.png_used and self.path_changed:
                inv_path = os.path.join(os.path.dirname(self.path_sofar),
                                        'easy-khair-180-gpc0.8-trans10-025000.pkl')
                print('Sending new picke and w file')
                self.set_pkl_path(os.path.join(inv_path, '0/fintuned_generator.pkl'))
                self.path_changed = False

        if imgui.begin_popup('browsing_png_popup'):
            self.path_changed = True
            if self.png_aquired and self.png_used:
                # second browse
                self.png_aquired = False
                self.png_used = False
                self.path_sofar = self._path_init

            if not self.is_image_file(self.path_sofar):
                self.png_aquired = False
                items = os.listdir(self.path_sofar)
                if items is None:
                    print('Go back! nothing here')

                for item in sorted(items):
                    if not item.startswith('.'):
                        clicked, _state = imgui.menu_item(item)
                        if clicked:
                            self.path_sofar = os.path.join(self.path_sofar, item)
            if self.is_image_file(self.path_sofar):
                self.png_aquired = True
                self.png_used = False

            self.cur_img = self.path_sofar
            imgui.end_popup()

        if self.png_aquired and not self.png_used:
            if viz.args.pkl is None:
                viz.result.message = 'Load network first'
            else:
                self.inverting.put(True)
                self.png_used = True
                try:
                    main_start_method = mp.get_start_method()
                    print(f'main_start_method: {main_start_method}')
                    if main_start_method is None:
                        # If not set, then set it to 'spawn'
                        mp.set_start_method('spawn')
                except RuntimeError as e:
                    print(f'error spawning {e}')
                    pass
                p = mp.Process(target=invert, args=(self.inverting, self.viz.args.pkl, self.path_sofar),
                               daemon=True)
                p.start()

#----------------------------------------------------------------------------
