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
#from pvrecorder import PvRecorder
import pyaudio
import wave
import struct

from pathlib import Path
import sys
p = Path(__file__).parents[2]
pp=str(p)+"/lip_sync"
sys.path.append(pp)
print("pp", pp)
p = Path(__file__).parents[3]
sys.path.append(str(p))
print("p", p)

from lip_sync.inference import main, get_args
from ffpyplayer.player import MediaPlayer

os.makedirs("audio", exist_ok=True)

# Sampling frequency
freq = 44100

#----------------------------------------------------------------------------

class AudioWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.capturing = False
        self.disabled_time = 0
        self.cap = None

#        self.recorder = PvRecorder(device_index=-1, frame_length=512)
        self.audio = []

        self.chunk = 1024
        # sample format
        self.FORMAT = pyaudio.paInt16
        # mono, change to 2 if you want stereo
        channels = 1
        # 44100 samples per second
        self.sample_rate = 44100
        record_seconds = 5
        # initialize PyAudio object
        self.p = pyaudio.PyAudio()

        self.cur_img = ''
        self.frames = []

        video_path = "./results/2023_10_09_22.44.59.mp4"
        # Path to video file
        print(video_path)
        self.vidObj = cv2.VideoCapture(video_path)

        self.count=0

        self.success = 1
        self.play=0
        self.making_speaking_avatar=0

        self._path_init = '../lip_sync/examples/source_image'
        self.path_sofar = self._path_init
        self.png_aquired = False
        self.png_used = False

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


            with imgui_utils.grayed_out(self.disabled_time != 0):
                imgui.text('Get audio')
                imgui.same_line(viz.label_w)
                # _changed, self.path = imgui_utils.input_text('##path', self.path, 1024,
                #     flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                #     width=(-1 - viz.button_w * 2 - viz.spacing * 2),
                #     help_text='PATH')
                # if imgui.is_item_hovered() and not imgui.is_item_active() and self.path != '':
                #     imgui.set_tooltip(self.path)
                # imgui.same_line()
                if imgui_utils.button('Start Audio Recording', width=viz.button_w, enabled=(self.disabled_time==0 and not self.capturing)):
                    print(f'Starting webcam')
                    self.capturing = True
                    viz.result.message = f'Running webcam...'
                    self.disabled_time = 0.5
                    # Open the default camera (usually the built-in webcam)
                    self.cap = cv2.VideoCapture(0)
                    delattr(viz.result, "message")

                    # self.recorder.start()

                    # open stream object as input & output
                    self.stream = self.p.open(format=self.FORMAT,
                                              channels=2,
                                              rate=self.sample_rate,
                                              input=True,
                                              output=True,
                                              frames_per_buffer=self.chunk)




                imgui.same_line()
                if imgui_utils.button('Stop Audio Recording', width=viz.button_w,  enabled=(self.disabled_time==0 and self.capturing)):
                    print(f'stopping webcam')
                    self.cap.release()
                    self.disabled_time = 0.5
                    self.capturing = False

                    # self.recorder.stop()
                    # with wave.open("audio.wav", 'w') as f:
                    #     f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
                    #     f.writeframes(struct.pack("h" * len(self.audio), *self.audio))
                    # self.recorder.delete()
                    # audio
                    self.stream.stop_stream()
                    self.stream.close()
                    # terminate pyaudio object
                    self.p.terminate()
                    # save audio file
                    # open the file in 'write bytes' mode
                    wf = wave.open("audio/audio1.wav", "wb")
                    # set the channels
                    wf.setnchannels(2)
                    # set the sample format
                    wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
                    # set the sample rate
                    wf.setframerate(self.sample_rate)
                    # write the frames as bytes
                    wf.writeframes(b"".join(self.frames))
                    # close the file
                    wf.close()

                    self.making_speaking_avatar=1
                    args = get_args(checkpoint_dir='../lip_sync/checkpoints',
                                    driven_audio='audio/audio1.wav',
                                    #source_image='../lip_sync/examples/source_image/art_0.png')
                                    source_image=self.path_sofar)
                    #viz.result.message = f'Please check the mp4...'
                    self.video_path = main(args)
                    #self.making_speaking_avatar = 0
                    #self.video_path = "./results/2023_10_09_22.44.59.mp4"
                    # Path to video file
                    print(self.video_path)
                    self.vidObj = cv2.VideoCapture(self.video_path)
                    self.player = MediaPlayer(self.video_path)
                    self.play=1




                imgui.text('Model Image')
                imgui.same_line(viz.label_w)
                changed, self.cur_img = imgui_utils.input_text('##png', self.cur_img, 1024,
                                                               flags=(
                                                                           imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                                                               width=(-1 - viz.button_w * 2 - viz.spacing * 2),
                                                               help_text='<PATH> | <URL> | <RUN_DIR> | <RUN_ID> | <RUN_ID>/<KIMG>.png')
                # if changed and self.is_image_file(self.cur_img):
                #     raise NotImplementedError("This function is not implemented yet.")
                imgui.same_line()
                # try:
                #     inversion_in_progress = self.inverting.get(block=False)
                #     self.inverting.put(inversion_in_progress)
                # except queue.Empty:
                #     inversion_in_progress = False

                if imgui_utils.button('Browse...', width=-1):
                    imgui.open_popup('browsing_png_popup')
                    # print(f'opening brows pop up')
                    self.browse_refocus = True


            if self.capturing:

                ret, frame = self.cap.read()
                viz.result.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = self.recorder.read()
                # self.audio.extend(frame)
                viz.result.message = f'Please say something clearly...'
                data = self.stream.read(self.chunk)
                # if you want to hear your voice while recording
                # stream.write(data)
                self.frames.append(data)

            if self.making_speaking_avatar:
                viz.result.message = 'Check the video and audio outputs ...'

            # play video
            # if self.play and self.success:
            #     self.success, image = self.vidObj.read()
            #     print(self.count)
            #     self.count += 1
            #     if self.success:
            #         viz.result.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        self.disabled_time = max(self.disabled_time - viz.frame_delta, 0)


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
                print(self.path_sofar)


            self.cur_img = self.path_sofar
            imgui.end_popup()

#-----------------------------------------------------------------------------