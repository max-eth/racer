'''
The main portion of this code was taken from OpenAI Gym and it is licensed under the following license:

The MIT License

Copyright (c) 2016 OpenAI (https://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
'''

import skimage.transform
import skimage.color
from gym.envs.box2d.car_racing import *


class CarRacingWrapper(CarRacing):
    def __init__(self, enable_linear_speed, enable_angular_speed, enable_abs, enable_steering, image_scaling):
        super(CarRacingWrapper, self).__init__(verbose=0)
        self.enable_linear_speed = enable_linear_speed
        self.enable_angular_speed = enable_angular_speed
        self.enable_abs = enable_abs
        self.enable_steering = enable_steering
        self.image_scaling = image_scaling
        self.reset()

    def render(self, mode='state_pixels'):
        assert mode in ['human', 'state_pixels', 'rgb_array']
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label('0000', font_size=36,
                                                 x=20, y=WINDOW_H * 2.5 / 40.00, anchor_x='left', anchor_y='center',
                                                 color=(255, 255, 255, 255))
            self.transform = rendering.Transform()

        if "t" not in self.__dict__: return  # reset() not called yet

        # zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)  # Animate zoom first second
        zoom = ZOOM * SCALE
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W / 2 - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            WINDOW_H / 4 - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)))
        self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode != "state_pixels")

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == 'rgb_array':
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == 'state_pixels':
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, '_nscontext'):
                pixel_scale = win.context._nscontext.view().backingScaleFactor()  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        #self.render_indicators(WINDOW_W, WINDOW_H)

        if mode == 'human':
            win.flip()
            return self.viewer.isopen

        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep='')
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def get_state(self):
        vectors = []
        if self.enable_linear_speed:
            linear_speed = np.sqrt(
                np.square(self.car.hull.linearVelocity[0])
                + np.square(self.car.hull.linearVelocity[1])
            )
            vectors.append(np.array([linear_speed]))

        if self.enable_angular_speed:
            vectors.append(np.array([self.car.hull.angularVelocity]))

        if self.enable_abs:
            vectors.append(np.array([self.car.wheels[i].omega for i in range(4)]))

        if self.enable_steering:
            vectors.append(np.array([self.car.wheels[0].joint.angle]))

        image = skimage.color.rgb2gray(self.state)
        if self.image_scaling != 1:
            assert (len(image) % self.image_scaling == 0 and len(image[0]) % self.image_scaling == 0)
            image = skimage.transform.downscale_local_mean(
                image, (self.image_scaling, self.image_scaling)
            )

        vectors.append(image.flatten())
        return np.concatenate(vectors, axis=0)
