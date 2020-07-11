"""
The main portion of this code was taken from OpenAI Gym and it is licensed under the following license:

The MIT License

Copyright (c) 2016 OpenAI (https://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
"""

# from pyformulas import screen
import matplotlib.pyplot as plt
import skimage.transform
from scipy.spatial.transform import Rotation as R
import skimage.color
from gym.envs.box2d.car_racing import *

from tqdm import tqdm

WINDOW_H = 1000
WINDOW_W = 1000


class CarRacingWrapper(CarRacing):
    def __init__(
        self,
        enable_linear_speed,
        enable_angular_speed,
        enable_abs,
        enable_steering,
        prerendered_data=None,
    ):
        """
        Wrapper for the car racing environment

        :param enable_linear_speed: whether to enable the linear speed feature
        :param enable_angular_speed: whether to enable the angular speed feature
        :param enable_abs: whether to enable the abs feature
        :param enable_steering: whether to enable the steering feature
        :param imgs: pre-rendered track images for each degree
        """
        super(CarRacingWrapper, self).__init__(verbose=0)
        self.enable_linear_speed = enable_linear_speed
        self.enable_angular_speed = enable_angular_speed
        self.enable_abs = enable_abs
        self.enable_steering = enable_steering

        if prerendered_data is not None:
            self.road_vertex_list = prerendered_data["road_vertex_list"]
            self.track = prerendered_data["track"]
            self.track_imgs = prerendered_data["track_imgs"]
            self.car = Car(self.world, *self.track[0][1:4])
        else:
            self.reset()
        # self.canvas = np.zeros((32, 32))
        # self.pyscreen = screen(self.canvas, "view")

    def export(self):
        return {
            "road_vertex_list": self.road_vertex_list,
            "track": self.track,
            "track_imgs": self.track_imgs,
        }

    def load_road(self):
        vertices = []
        colors = []
        current_color = (0.4, 0.8, 0.4, 1.0)
        colors.append(current_color)
        vertices.append((-PLAYFIELD, +PLAYFIELD, 0))
        colors.append(current_color)
        vertices.append((+PLAYFIELD, +PLAYFIELD, 0))
        colors.append(current_color)
        vertices.append((+PLAYFIELD, -PLAYFIELD, 0))
        colors.append(current_color)
        vertices.append((-PLAYFIELD, -PLAYFIELD, 0))

        current_color = (0.4, 0.9, 0.4, 1.0)
        k = PLAYFIELD / 20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                vertices.append((k * x + k, k * y + 0, 0))
                colors.append(current_color)
                vertices.append((k * x + 0, k * y + 0, 0))
                colors.append(current_color)
                vertices.append((k * x + 0, k * y + k, 0))
                colors.append(current_color)
                vertices.append((k * x + k, k * y + k, 0))
                colors.append(current_color)

        for poly, color in self.road_poly:
            for p in poly:
                colors.append((color[0], color[1], color[2], 1))
                vertices.append((p[0], p[1], 0))

        self.road_vertex_list = pyglet.graphics.vertex_list(
            len(vertices),
            ("v3f", tuple(e for v in vertices for e in v)),
            ("c4f", tuple(e for c in colors for e in c)),
        )

    def render_road(self):
        if self.road_vertex_list is None:
            self.load_road()
        self.road_vertex_list.draw(gl.GL_QUADS)

    def draw_crosshair(self, img, x, y):
        img[x, :, 0] = 255
        img[:, y, 0] = 255
        self.imshow(img)

    def crop_current(self):

        rot_mat = R.from_euler("z", +self.car.hull.angle, degrees=False)
        rot_car_x, rot_car_y, _ = rot_mat.apply(
            [self.car.hull.position[1], self.car.hull.position[0], 0]
        )
        car_x = 500 - int(rot_car_x)
        car_y = 500 + int(rot_car_y)

        img = self.track_imgs[int(math.degrees(-self.car.hull.angle)) % 360]
        # self.draw_crosshair(img, car_x, car_y)

        CROP_SIZE = 20
        crop = img[
            car_x - 24 : car_x + 8, car_y - 16 : car_y + 16, :,
        ]

        # self.pyscreen.update(crop)

        return crop

    def imshow(self, img):
        plt.imshow(img)
        plt.show()

    def render_whole_track(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label(
                "0000",
                font_size=36,
                x=20,
                y=WINDOW_H * 2.5 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(255, 255, 255, 255),
            )
            self.transform = rendering.Transform()

        self.transform.set_scale(1, 1)
        self.transform.set_translation(WINDOW_W / 2, WINDOW_H / 2)

        imgs = []

        for i in tqdm(range(360)):
            self.transform.set_rotation(math.radians(i))
            win = self.viewer.window
            win.switch_to()
            win.dispatch_events()

            win.clear()
            t = self.transform
            pixel_scale = 1
            if hasattr(win.context, "_nscontext"):
                pixel_scale = (
                    win.context._nscontext.view().backingScaleFactor()
                )  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

            gl.glViewport(0, 0, VP_W, VP_H)
            t.enable()
            self.render_road()
            t.disable()

            win.flip()
            gl.glFinish()

            image_data = (
                pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            )
            arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
            arr = arr.reshape(WINDOW_H, WINDOW_W, 4)
            arr = arr[::-1, :, 0:3]
            imgs.append(arr)

        return imgs

    def step(self, action):
        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self.crop_current()

        step_reward = 0
        done = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track):
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100

        return self.state, step_reward, done, {}

    def render(self, mode="human"):
        assert mode in ["human", "state_pixels", "rgb_array"]
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label(
                "0000",
                font_size=36,
                x=20,
                y=WINDOW_H * 2.5 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(255, 255, 255, 255),
            )
            self.transform = rendering.Transform()

        if "t" not in self.__dict__:
            return  # reset() not called yet

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
            WINDOW_W / 2
            - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            WINDOW_H / 4
            - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)),
        )
        self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode != "state_pixels")

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == "rgb_array":
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == "state_pixels":
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, "_nscontext"):
                pixel_scale = (
                    win.context._nscontext.view().backingScaleFactor()
                )  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        # self.render_indicators(WINDOW_W, WINDOW_H)

        if mode == "human":
            win.flip()
            return self.viewer.isopen

        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
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

        image = skimage.color.rgb2gray(self.state.astype(np.float32))
        if self.image_scaling != 1:
            assert (
                len(image) % self.image_scaling == 0
                and len(image[0]) % self.image_scaling == 0
            )
            image = skimage.transform.downscale_local_mean(
                image, (self.image_scaling, self.image_scaling)
            )

        return (
            image.reshape(1, 1, image.shape[0], image.shape[0]),
            np.concatenate(vectors, axis=0).astype(np.float32),
        )

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print(
                    "retry to generate track (normal if there are not many instances of this message)"
                )
        self.car = Car(self.world, *self.track[0][1:4])
        self.track_imgs = self.render_whole_track()

        return self.step(None)[0]


if __name__ == "__main__":
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    env = CarRacingWrapper(True, True, True, True, 1)
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, "/tmp/video-test", force=True)
    isopen = True
    while isopen:
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            steps += 1
            isopen = env.render()
    env.close()
