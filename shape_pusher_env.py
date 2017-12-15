import os
import tempfile
from contextlib import contextmanager

import gym
import mujoco_py
import numpy as np
import six
import stl
from gym import spaces
from gym.envs.mujoco import mujoco_env


def create_pyramid_mesh(half_height=0.05, radius=0.05, num_sides=4):
    vertices = []
    for i in range(num_sides):
        angle = i * 2 * np.pi / num_sides
        vertex = [radius * np.cos(angle), radius * np.sin(angle), -half_height]
        vertices.append(vertex)
    vertices.append([0, 0, -half_height])
    vertices.append([0, 0, half_height])
    faces = []
    for i in range(num_sides):
        face = [i, num_sides, (i + 1) % num_sides]
        faces.append(face)
    for i in range(num_sides):
        face = [(i + 1) % num_sides, num_sides + 1, i]
        faces.append(face)
    vertices = np.array(vertices)
    faces = np.array(faces)
    mesh = stl.mesh.Mesh(np.zeros(len(faces), dtype=stl.mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mesh.vectors[i][j] = vertices[f[j], :]
    return mesh


def create_prism_mesh(half_height=0.05, radius=0.05, num_sides=4):
    vertices = []
    for h in [-half_height, half_height]:
        for i in range(num_sides):
            angle = i * 2 * np.pi / num_sides
            vertex = [radius * np.cos(angle), radius * np.sin(angle), h]
            vertices.append(vertex)
        vertices.append([0, 0, h])
    faces = []
    for i in range(num_sides):
        face = [i, num_sides, (i + 1) % num_sides]
        faces.append(face)
        face = [(i + 1) % num_sides + num_sides + 1, 2 * num_sides + 1, i + num_sides + 1]
        faces.append(face)
        face = [i, (i + 1) % num_sides, i + num_sides + 1]
        faces.append(face)
        face = [(i + 1) % num_sides + num_sides + 1, i + num_sides + 1, (i + 1) % num_sides]
        faces.append(face)
    vertices = np.array(vertices)
    faces = np.array(faces)
    mesh = stl.mesh.Mesh(np.zeros(len(faces), dtype=stl.mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mesh.vectors[i][j] = vertices[f[j], :]
    return mesh


class MJCModel(object):
    def __init__(self, name):
        self.name = name
        self.root = MJCTreeNode("mujoco").add_attr('model', name)

    @contextmanager
    def asfile(self):
        """
        Usage:

        model = MJCModel('reacher')
        with model.asfile() as f:
            print f.read()  # prints a dump of the model

        """
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.xml', delete=True) as f:
            self.root.write(f)
            f.seek(0)
            yield f

    def open(self):
        self.file = tempfile.NamedTemporaryFile(mode='w+b', suffix='.xml', delete=True)
        self.root.write(self.file)
        self.file.seek(0)
        return self.file

    def save(self, path):
        with open(path, 'w') as f:
            self.root.write(f)

    def close(self):
        self.file.close()


class MJCModelRegen(MJCModel):
    def __init__(self, name, regen_fn):
        super(MJCModelRegen, self).__init__(name)
        self.regen_fn = regen_fn

    def regenerate(self):
        self.root = self.regen_fn().root


class MJCTreeNode(object):
    def __init__(self, name):
        self.name = name
        self.attrs = {}
        self.children = []

    def add_attr(self, key, value):
        if isinstance(value, str):  # should be basestring in python2
            pass
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            value = ' '.join([str(val) for val in value])

        self.attrs[key] = value
        return self

    def __getattr__(self, name):
        def wrapper(**kwargs):
            newnode =  MJCTreeNode(name)
            for (k, v) in kwargs.items(): # iteritems in python2
                newnode.add_attr(k, v)
            self.children.append(newnode)
            return newnode
        return wrapper

    def dfs(self):
        yield self
        if self.children:
            for child in self.children:
                for node in child.dfs():
                    yield node

    def write(self, ostream, tabs=0):
        contents = ' '.join(['%s="%s"'%(k,v) for (k,v) in self.attrs.items()])
        if self.children:

            ostream.write('\t'*tabs)
            ostream.write('<%s %s>\n' % (self.name, contents))
            for child in self.children:
                child.write(ostream, tabs=tabs+1)
            ostream.write('\t'*tabs)
            ostream.write('</%s>\n' % self.name)
        else:
            ostream.write('\t'*tabs)
            ostream.write('<%s %s/>\n' % (self.name, contents))

    def __str__(self):
        s = "<"+self.name
        s += ' '.join(['%s="%s"'%(k,v) for (k,v) in self.attrs.items()])
        return s+">"


def create_shape_pusher(manip=None):
    if manip is None:
        manip = 'sphere'

    mjcmodel = MJCModel('shape_pusher')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(timestep="0.01", gravity="0 0 -9.81", iterations="20", integrator="Euler")

    default = mjcmodel.root.default()
    default.joint(armature="0.04", damping="1", limited="true")
    default.geom(friction=".8 .1 .1", density="300", margin="0.002", condim="1", contype="1", conaffinity="1", rgba="0.7 0.7 0 1")

    worldbody = mjcmodel.root.worldbody()
    worldbody.light(diffuse=".5 .5 .5", pos="0 0 3", dir="0 0 -1")

    worldbody.geom(name="table", type="plane", pos="0 0 0", size=".5 .5 .1", rgba="0.5 0.5 0.5 1")

    border = worldbody.body(name="border", pos="0 0 0.025")
    border_front = border.body(name="border_front", pos="0 -.525 0")
    border_front.geom(type="box", size=".55 .025 .025", rgba="0.9 0.4 0.6 1")
    border_rear = border.body(name="border_rear", pos="0 .525 0")
    border_rear.geom(type="box", size=".55 .025 .025", rgba="0.9 0.4 0.6 1")
    border_right = border.body(name="border_right", pos=".525 0 0")
    border_right.geom(type="box", size=".025 .5 .025", rgba="0.9 0.4 0.6 1")
    border_left = border.body(name="border_left", pos="-.525 0 0")
    border_left.geom(type="box", size=".025 .5 .025", rgba="0.9 0.4 0.6 1")

    if manip == 'sphere':
        base = worldbody.body(name="base", pos="0 0 0.05")
        base.geom(type="sphere", size=".05", rgba="0.0 0.4 0.6 1")
        base.joint(name="base_x", type="slide", axis="1 0 0", range="-.5 .5")
        base.joint(name="base_y", type="slide", axis="0 1 0", range="-.5 .5")
    else:
        base = worldbody.body(name="base", pos="0 0 0.05")
        base.geom(type="sphere", size=".02", rgba="0.9 0.4 0.6 1")
        base.joint(name="base_x", type="slide", axis="1 0 0", range="-.5 .5")
        base.joint(name="base_y", type="slide", axis="0 1 0", range="-.5 .5")
        gripper = base.body(name="gripper", pos="0 0 0")
        gripper.geom(type="capsule", size=".02", fromto="0.0 -0.1 0.0 0.0 +0.1 0.0", rgba="0.0 0.4 0.6 1")
        gripper.geom(type="capsule", size=".02", fromto="0.0 -0.1 0.0 0.1 -0.1 0.0", rgba="0.0 0.4 0.6 1")
        gripper.geom(type="capsule", size=".02", fromto="0.0 +0.1 0.0 0.1 +0.1 0.0", rgba="0.0 0.4 0.6 1")
        gripper.joint(name="base_angle", type="hinge", axis="0 0 1", limited="false")

    obj0 = worldbody.body(name="obj0")
    obj0.geom(name="obj0_geom", type="box", size=".05 .05 .05", rgba="1 1 1 1")

    obj1 = worldbody.body(name="obj1")
    obj1.geom(name="obj1_geom", type="sphere", size=".05", rgba="1 1 1 1")

    obj2 = worldbody.body(name="obj2")
    obj2.geom(name="obj2_geom", type="cylinder", size=".05 .05", rgba="1 1 1 1")

    obj3 = worldbody.body(name="obj3")
    obj3.geom(name="obj3_geom0", type="box", size=".05 .025 .05", pos="0 0 0", rgba="1 1 1 1")
    obj3.geom(name="obj3_geom1", type="box", size=".025 .0125 .05", pos="0 -0.0375 0", rgba="1 1 1 1")
    obj3.geom(name="obj3_geom2", type="box", size=".025 .0125 .05", pos="0 0.0375 0", rgba="1 1 1 1")

    obj4 = worldbody.body(name="obj4")
    obj4.geom(name="obj4_geom0", type="capsule", size=".025 .025 .05", pos="-0.025 0 0", rgba="1 1 1 1")
    obj4.geom(name="obj4_geom1", type="capsule", size=".025 .025 .05", pos="0.025 0 0", rgba="1 1 1 1")
    obj4.geom(name="obj4_geom2", type="capsule", size=".025 .025 .05", pos="0 -0.025 0", rgba="1 1 1 1")
    obj4.geom(name="obj4_geom3", type="capsule", size=".025 .025 .05", pos="0 0.025 0", rgba="1 1 1 1")

    obj5 = worldbody.body(name="obj5")
    obj5.geom(name="obj5_geom0", type="cylinder", size=".025 .05", pos="-0.025 0 0", rgba="1 1 1 1")
    obj5.geom(name="obj5_geom1", type="cylinder", size=".025 .05", pos="0.025 0 0", rgba="1 1 1 1")
    obj5.geom(name="obj5_geom2", type="cylinder", size=".025 .05", pos="0 -0.025 0", rgba="1 1 1 1")
    obj5.geom(name="obj5_geom3", type="cylinder", size=".025 .05", pos="0 0.025 0", rgba="1 1 1 1")

    obj6 = worldbody.body(name="obj6")
    obj6.geom(name="obj6_geom", type="mesh", mesh="prism3", rgba="1 1 1 1")

    obj7 = worldbody.body(name="obj7")
    obj7.geom(name="obj7_geom", type="mesh", mesh="prism5", rgba="1 1 1 1")

    obj8 = worldbody.body(name="obj8")
    obj8.geom(name="obj8_geom", type="mesh", mesh="prism6", rgba="1 1 1 1")

    for i, obj in enumerate([obj0, obj1, obj2, obj3, obj4, obj5, obj6, obj7, obj8]):
        obj.attrs['pos'] = "%.1f 0.6 0.05" % (i * 0.1 - 0.4)
        obj.joint(name="obj%d_joint" % i, type="free", limited="false")

    asset = mjcmodel.root.asset()
    asset.mesh(file=os.path.join(gym.__path__[0], 'envs/mujoco/assets/prism3.stl'), name="prism3")
    asset.mesh(file=os.path.join(gym.__path__[0], 'envs/mujoco/assets/prism5.stl'), name="prism5")
    asset.mesh(file=os.path.join(gym.__path__[0], 'envs/mujoco/assets/prism6.stl'), name="prism6")

    actuator = mjcmodel.root.actuator()

    actuator.motor(joint="base_x", ctrlrange="-1.0 1.0", ctrllimited="true")
    actuator.motor(joint="base_y", ctrlrange="-1.0 1.0", ctrllimited="true")
    if manip != 'sphere':
        actuator.motor(joint="base_angle", ctrlrange="-1.0 1.0", ctrllimited="true")

    return mjcmodel


class ShapePusherEnv(mujoco_env.MujocoEnv):
    def __init__(self, write_assets=True):
        if write_assets:
            # generate and save pyramids
            for num_sides in range(3, 7):
                pyramid_mesh = create_pyramid_mesh(num_sides=num_sides)
                pyramid_mesh.save(os.path.join(gym.__path__[0], 'envs/mujoco/assets/pyramid%d.stl' % num_sides))

            # generate and save prisms
            for num_sides in range(3, 7):
                prism_mesh = create_prism_mesh(num_sides=num_sides)
                prism_mesh.save(os.path.join(gym.__path__[0], 'envs/mujoco/assets/prism%d.stl' % num_sides))

            # generate and save pusher
            model = create_shape_pusher()
            model.save(os.path.join(gym.__path__[0], 'envs/mujoco/assets/shape_pusher.xml'))

        # MujocoEnv's superclass init
        self.frame_skip = 5
        self.model = mujoco_py.MjModel(os.path.join(gym.__path__[0], 'envs/mujoco/assets/shape_pusher.xml'))
        self.data = self.model.data

        self.width = 256
        self.height = 256
        self.viewer = mujoco_py.MjViewer(init_width=self.width, init_height=self.height)
        self.viewer.start()
        self.viewer.set_model(self.model)
        self.viewer_setup()

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.action_space = spaces.Box(-1, 1, shape=(2,))

        joints_range = self.model.jnt_range.copy()
        self.observation_space = {'image': spaces.Box(0, 255, shape=(self.width, self.height)),
                                  'manip_xy': spaces.Box(joints_range[:2, 0], joints_range[:2, 1]),
                                  'obj_pose': spaces.Box(-np.inf, np.inf, shape=(7,))}

        self._seed()

        self.num_objs = len([body_name for body_name in self.model.body_names if body_name.startswith(six.b('obj'))])
        self.orig_body_mass = self.model.body_mass.copy()
        self.obj_idx = None

    def do_simulation(self, vel, n_frames):
        np.clip(vel, self.action_space.low, self.action_space.high, out=vel)
        qvel = np.array(self.model.data.qvel)
        qvel[:len(vel)] = vel[:, None]
        self.model.data.qvel = qvel
        for _ in range(n_frames):
            self.model.step()

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        reward = 0.0
        done = False
        return obs, reward, done, dict()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 1.25
        self.viewer.cam.azimuth = 0
        self.viewer.cam.elevation = -90
        self.viewer.cam.lookat[:] = [0] * 3

    def reset_model(self):
        manip_xy = np.random.uniform(-0.4, 0.4, 2)
        objs_pose = np.empty((self.num_objs, 7))
        objs_pose[:, 0] = np.linspace(-0.4, 0.4, num=self.num_objs)
        objs_pose[:, 1:] = [0.6, 0.05, 1, 0, 0, 0]

        self.obj_idx = np.random.randint(0, self.num_objs)
        obj_xy = None
        while obj_xy is None or np.linalg.norm(obj_xy - manip_xy) < 0.1:
            obj_xy = np.random.uniform(-0.4, 0.4, 2)
        objs_pose[self.obj_idx, :2] = obj_xy
        qpos = np.append(manip_xy, objs_pose)

        geom_rgba = self.model.geom_rgba.copy()
        obj_rgba = np.random.uniform(0.0, 1.0, 3)
        for geom_idx, body_idx in enumerate(np.squeeze(self.model.geom_bodyid, axis=1)):
            body_name = self.model.body_names[body_idx]
            if body_name == six.b('obj%d' % self.obj_idx):
                geom_rgba[geom_idx, :3] = obj_rgba
                geom_rgba[geom_idx, 3] = 1.0
            elif body_name.startswith(six.b('obj')):
                geom_rgba[geom_idx, 3] = 0.0

        dof_damping = self.model.dof_damping.copy()
        obj_damping = 10 ** np.random.uniform(0.0, 1.0)
        for dof_idx, body_idx in enumerate(np.squeeze(self.model.dof_bodyid, axis=1)):
            body_name = self.model.body_names[body_idx]
            if body_name == six.b('obj%d' % self.obj_idx):
                dof_damping[dof_idx] = obj_damping
            elif body_name.startswith(six.b('obj')):
                dof_damping[dof_idx] = 1.0

        body_mass = self.model.body_mass.copy()
        obj_mass_multiplier = 10 ** np.random.uniform(0.0, 1.0)
        for body_idx, body_name in enumerate(self.model.body_names):
            if body_name == six.b('obj%d' % self.obj_idx):
                body_mass[body_idx] = obj_mass_multiplier * self.orig_body_mass[body_idx]
            elif body_name.startswith(six.b('obj')):
                body_mass[body_idx] = self.orig_body_mass[body_idx]

        self.model.data.qpos = qpos
        self.model.geom_rgba = geom_rgba
        self.model.dof_damping = dof_damping
        self.model.body_mass = body_mass

        self.model._compute_subtree()  # pylint: disable=W0212
        self.model.forward()
        return self._get_obs()

    def _get_obs(self):
        image = self.render('rgb_array')
        manip_xy, objs_pose = np.split(self.model.data.qpos, [2])
        manip_xy = np.squeeze(manip_xy, axis=1)
        objs_pose = objs_pose.reshape((-1, 7))
        obj_pose = objs_pose[self.obj_idx]
        return {'image': image,
                'manip_xy': manip_xy,
                'obj_pose': obj_pose}
