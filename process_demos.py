import gym
import cv2
import os
import sys
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import pickle
from glob import glob
from tqdm import tqdm
import time as timer

import adept_envs
from adept_envs.utils.parse_mjl import parse_mjl_logs, viz_parsed_mjl_logs

demo_dir = 'kitchen_demos_multitask/'
render = 'offscreen'
CAMERA_ID = 1
FRAME_SKIP = 40
FPS = 30.
FOURCC = cv2.VideoWriter_fourcc(*'XVID')
VIDEO_EXT = 'avi'

# RENDER_SIZE = (64, 64)
RENDER_SIZE = (128, 128)
# RENDER_SIZE = (120, 160)
# RENDER_SIZE = (256, 256)
# RENDER_SIZE = (240, 320)
# RENDER_SIZE = (480, 640)
# RENDER_SIZE = (1920, 2560)

out_dir = 'workdir/'
# which_set = 'postcorl'
# which_set = 'friday'
which_set = None

obj_indices_map = { # https://relay-policy-learning.github.io/assets/mp4/collage_RPL_success2.mp4
    'microwave': [22], # M
    'kettle': [23, 24, 26, 29], # K
    'kettle_move': [25, 27, 28],
    'bottomknob': [11, 12], # BB
    'topknob': [15, 16], # TB
    'switch': [17, 18], # L
    'slide': [19], # S
    'hinge': [21], # H
}
indices_obj_map = {}
for k, v in obj_indices_map.items():
    for x in v:
        assert x not in indices_obj_map.keys()
        indices_obj_map[x] = k
print(obj_indices_map)
print(indices_obj_map)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
else:
    pass
    # raise Exception('out dir exists')


env = gym.make('kitchen_relax-v1', camera_id=CAMERA_ID)
env = env.env # remove TimeLimit wrapper

random.seed(0)
np.random.seed(0)
env.seed(0)


def save_video(render_buffer, filepath):
    vw = cv2.VideoWriter(filepath, 
                         FOURCC, FPS, RENDER_SIZE[::-1])
    for frame in render_buffer:
        frame = frame.copy()
        # frame = cv2.resize(frame, RENDER_SIZE[::-1])
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        vw.write(frame)
    vw.release()


def render_demo(data, use_physics=False):
    render_skip = max(1, round(1. / \
        (FPS * env.sim.model.opt.timestep * env.frame_skip)))
    t0 = timer.time()

    init_qpos = data['qpos'][0].copy()
    init_qvel = data['qvel'][0].copy()

    if use_physics:
        # initialize
        env.reset()
        act_mid = env.act_mid
        act_rng = env.act_amp

        # prepare env
        env.sim.data.qpos[:] = init_qpos
        env.sim.data.qvel[:] = init_qvel
        env.sim.forward()

    render_buffer = []
    path = dict(observations=[], actions=[])
    N = data['ctrl'].shape[0]
    i_frame = 0

    while True:
        if use_physics:
            # Reset every time step
            # if i_frame % 1 == 0:
            #     qp = data['qpos'][i_frame].copy()
            #     qv = data['qvel'][i_frame].copy()
            #     env.sim.data.qpos[:] = qp
            #     env.sim.data.qvel[:] = qv
            #     env.sim.forward()

            obs = env._get_obs()
        else:
            if i_frame == N: # w/o physics, not getting terminal obseration, so will have one less obs than w/ physics
                break

            env.sim.data.qpos[:] = data['qpos'][i_frame].copy()
            env.sim.data.qvel[:] = data['qvel'][i_frame].copy()
            env.sim.forward()

        if i_frame % render_skip == 0:
            if render == 'onscreen':
                env.mj_render()
            elif render == 'offscreen':
                curr_frame = env.render(mode='rgb_array', height=RENDER_SIZE[0], width=RENDER_SIZE[1])
                render_buffer.append(curr_frame)
                # return render_buffer
            else:
                raise ValueError
            print(f'{i_frame}', end=', ', flush=True)

        if use_physics:
            path['observations'].append(obs)
            if i_frame == N: # for adding final observation
                break

            # Construct the action
            # ctrl = (data['qpos'][i_frame + 1][:9] - obs[:9]) / (env.skip * env.model.opt.timestep)
            ctrl = (data['ctrl'][i_frame] - obs[:9])/(env.skip*env.model.opt.timestep)
            act = (ctrl - act_mid) / act_rng
            act = np.clip(act, -0.999, 0.999)

            next_obs, reward, done, env_info = env.step(act)

            path['actions'].append(act)

        i_frame += 1

    print('')
    print("physics = %i, time taken = %f" % (use_physics, timer.time() - t0))

    if use_physics:
        path['observations'] = np.vstack(path['observations'])
        path['actions'] = np.vstack(path['actions'])
        return render_buffer, path
    else:
        return render_buffer


def process_demo(filepath,
                 view_demo=True, # view demos (physics ignored)
                 playback_demo=False, # playback demos and get data(physics respected)
                 save_data=True,
                 graph=False):
    _, task, fn = filepath.split('/')
    f = fn.split('.mjl')[0]
    outpath = os.path.join(out_dir, task, f)

    data = parse_mjl_logs(filepath, FRAME_SKIP)

    fn_meta = f'c{CAMERA_ID}h{RENDER_SIZE[0]}w{RENDER_SIZE[1]}'

    if view_demo:
        render_buffer = render_demo(data, use_physics=False)
        save_video(render_buffer, outpath + f'_view{fn_meta}.{VIDEO_EXT}')

    if playback_demo:
        try:
            render_buffer, path = render_demo(data, use_physics=True)
        except Exception as e:
            print(f'skipped {filepath}, {e}')
            return -1
        data['path'] = path
        save_video(render_buffer, outpath + f'_playback{fn_meta}.{VIDEO_EXT}')

    if save_data:
        pickle.dump(data, open(outpath + '.pkl', 'wb'))

    if graph:
        viz_parsed_mjl_logs(data)


def plot_task(task, demos, task_plots_dir=None):
    data = [parse_mjl_logs(demo, FRAME_SKIP) for demo in demos]

    fig, axs = plt.subplots(7, 3, constrained_layout=True)
    for o in range(7):
        for d in data:
            for i in range(3):
                j = 3*o + i + env.n_jnt
                ax = axs[o][i]
                ax.plot(d['time'], d['qpos'][..., j])
                obj = indices_obj_map[j] if j in indices_obj_map.keys() else ''
                title = f"{j}: {obj}"
                ax.set_title(title)

    fig.suptitle(task)
    if task_plots_dir is None:
        plt.show()
    else:
        fig.set_size_inches(15, 10)
        fig.savefig(os.path.join(task_plots_dir, f'{task}.png'), dpi=300)
        plt.close(fig)


def fit_obj_goal_model(obj, demos):
    data = [parse_mjl_logs(demo, FRAME_SKIP) for demo in demos]

    indices_data = {}
    for i in obj_indices_map[obj]:
        l = []
        for d in data:
            l.append(d['qpos'][-1, i])
        indices_data[i] = l

    # TODO: take IQR of trajectories? 
    # for microwave:
    # median: array([-0.64916694]
    # mean: array([-0.41988434])

    # model = {}
    # for i in obj_indices_map[obj]:
    #     i_data = np.array(indices_data[i])
    #     mean = np.mean(i_data)
    #     std = np.std(i_data)
    #     model[i] = (mean, std)

    combined = []
    for i in obj_indices_map[obj]:
        combined.append(np.array(indices_data[i]))
    combined = np.vstack(combined)
    mean = np.mean(combined, axis=1)
    cov = np.cov(combined)
    if len(obj_indices_map[obj]) == 1:
        cov = cov[np.newaxis][np.newaxis]
    model = (mean, cov)
    return model


if __name__ == '__main__':
    tasks = os.listdir(demo_dir)
    demos = glob(demo_dir + '*/*.mjl')

    if which_set is not None:
        tasks = list(filter(lambda x: which_set in x, tasks))
        demos = list(filter(lambda x: which_set in x, demos))

    print(f'tasks: {tasks}')
    print(f'num tasks: {len(tasks)}')
    print(f'num demos: {len(demos)}')
    print()

    for task in tasks:
        if not os.path.exists(os.path.join(out_dir, task)):
            os.makedirs(os.path.join(out_dir, task))

    # task_plots_dir = os.path.join(out_dir, 'task_plots')
    # if not os.path.exists(task_plots_dir):
    #     os.makedirs(task_plots_dir)
    # for i, task in enumerate(tasks):
    #     print(f'plotting task {i}: {task}')
    #     task_demos = list(filter(lambda x: task in x, demos))
    #     plot_task(task, task_demos, task_plots_dir=task_plots_dir)
    # exit()

    # for obj in ['microwave', 'kettle', 'bottomknob', 'topknob', 'switch', 'slide', 'hinge']:
    #     obj_demos = list(filter(lambda x: obj in x, demos))
    #     model = fit_obj_goal_model(obj, demos)
    #     print(f'{obj}: {model}')
    # exit()

    # print = tqdm.write
    for filepath in tqdm(demos, file=sys.stdout):
        # if 'friday_microwave_kettle_bottomknob_hinge' not in filepath: continue
        print(filepath)
        process_demo(filepath)
        # break