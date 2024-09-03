import argparse
import os
import random
import sys
import time
from pathlib import Path

import shutil
from random import randint

import gym
import numpy as np
from PIL import Image
import pandas as pd
from typing import Tuple
import config
from agents.random_agent import RandomAgent
from agents.supervised_agent import SupervisedAgent
from config import NUM_CONTROL_NODES, MAX_ANGLE, NUM_SAMPLED_POINTS
from envs.udacity.config import MIN_SPEED_UDACITY, MAX_SPEED_UDACITY
from envs.udacity.udacity_gym_env import UdacityGymEnv
from global_log import GlobalLog
from test_generators.one_plus_one_test_generator import OnePlusOneTestGenerator
#from thirdeye.heatmap_generator import compute_heatmap
from attention_manager_2 import AttentionManager

from utils.randomness import set_random_seed
from utils.road_utils import get_closest_control_point, get_closest_previous_control_point

import csv_logger

import gc
import tracemalloc



parser = argparse.ArgumentParser()
parser.add_argument('--udacity-exe-path', help="Path to the udacity simulator executor", type=str, default=None)
parser.add_argument('--seed', help='Random seed', type=int, default=-1)
parser.add_argument('--agent-type', help="Agent type", type=str, choices=['random', 'supervised'], default='random')
parser.add_argument('--add-to-port', help="Modify default simulator port", type=int, default=-1)
parser.add_argument('--num-episodes', help="Number of tracks to generate", type=int, default=3)
parser.add_argument('--num-control-nodes',
                    help="Number of control nodes of the generated road (only valid with random generator)", type=int,
                    default=NUM_CONTROL_NODES)
parser.add_argument('--max-angle', help="Max angle of a curve of the generated road (only valid with random generator)",
                    type=int, default=MAX_ANGLE)
parser.add_argument('--num-spline-nodes',
                    help="Number of points to sample among control nodes of the generated road (only valid with random generator)",
                    type=int, default=NUM_SAMPLED_POINTS)
parser.add_argument('--model-path', help="Path to agent model with extension (only if agent_type == 'supervised')",
                    type=str, default=None)
parser.add_argument('--mutation-type', help="XAI to 'high' or 'low' attention direction, or 'random'",
                    type=str, default='RANDOM')
parser.add_argument('--mutation-method', help="mutation to 'high' or 'low' attention direction, or just 'random'",
                    type=str, choices=['random', 'attention_same', 'attention_opposite'], default='random')

args = parser.parse_args()


def perform_one_test(seed: int = 0,
                     mutation_type: str = args.mutation_type,
                     mutation_method: str = args.mutation_method) -> bool:
    """
    perform one test with one plus one generator, where the control point selection is either XAI guided or random.
    Params:
        mutation_type: str = "XAI" or "RANDOM"
        mutation_method: = 'random', 'attention_same', 'attention_opposite'
    """

    time_start = time.strftime('%m-%d-%H-%M', time.localtime())
    logger = GlobalLog("main")
    run_path = os.path.join(time_start + '-' + mutation_type + '-seed=' + str(seed))

    # 1 plus 1 test generator
    test_generator = OnePlusOneTestGenerator(map_size=250, num_control_nodes=args.num_control_nodes)

    env = UdacityGymEnv(
        seed=seed,
        test_generator=test_generator,
        exe_path=args.udacity_exe_path,
    )

    if args.agent_type == "random":
        agent = RandomAgent(env=env)
    elif args.agent_type == "supervised":
        assert os.path.exists(args.model_path), "Model path {} does not exist".format(args.model_path)
        agent = SupervisedAgent(
            env=env,
            model_path=args.model_path,
            min_speed=MIN_SPEED_UDACITY,
            max_speed=MAX_SPEED_UDACITY
        )
    else:
        raise RuntimeError("Unknown agent type: {}".format(args.agent_type))

    times_elapsed = []
    # episode_lengths = []
    episode_count = 0
    # success_sum = 0
    speed = 0.0
    mutation_point = None
    direction = None
    """
    method:
        None - random
        "orthogonal_curve" - orthogonal to the curve and increase the curvature
        "orthogonal_random" - orthogonal to the curve and randomly select left or right
        "orthogonal_L" - orthogonal to the curve and move to left
        "orthogonal_R" - orthogonal to the curve and move to right
        "directional" - given direction
    """
    while episode_count < args.num_episodes:
        done = False
        frame_id = 0
        start_time_float = time.perf_counter()
        # generate track
        if mutation_method == "attention_same" and direction is not None:
            if direction == "L":
                mutation_direction = "orthogonal_L" # "orthogonal_R"
            elif direction == "R":
                mutation_direction = "orthogonal_R" # "orthogonal_L"
            else:
                mutation_direction = "orthogonal_curve"

        elif mutation_method == "attention_opposite" and direction is not None:
            if direction == "L":
                mutation_direction = "orthogonal_R" # "orthogonal_R"
            elif direction == "R":
                mutation_direction = "orthogonal_L" # "orthogonal_L"
            else:
                mutation_direction = "orthogonal_curve"
        elif mutation_method == "random":
            mutation_direction = "orthogonal_random"
        else:
            mutation_direction = None

        if mutation_point is not None:
            logger.debug("episode_count %d, mutation_point: %d, direction: %s"
                  % (episode_count, mutation_point, mutation_direction))
        
        time.sleep(1)

        obs = env.reset(mutation_info=[mutation_point, mutation_direction],
                        skip_generation=False)
        time.sleep(1)

        # TODO cases that unable to generate a valid road

        # these are the interpolated road points
        # print(env.executor.current_track.get_concrete_representation())

        # these are the original control points
        control_points = env.executor.current_track.get_control_points()

        # TODO: keep mutating until misbehaviour, or a timeout
        while not done:
            action = agent.predict(obs=obs, speed=speed)
            # clip action to avoid out of bound errors
            if isinstance(env.action_space, gym.spaces.Box):
                action = np.clip(action, env.action_space.low, env.action_space.high)

            # obs is the image, info contains the road and the position of the car
            obs, done, info = env.step(action)

            # car position
            car_position = info['pos']
            # control points 0, 8, 9 are always not considered
            # TODO: refine get_closest_control_point to avoid useless computations
            closest = get_closest_control_point(car_position, control_points)
            # the second last cp is the end of the road. If closest is the last point then it is out of the road
            closest_previous = get_closest_previous_control_point(car_position, control_points)

            #logger.debug(f"cloest CP {closest}, previous CP {closest_previous}")

            """if closest == args.num_control_nodes:
                logger.info("Past the finish line, end of the game")
                done = True"""

            # logger.debug(f'info {info} closest control point is #{closest}/{len(control_points)}')

            # create directory to save the simulation log

            episode_sub_path = os.path.join('simulations', run_path, 'episode' + str(episode_count))

            if not Path(episode_sub_path).exists():
                Path(episode_sub_path).mkdir(parents=True, exist_ok=True)

            # save the image
            filename = os.path.join(episode_sub_path, 'img'
                                    + str(frame_id)
                                    + '-'
                                    + str(info['pos'][0])
                                    + '-'
                                    + str(info['pos'][1])
                                    + '-'
                                    + str(closest_previous)
                                    + ".jpg")
            Image.fromarray(obs).save(filename)

            speed = 0.0 if info.get("speed", None) is None else info.get("speed", None)

            local_log_info = {
                        "frame_id": frame_id,
                        "img": filename,
                        "car_x": info['pos'][0],
                        "car_y": info['pos'][1],
                        "closest_cp": closest_previous,
                        "speed": speed,
                        "cte": info['cte'],
                        "is_success": info['is_success'],
                        'throttle':  info['throttle'],
                        'steering':  info['steering']
            }
            # TODO log mutation_point
            csv_logger.episode_logger(episode_sub_path, local_log_info)

            frame_id += 1
            if done:
                env.close()
                time.sleep(5)

                global_log_info = {
                    "Time": time_start,
                    "episode id": episode_count,
                    "episode length": frame_id,
                    "is success": info['is_success'],
                    'seed': seed,
                    'mutation type': mutation_type,
                    'mutation method': mutation_method,
                    'episode_sub_path': episode_sub_path,
                    'num episodes': str(args.num_episodes),
                    'max speed': MAX_SPEED_UDACITY,
                    'num control nodes': args.num_control_nodes,
                    'max angle': args.max_angle,
                    "mutation point": mutation_point,
                }

                current_track = env.get_track_points()
                for i, cp in enumerate(current_track):
                    global_log_info[f'CP_{i+1}'] = str(cp[0]) + "_" + str(cp[1])

                csv_logger.episode_logger(os.path.join("simulations", "global_log.csv"), global_log_info)

                times_elapsed.append(time.perf_counter() - start_time_float)
                logger.debug('Episode #{}'.format(episode_count + 1))
                logger.debug("Episode Length: {}".format(frame_id))
                logger.debug("Successfully finished track: {}".format(info['is_success']))

                # episode_lengths.append(frame_id)
                episode_count += 1

                if info['is_success'] == 1:
                    # success_sum += 1
                    if mutation_type.upper() == "XAI":

                        # run ThirdEye to compute the hottest point in the simulation
                        attention_manager = AttentionManager(simulation_name=episode_sub_path,
                                                             index_step=10,
                                                             attention_type="Faster-ScoreCAM")
                        mutation_point, direction = attention_manager.compute_attention_maps()
                        # mutation_point = attention_manager.control_point_selection()

                        # attention_manager.save_csv()
                        # attention_manager.delete()

                        logger.debug("closest control point to the hottest point is %d\n" % mutation_point)
                    elif mutation_type.upper() == "RANDOM":
                        mutation_point = random.randint(1, args.num_control_nodes-2)
                        logger.debug("Random selected control point is %d\n" % mutation_point)
                else:
                    logger.debug("Misbehavior detected. Mutation finished.")
                    # logger.debug("Success rate: {:.2f}".format(success_sum / episode_count))
                    logger.debug("Mean time elapsed: {:.2f}s".format(np.mean(times_elapsed)))

                    #env.reset(mutation_point=mutation_point, skip_generation=False)
                    time.sleep(5)
                    env.close()
                    time.sleep(5)
                    if episode_count == 1:
                        """
                        Failed on the first try so there is no road mutation.
                        We remove this result since it is meaningless for our research
                        """
                        data_folder = os.path.join('simulations', run_path)
                        shutil.rmtree(data_folder)
                        print("Failed on the first try, remove folder " + data_folder)
                        gc.collect()
                        logger.debug("Release memory")
                        time.sleep(10)
                        return False
                    else:
                        gc.collect()
                        logger.debug("Release memory")
                        time.sleep(10)
                        return True

    # logger.debug("Success rate: {:.2f}".format(success_sum / episode_count))
    logger.debug("Mean time elapsed: {:.2f}s".format(np.mean(times_elapsed)))

    # env.reset(mutation_point=mutation_point, skip_generation=False)

    time.sleep(5)
    env.close()

    time.sleep(5)
    # gc.collect()
    # logger.debug("Release memory") # DO NOT HELPS
    # time.sleep(10)

    return True


def main(number_of_test=1):
    args.agent_type = 'supervised'
    args.udacity_exe_path = './sim/udacity_sim_linux/udacity_sim_linux.x86_64'
    args.model_path = './models/udacity-dave2.h5'
    args.num_episodes = 40

    #args.num_control_nodes = 8

    if args.seed == -1:
        args.seed = np.random.randint(2 ** 32 - 1)
    #tracemalloc.start()

    for i in range(number_of_test):

        # set seed out of the loop
        print(f"Test IDX {i} seed {args.seed + i} start")
        print(f"--------------{args.mutation_type} test generation--------------")
        seed_ = args.seed + i
        set_random_seed(seed=seed_)
        result = perform_one_test(seed=seed_, mutation_type=args.mutation_type, mutation_method=args.mutation_method)
        """if not result:
            print("switch to the next seed")
            continue"""


if __name__ == "__main__":
    num_of_test = 1
    main(num_of_test)
