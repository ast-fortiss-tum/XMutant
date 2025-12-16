import argparse
import csv
import os
import sys
import time
from pathlib import Path

import gym
import numpy as np
import pandas as pd
from agents.random_agent import RandomAgent
from agents.supervised_agent import SupervisedAgent
from config import MAX_ANGLE, NUM_CONTROL_NODES, NUM_SAMPLED_POINTS
from envs.udacity.config import MAX_SPEED_UDACITY, MIN_SPEED_UDACITY
from envs.udacity.udacity_gym_env import UdacityGymEnv
from global_log import GlobalLog
from PIL import Image
from test_generators.one_plus_one_test_generator import OnePlusOneTestGenerator
from thirdeye.heatmap_generator import compute_heatmap
from utils.randomness import set_random_seed
from utils.road_utils import get_closest_control_point

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs")
parser.add_argument(
    "--udacity-exe-path", help="Path to the udacity simulator executor", type=str, default=None
)
parser.add_argument("--seed", help="Random seed", type=int, default=-1)
parser.add_argument(
    "--agent-type", help="Agent type", type=str, choices=["random", "supervised"], default="random"
)
parser.add_argument("--add-to-port", help="Modify default simulator port", type=int, default=-1)
parser.add_argument("--num-episodes", help="Number of tracks to generate", type=int, default=3)
parser.add_argument(
    "--num-control-nodes",
    help="Number of control nodes of the generated road (only valid with random generator)",
    type=int,
    default=NUM_CONTROL_NODES,
)
parser.add_argument(
    "--max-angle",
    help="Max angle of a curve of the generated road (only valid with random generator)",
    type=int,
    default=MAX_ANGLE,
)
parser.add_argument(
    "--num-spline-nodes",
    help="Number of points to sample among control nodes of the generated road (only valid with random generator)",
    type=int,
    default=NUM_SAMPLED_POINTS,
)
parser.add_argument(
    "--model-path",
    help="Path to agent model with extension (only if agent_type == 'supervised')",
    type=str,
    default=None,
)
args = parser.parse_args()

args.agent_type = "supervised"
args.udacity_exe_path = "./sim/udacity_sim_linux/udacity_sim_linux.x86_64"
args.model_path = "./models/udacity-dave2.h5"
args.num_episodes = 8
args.seed = 0
args.num_control_nodes = 10


if __name__ == "__main__":

    # TODO: make it a parser argument?
    GENERATE_CSV = True

    time_start = time.strftime("%y-%m-%d-%H-%M", time.localtime())
    folder = args.folder
    logger = GlobalLog("main")

    if args.seed == -1:
        args.seed = np.random.randint(2**32 - 1)

    set_random_seed(seed=args.seed)

    # 1 plus 1 vit_model generator
    test_generator = OnePlusOneTestGenerator(map_size=250, num_control_nodes=args.num_control_nodes)

    env = UdacityGymEnv(
        seed=args.seed,
        test_generator=test_generator,
        exe_path=args.udacity_exe_path,
    )

    if args.agent_type == "random":
        agent = RandomAgent(env=env)
    elif args.agent_type == "supervised":
        assert os.path.exists(args.model_path), "Model path {} does not exist".format(
            args.model_path
        )
        agent = SupervisedAgent(
            env=env,
            model_path=args.model_path,
            min_speed=MIN_SPEED_UDACITY,
            max_speed=MAX_SPEED_UDACITY,
        )
    else:
        raise RuntimeError("Unknown agent type: {}".format(args.agent_type))

    times_elapsed = []
    episode_lengths = []
    episode_count = 0
    success_sum = 0
    speed = 0.0
    mutation_point = None

    while episode_count < args.num_episodes:
        done = False
        episode_length = 0

        # generate track
        if mutation_point is not None:
            print("episode_count %d, mutation_point: %d" % (episode_count, mutation_point))
        obs = env.reset(mutation_point=mutation_point, skip_generation=False)

        # these are the interpolated road points
        # print(env.executor.current_track.get_concrete_representation())

        # these are the original control points
        control_points = env.executor.current_track.get_control_points()

        start_time = time.perf_counter()

        # TODO: keep mutating until misbehaviour, or a timeout
        while not done:

            action = agent.predict(obs=obs, speed=speed)

            # clip action to avoid out of bound errors
            if isinstance(env.action_space, gym.spaces.Box):
                action = np.clip(action, env.action_space.low, env.action_space.high)

            # obs is the image, info contains the road and the position of the car
            obs, done, info = env.step(action)

            # car position
            car_position = info["pos"]
            # control points 0, 8, 9 are always not considered
            # TODO: refine get_closest_control_point to avoid useless computations
            closest = get_closest_control_point(car_position, control_points)
            logger.debug(f"info {info} closest control point is #{closest}/{len(control_points)}")

            # create directory to save the simulation log
            runpath = os.path.join(
                time_start
                + "-seed="
                + str(args.seed)
                + "-num-episodes="
                + str(args.num_episodes)
                + "-agent="
                + str(args.agent_type)
                + "-num-control-nodes="
                + str(args.num_control_nodes)
                + "-max-angle="
                + str(args.max_angle)
            )
            filepath = os.path.join("simulations", runpath, "episode" + str(episode_count))
            if not Path(filepath).exists():
                Path(filepath).mkdir(parents=True, exist_ok=True)

            # save the image
            filename = os.path.join(
                filepath,
                "img"
                + str(episode_length)
                + "-"
                + str(info["pos"][0])
                + "-"
                + str(info["pos"][1])
                + ".jpg",
            )
            Image.fromarray(obs).save(filename)

            speed = 0.0 if info.get("speed", None) is None else info.get("speed", None)

            # write CSV to file
            if not os.path.exists(filepath + ".csv"):
                # create csv file header
                with open(filepath + ".csv", "w", encoding="UTF8") as f:
                    writer = csv.writer(
                        f,
                        delimiter=",",
                        quotechar='"',
                        quoting=csv.QUOTE_MINIMAL,
                        lineterminator="\n",
                    )

                    # write the header
                    writer.writerow(
                        [
                            "episode_count",
                            "episode_length",
                            "img",
                            "car_x",
                            "car_y",
                            "closest_cp",
                            "speed",
                            "cte",
                            "is_success",
                            "mutated_cp",
                        ]
                    )

            with open(filepath + ".csv", "a", encoding="UTF8") as f:
                writer = csv.writer(
                    f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
                )
                writer.writerow(
                    [
                        episode_count,
                        episode_length,
                        filename,
                        info["pos"][0],
                        info["pos"][1],
                        closest,
                        speed,
                        info["cte"],
                        info["is_success"],
                        mutation_point,
                    ]
                )

            episode_length += 1
            if done:
                times_elapsed.append(time.perf_counter() - start_time)
                logger.debug("Episode #{}".format(episode_count + 1))
                logger.debug("Episode Length: {}".format(episode_length))
                logger.debug("Is success: {}".format(info["is_success"]))

                episode_lengths.append(episode_length)
                episode_count += 1

                if info["is_success"] == 1:

                    # run ThirdEye to compute the hottest point in the simulation
                    max_gradient_score_idx, _ = compute_heatmap(simulation_name=filepath)

                    # print(f"max_gradient_score_idx {max_gradient_score_idx}")
                    # retrieve the control point which is closest to the hottest point
                    df = pd.read_csv(filepath + ".csv")
                    result = df.iloc[max_gradient_score_idx]
                    hotpoint_position = result[["closest_cp"]][0]

                    # print(f"result {result}")
                    # print(f"hotpoint_position {hotpoint_position}")
                    # print(f"control_points {control_points}")

                    mutation_point = hotpoint_position  # get_closest_control_point(hotpoint_position, control_points)
                    print("closest control point to the hottest point is %d\n" % mutation_point)
                else:
                    env.close()
                    sys.exit("Misbehavior detected. Mutation finished.")
    logger.debug("Success rate: {:.2f}".format(success_sum / episode_count))
    logger.debug("Mean time elapsed: {:.2f}s".format(np.mean(times_elapsed)))

    env.reset(mutation_point=mutation_point, skip_generation=False)
    time.sleep(5)
    env.close()
