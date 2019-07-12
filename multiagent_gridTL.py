


import json
import argparse

import ray
import flow
 
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray import tune

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter
from flow.scenarios.grid import SimpleGridScenario



# time horizon of a single rollout
HORIZON = 400
#number of CPU's
N_CPUS = 2
#benchmark name
benchmark_name = 'grid0'
#The number of rollouts to train over
N_ROLLOUTS = 50



# inflow rate of vehicles at every edge
EDGE_INFLOW = 300
# enter speed for departing vehicles
V_ENTER = 30
# number of row of bidirectional lanes
N_ROWS = 3
# number of columns of bidirectional lanes
N_COLUMNS = 3
# length of inner edges in the grid network
INNER_LENGTH = 300
# length of final edge in route
LONG_LENGTH = 100
# length of edges that vehicles start on
SHORT_LENGTH = 300
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 0, 0, 0, 0



# we place a sufficient number of vehicles to ensure they confirm with the
# total number specified above. We also use a "right_of_way" speed mode to
# support traffic light compliance
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
        max_speed=V_ENTER,
        speed_mode="right_of_way",
    ),
    routing_controller=(GridRouter, {}),
    num_vehicles=(N_LEFT + N_RIGHT) * N_COLUMNS + (N_BOTTOM + N_TOP) * N_ROWS)

# inflows of vehicles are placed on all outer edges (listed here)
outer_edges = []
outer_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
outer_edges += ["right0_{}".format(i) for i in range(N_ROWS)]
outer_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
outer_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

# random inflows for each edge (as dictated by probability constant)
inflow = InFlows()
for edge in outer_edges:
    if edge=="bot0_0":
        prob=0.10
    elif edge=="bot1_0":
        prob=0.10
    elif edge=="bot2_0":
        prob=0.10
    elif edge=="top0_3":
        prob=0.10
    elif edge=="top1_3":
        prob=0.10
    elif edge=="top2_3":
        prob=0.10
    elif edge=="right0_0":
        prob=0.10
    elif edge=="right0_1":   #middelste verticale onderaan
        prob=0.10
    elif edge=="right0_2":
        prob=0.10
    elif edge=="left3_0":
        prob=0.10
    elif edge=="left3_1":   #middelste verticale bovenaan 
        prob=0.10
    elif edge=="left3_2":
        prob=0.10
    inflow.add(
        veh_type="human",
        edge=edge,
	probability=prob,
        #vehs_per_hour=EDGE_INFLOW,
        departLane="free",
        departSpeed="max")

flow_params = dict(
    # name of the experiment
    exp_tag="3x3Multi_Agent",

    # name of the flow environment the experiment is running on
    env_name="PO_MultiAgentTLenv",

    # name of the scenario class the experiment is running on
    scenario="SimpleGridScenario",

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        restart_instance=True,
        sim_step=1,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            "target_velocity": 50,
            "switch_time": 2,
            "num_observed": 2,
            "discrete": True,
            "tl_type": "actuated"
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params={
            "speed_limit": V_ENTER + 5,
            "grid_array": {
                "short_length": SHORT_LENGTH,
                "inner_length": INNER_LENGTH,
                "long_length": LONG_LENGTH,
                "row_num": N_ROWS,
                "col_num": N_COLUMNS,
                "cars_left": N_LEFT,
                "cars_right": N_RIGHT,
                "cars_top": N_TOP,
                "cars_bot": N_BOTTOM,
            },
            "horizontal_lanes": 1,
            "vertical_lanes": 1,
        },
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing='custom',
        shuffle=True,
    ),

)


def setup_exps():

    # Import the benchmark and fetch its flow_params
    benchmark = __import__(
        "flow.benchmarks.%s" % benchmark_name, fromlist=["flow_params"])
    flow_params = benchmark.flow_params


    alg_run = 'PPO'
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config['num_workers'] = N_CPUS
    config['train_batch_size'] = HORIZON * N_ROLLOUTS
    config['simple_optimizer'] = True
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [32, 32]})
    config['lr'] = tune.grid_search([1e-5])
    config['horizon'] = HORIZON
    config['clip_actions'] = False
    config['observation_filter'] = 'NoFilter'

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)

    test_env = create_env()
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def gen_policy():
        return (PPOPolicyGraph, obs_space, act_space, {})

    # Setup PG with an ensemble of `num_policies` different policy graphs
    policy_graphs = {'av': gen_policy()}

    def policy_mapping_fn(_):
        return 'av'

    config.update({
        'multiagent': {
            'policy_graphs': policy_graphs,
            'policy_mapping_fn': tune.function(policy_mapping_fn),
            'policies_to_train': ['av']
        }
    })

    return alg_run, env_name, config

if __name__ == '__main__':
    alg_run, env_name, config = setup_exps()
    ray.init(num_cpus=N_CPUS + 1)

    run_experiments({
        flow_params['exp_tag']: {
            'run': alg_run,
            'env': env_name,
            'checkpoint_freq': 10,
            'stop': {
                'training_iteration': 500
            },
            'config': config,
            # 'upload_dir': 's3://<BUCKET NAME>'
        },
    })

