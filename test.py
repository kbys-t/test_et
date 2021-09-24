# coding:utf-8

import os
import sys
import itertools

import matplotlib as mpl
mpl.use("Agg")
import torch.multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from modules.ac import AC as RL

import gym
import gym.wrappers
import pybullet_envs

# torch.autograd.set_detect_anomaly(True)
######################################################
# hyperparameters
# number of multiprocess
N_PROCESS = int(mp.cpu_count() / 2)
# N_PROCESS = 1

# whether use cuda or not: in such simple regression tasks, cpu is basically faster than gpu
USE_CUDA = False

SAVE_DIR = "./result/"

# if only viewing final performances and replotting previous results, please change to False
N_TRIAL = 20
N_RESUME = 0
IS_LEARN = True
IS_VIEW = True
N_TEST = 100 if IS_LEARN else 0
REDUCTION_WAY = "median"

PLOT_LEARN = True
PLOT_DECAY = True
NCOL = 3

# comparison

METHODS = [
    {"lambda1": 0.0, "lambda2": 0.0, "div_weight": 0.0},
    {"lambda1": 0.9, "lambda2": 0.0, "div_weight": 0.0},
    {"lambda1": 0.9, "lambda2": 0.0, "div_weight": 1.0},
    {"lambda1": 0.0, "lambda2": 0.9, "div_weight": 0.0},
    {"lambda1": 0.0, "lambda2": 0.9, "div_weight": 1.0},
    {"lambda1": 0.5, "lambda2": 0.9, "div_weight": 1.0},
    ]

print(METHODS)

MAX_TIME = 2000
ENVIRONMENTS = [
    {"env_name": "InvertedPendulumBulletEnv-v0", "n_episode": 200, "score": "sum"},
    {"env_name": "InvertedPendulumSwingupBulletEnv-v0", "n_episode": 200, "score": "sum"},
    {"env_name": "HalfCheetahBulletEnv-v0", "n_episode": 2000, "score": "sum"},
    {"env_name": "AntBulletEnv-v0", "n_episode": 2000, "score": "sum"},
    ]

######################################################
def make_Dirs(d):
    if not os.path.exists(d):
        os.makedirs(d)

######################################################
def set_Seed(seed, env):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

######################################################
def save_Data(episodes, data, sdir, name):
    np.savetxt(sdir + name + ".csv", np.array([episodes, data]).T, delimiter=",")
    plt.clf()
    plt.plot(episodes, data)
    plt.xlabel("Episodes")
    plt.ylabel(name)
    plt.tight_layout()
    plt.savefig(sdir + name + ".pdf")

######################################################
def rename_Env(env_name):
    if "Swingup" in env_name:
        env_name = "Swingup"
    elif "DoublePendulum" in env_name:
        env_name = "DoublePendulum"
    else:
        env_name = env_name[0:env_name.find("-")]
        if "BulletEnv" in env_name:
            env_name = env_name[0:-9]
    return env_name

######################################################
def process(args):
    n_trial, env_info, method = args
    env_name = env_info["env_name"]
    n_episode = env_info["n_episode"]
    env_score = env_info["score"]
    sdir = SAVE_DIR + str(env_name) + "/" + str(tuple(method.values())) + "/" + str(n_trial) + "/"
    make_Dirs(sdir)

    env = None
    age = None
    if IS_LEARN or N_TEST > 0:
        env = gym.make(env_name)
        set_Seed(n_trial, env)
        s_dim = env.observation_space.shape[0] if len(env.observation_space.shape) == 1 else (env.observation_space.shape[2], env.observation_space.shape[0], env.observation_space.shape[1])
        is_discrete = type(env.action_space) == gym.spaces.discrete.Discrete
        a_dim = env.action_space.n if is_discrete else env.action_space.shape[0]
        age = RL(s_dim, a_dim, is_discrete, use_cuda=USE_CUDA, **method)

    # training agent
    if IS_LEARN:
        list_score = []
        list_train_loss = []
        list_average_decay = []
        episodes = range(1, n_episode+1)
        print("Start training: {}".format(sdir))
        for epi in episodes:
            score, train_loss, average_decay, t_end = train(age, env, env_score)
            list_score.append(score)
            list_train_loss.append(train_loss)
            list_average_decay.append(average_decay)
            print("====> Episode: {} of {}\n\t Elapsed time: {:d}\t Score: {:.2e}\t Loss: {:.2e}\t Lambda: {:.3f}".format(epi, sdir, t_end, score, train_loss, average_decay))
        # save results
        age.release(sdir)
        #
        save_Data(episodes, list_score, sdir, "score")
        save_Data(episodes, list_train_loss, sdir, "train_loss")
        save_Data(episodes, list_average_decay, sdir, "average_decay")
        #
        env.close()
    # collect test results
    if N_TEST > 0:
        # renew environment
        env = gym.make(env_name)
        if IS_VIEW:
            env = gym.wrappers.Monitor(env, sdir + "video/", force=True)
        set_Seed(2**31-n_trial, env)
        if age.load(sdir):
            list_score = []
            tests = range(1, N_TEST+1)
            for tes in tests:
                #
                print("Start testing: {} of {}".format(tes, sdir))
                score, t_end = test(age, env, env_score, sdir)
                list_score.append(score)
                print("====> Test: {}\n\t Elapsed time: {:d}\t Score: {:.6e}".format(sdir, t_end, score))
            np.savetxt(sdir + "test_result.csv", np.array([tests, list_score]).T, delimiter=",")
        #
        env.close()
    return sdir

######################################################
def train(age, env, env_score):
    age.train()
    score = np.finfo(np.float32).min if "max" in env_score else 0.0
    train_loss = 0.0
    average_decay = 0.0
    t_end = 0

    age.reset()
    state = env.reset()
    done = False
    action, base_policy, base_value = age(state)
    _, policy, value = age(state)

    for t_ in range(MAX_TIME):
        # step
        next_state, reward, done, info = env.step(age.act_sim2env(action, env.action_space))
        # forward
        next_action, next_base_policy, next_base_value = age(next_state)
        # compute loss functions
        tderr, decay, loss_rl = age.criterion(policy, base_policy, action, value, base_value, next_base_value, reward, done)
        # update network
        age.update(tderr, decay, loss_rl)
        # store results
        if "max" in env_score and reward > score:
            score = reward
        else:
            score += reward
        train_loss += loss_rl.item()
        average_decay += decay.item()
        # update time step
        t_end += 1
        if done:
            break
        # substitute
        state = next_state
        action = next_action
        base_policy = next_base_policy
        base_value = next_base_value
        # compute again w.r.t the current parameters
        _, policy, value = age(state)
    # return results
    if "mean" in env_score:
        score /= t_end
    train_loss /= t_end
    average_decay /= t_end
    return score, train_loss, average_decay, t_end

######################################################
def test(age, env, env_score, sdir):
    age.eval()
    score = np.finfo(np.float32).min if "max" in env_score else 0.0
    t_end = 0

    age.reset()
    if IS_VIEW:
        env.stats_recorder.save_complete()
        env.stats_recorder.done = True
    state = env.reset()
    done = False
    action, _, _ = age(state)

    for t_ in range(MAX_TIME):
        # step
        state, reward, done, info = env.step(age.act_sim2env(action, env.action_space))
        # forward
        action, _, _ = age(state)
        # store results
        if "max" in env_score and reward > score:
            score = reward
        else:
            score += reward
        # update time step
        t_end += 1
        if done:
            break
    # return results
    if "mean" in env_score:
        score /= t_end
    return score, t_end

######################################################
def eval(sdirs, reduction="none", ncol=0):
    # configurations of plot
    sns.set(context = "paper", style = "white", palette = "Set2", font = "Arial", font_scale = 1.8, rc = {"lines.linewidth": 1.0, "pdf.fonttype": 42})
    sns.set_palette("Set2", 8, 1)
    colors = sns.color_palette(n_colors=10)
    markers = ["o", "s", "d", "*", "+", "x", "v", "^", "<", ">"]
    fig = plt.figure(figsize=(8, 6))

    legend = ncol > 0

    # prepare dataframes
    columns = ["Environment", "Method", "n_repeat", "Episode", "Score"]
    columns_param = ["Environment", "Method", "n_repeat", "Episode", "Value"]
    common_dir = os.path.abspath(os.path.commonpath(sdirs)) + "/"
    summary = pd.DataFrame(columns=columns)
    learn = pd.DataFrame(columns=columns)
    decay = pd.DataFrame(columns=columns_param)
    balance_rr = pd.DataFrame(columns=columns_param)
    balance_te = pd.DataFrame(columns=columns_param)

    # load and append file
    def load_File(dir, name, df, col, info, reduction=None):
        filename = dir + name + ".csv"
        if os.path.isfile(filename):
            dat = pd.read_csv(filename, header=None)
            if reduction is not None:
                if reduction == "mean":
                    dat = dat.mean().to_frame().transpose()
                if reduction == "median":
                    dat = dat.median().to_frame().transpose()
                if reduction == "max":
                    dat = dat.max().to_frame().transpose()
            dat.columns = col[3:]
            dat = pd.concat([dat, pd.DataFrame([info]*len(dat.index), columns=col[0:3])], axis=1)
            df = df.append(dat, ignore_index=True, sort=True)
        else:
            print(filename + " is none")
        return df

    # plot episode vs value
    def plot_Data(dir, name, df, col, flag):
        if flag:
            print("Plot " + name)
            for env in df[col[0]].unique():
                print("For {}".format(env))
                dat = df[df[col[0]] == env]
                plt.clf()
                sns.lineplot(x=col[-2], y=col[-1], hue=col[1], data=dat)
                if legend:
                    handles, labels = plt.gca().get_legend_handles_labels()
                    plt.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(0.5, 1.175), loc="upper center", frameon=True, ncol=ncol)
                else:
                    plt.gca().get_legend().remove()
                plt.xlim([dat[col[-2]].min(), dat[col[-2]].max()])
                # plt.gca().ticklabel_format(style="sci", axis="y", scilimits=(0,0))
                plt.savefig(dir + name + "_" + env + ".pdf")
        else:
            print("No " + name)

    # load files
    print("Load files")
    for dir in sdirs:
        # get conditions
        info = dir.split("/")[-4:-1]
        info[0] = rename_Env(info[0])

        # load files
        summary = load_File(dir, "test_result", summary, columns, info, reduction)
        if PLOT_LEARN:
            learn = load_File(dir, "score", learn, columns, info)
        if PLOT_DECAY:
            decay = load_File(dir, "average_decay", decay, columns_param, info)

    # save as csv at first
    summary.to_csv(common_dir + "summary.csv", index=False)
    if PLOT_LEARN:
        learn.to_csv(common_dir + "learn.csv", index=False)
    if PLOT_DECAY:
        decay.to_csv(common_dir + "decay.csv", index=False)

    # plot results
    if len(summary) > 0:
        # plot test results
        print("Plot test results as bar chart")
        plt.clf()
        plt.axhline(0, c="k", ls="dashed")
        # sns.barplot(x=columns[0], y=columns[-1], hue=columns[1], data=summary)
        sns.boxplot(x=columns[0], y=columns[-1], hue=columns[1], data=summary, showfliers = False)
        # sns.boxenplot(x=columns[0], y=columns[-1], hue=columns[1], data=summary, outlier_prop=0.025)
        if legend:
            plt.legend(bbox_to_anchor=(0.5, 1.175), loc="upper center", frameon=True, ncol=ncol)
        else:
            plt.gca().get_legend().remove()
        plt.savefig(common_dir + "summary.pdf")
        #
        print("Print values")
        for env in summary[columns[0]].unique():
            for method in summary[columns[1]].unique():
                print(env, method)
                df = pd.DataFrame(summary[(summary[columns[0]] == env) & (summary[columns[1]] == method)][columns[-1]])
                print(df.describe().transpose())
    else:
        print("No summary")

    # plot learning curves
    plot_Data(common_dir, "learn", learn, columns, PLOT_LEARN and len(learn) > 0)
    # plot decaying factor
    plot_Data(common_dir, "decay", decay, columns_param, PLOT_DECAY and len(decay) > 0)

######################################################
def main():
    make_Dirs(SAVE_DIR)

    sdirs = []
    if N_PROCESS > 1:
        pool = mp.Pool(processes=N_PROCESS)
        sdirs = pool.map(process, [(n_trial, env_info, method) for n_trial, env_info, method in itertools.product(range(1+N_RESUME, N_TRIAL+1), ENVIRONMENTS, METHODS)])
        pool.close()
    else:
        sdirs = [process((n_trial, env_info, method)) for n_trial, env_info, method in itertools.product(range(1+N_RESUME, N_TRIAL+1), ENVIRONMENTS, METHODS)]

    print("Evaluate data")
    eval(sdirs, REDUCTION_WAY, NCOL)

######################################################
if __name__ == "__main__":
    main()
