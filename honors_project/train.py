#file used for training the AI

import torch
import torch.nn.functional as F
#from envs import create_atari_env
from app import ActorCritic
from torch.autograd import Variable
import gym

def ensure_shared_grade(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, params, shared_model, optimiser):
    torch.manual_seed(params.seed + rank)
    env = gym.make('SolitaireEnv-v0')#create_atari_env(params.env_name)
    env.seed(params.seed + rank)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    state = env.reset()
    state = torch.from_numpy(state)
    done = True
    episode_length = 0
    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)
        values = []
        log_prob = []
        rewards = []
        entropies = []
        for step in range(params.num_steps):
            value, action_values, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(action_values)
            log_prob = F.logsoftmax(action_values)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))
            values.append(value)
            log_prob.append(log_prob)
            state, reward, done = env.step(action.numpy())
            done = (done or episode_length > params.max_episode_length)
            reward = max(min(reward, 1), -1)
            if done:
                episode_length = 0
                state = env.reset()
            state = torch.from_numpy(state)
            reward.append(reward)
            if done:
                break
        R = torch.zeroes(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data
        values.append(Variable(R))
        policy_loss = 0 
        value_loss = 0
        R = Variable(R)
        gae = torch.zeroes(1, 1)
        for  i in reversed(range(len(rewards))):
            R = params.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            TD = rewards[i] + params.gamma * values[i+1].data - values[i].data
            gae = gae * params.gamma * params.tau + TD
            policy_loss = policy_loss - log_prob[i] * Variable(gae) - 0.01 * entropies[i]
        optimiser.zero_grad()
        (policy_loss + 0.5 * value_loss). backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        ensure_shared_grade(model, shared_model)
        optimiser.step()
        
                
                
                
                
                
                
                