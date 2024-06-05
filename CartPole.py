from math import radians
from random import random
import gym
import matplotlib.pyplot as plt



buckets = (1, 1, 6, 12,)  
def discretize(obs, env):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -radians(50)]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, int(round((buckets[i] - 1) * ratios[i])))) for i in range(len(obs))]

    return tuple(new_obs)


def add_state(q, state, action_space):
    if state not in q:
        q.update({state: {x: 0 for x in action_space}})


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    action_space = [i for i in range(env.action_space.n)]

    win_score = 195
    
    n_episodes = 10000
    max_turns = 200

    get_alpha = lambda t: .6 if t < 150 else .1
    get_epsilon = lambda t: 1 if t < 150 else .1
    gamma = 0.99

    q = {}
    wins = 0
    scores = []
    for episode in range(n_episodes):
        r = 0  # cumulative reward

        alpha = get_alpha(episode)
        epsilon = get_epsilon(episode)

        observation = env.reset()
        state = discretize(observation, env)
        add_state(q, state, action_space)

        for _ in range(max_turns):
            action = max(q[state], key=q[state].get) if random() > epsilon else env.action_space.sample()

            observation, reward, done, info = env.step(action)

            next_state = discretize(observation, env)
            add_state(q, next_state, action_space)

            q[state][action] = (1 - alpha) * q[state][action] + alpha * (reward + gamma * max(q[next_state].values()))

            state = next_state
            
            r += reward

            if done:
                break

        scores.append(r)
        if not episode % 10:
            print(f'Episode: {episode}, Reward: {r}\n\n')
        
        if r >= win_score:
            wins += 1
            if wins == 50:
                print(f'O problema foi resolvido no episodio: {episode}')
                break
        else:
            wins = 0        

    env.close()

plt.plot(scores)
plt.xlabel('Episódios')
plt.ylabel('Pontuação (Passos de Tempo)')
plt.title('Pontuação ao longo dos episódios')
plt.show()
