import numpy as np
import random

class Environment:
    def __init__(self, grid_size=3):
        self.grid_size = grid_size
        self.reset()
    
    def reset(self):
        # Agent starts at the top-left corner
        self.agent_pos = [0, 0]
        return self._get_state()
    
    def _get_state(self):
        # Represent state as a single number (row * grid_size + column)
        return self.agent_pos[0] * self.grid_size + self.agent_pos[1]
    
    def step(self, action):
        # Actions: 0: up, 1: right, 2: down, 3: left
        if action == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[1] < self.grid_size - 1:
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] < self.grid_size - 1:
            self.agent_pos[0] += 1
        elif action == 3 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        
        state = self._get_state()
        reward = 0
        done = False
        # Define goal as the bottom-right corner
        if self.agent_pos == [self.grid_size - 1, self.grid_size - 1]:
            reward = 1
            done = True
        return state, reward, done

class Agent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_states = env.grid_size * env.grid_size
        self.num_actions = 4  # up, right, down, left
        self.Q = np.zeros((self.num_states, self.num_actions))
    
    def choose_action(self, state):
        # Epsilon-greedy policy for exploration/exploitation
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.Q[state])
    
    def learn(self, state, action, reward, next_state, done):
        # Q-learning update
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.discount_factor * self.Q[next_state, best_next_action] * (not done)
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.learning_rate * td_error

def main():
    env = Environment(grid_size=3)
    agent = Agent(env)
    num_episodes = 500
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            
    print("Trained Q-Table:")
    print(agent.Q)

if __name__ == "__main__":
    main()
