import tqdm
import random
import numpy as np

"""
Q-learning algorithm 예제입니다. 
간단한 Grid World 환경에서 에이전트가 최적의 정책을 학습하는 과정을 구현합니다.
간단한 격자판 위에, 장해물을 피해서 목표 지점에 도달하는 문제를 다룹니다.
환경 설정, Q-테이블 초기화, 학습 과정, 그리고 최종 정책 출력까지 포함되어 있습니다.

"""
class SimpleGridworld: 
    """
    4x4 격자판 환경에서 에이전트가 시작점(0,0)에서 목표점(3,3)까지 이동하는 문제
    구멍(1,1)과 (2,2)을 피해야 함
    행동:  0: 위, 1: 아래, 2: 왼쪽, 3: 오른쪽
    상태: (row, col) 형태의 좌표
    16개의 상태 (0~15 인덱스)
    4개의 행동
    1. 목표 도달 시 보상 +1.0
    2. 구멍에 빠질 시 보상 -1.0
    3. 그 외에는 작은 패널티 -0.01
    4. 에피소드 종료 조건: 목표 도달 또는 구멍에 빠짐
    """

    def __init__(self): ##환경 초기화 
        self.size = 4 
        self.start = (0, 0) 
        self.goal = (3, 3) 
        self.holes = [(1, 1), (2, 2)] 
        self.state =self.start  ## State 초기화

    def reset(self): ## 환경 재설정
        self.state = self.start
        return self._state_to_idx(self.state)
    
    def _state_to_idx(self, 
                      state): 
        """
        state를 인덱스로 변환
        """
        return state[0] * self.size + state[1]  # 0~15 인덱스 반환
    
    def step(self, 
             action): ## 행동에 따른 상태 전이 및 보상 반환
        """ 
        행동 : 0: 위, 1: 아래, 2: 왼쪽, 3: 오른쪽 
        격자판을 움직이는 함수 
        """
        # 현재 상태
        row, col = self.state

        # 행동에 따른 상태 변화
        if action == 0: ## 위쪽 
            row = max(0, row - 1)
        elif action == 1: ## 아래쪽
            row = min(self.size - 1, row + 1)
        elif action == 2: ## 왼쪽
            col = max(0, col - 1)
        elif action == 3: ## 오른쪽
            col = min(self.size - 1, col + 1)
        
        self.state = (row, col) # 상태 업데이트 / 격자판 위에서 에이전트가 현재 위치한 좌표를 나타냄
        ## 보상설정
        if self.state == self.goal: ## 목표 도달
            reward = 1.0 
            done = True
        elif self.state in self.holes: ## 구멍에 빠짐
            reward = -1.0 
            done = True
        else:
            reward = -0.01 # 작은 패널티(이유: 빨리 목표에 도달하도록 유도함)
            done = False

        return self._state_to_idx(self.state), reward, done ## 다음 상태, 보상, 종료 여부 반환

# Q-Learning 알고리즘 구현
class QLearning:
    def __init__(self, 
                 n_states, 
                 n_actions,
                 learning_rate=0.1,
                 discount_factor = 0.99,
                 epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions)) ## Q-테이블 초기화
        self.lr = learning_rate ## 학습률
        self.gamma = discount_factor ## 할인율
        self.epsilon = epsilon ## 탐험 확률
        self.n_actions = n_actions # 행동 수

    def get_action(self, 
                   state):
        """
        ε-greedy policy 으로 행동 선택
        """
        if random.random() < self.epsilon: ## 탐험
            return random.randint(0, self.n_actions -1)
        else: 
            return np.argmax(self.q_table[state]) ## 활용
    
    def update(self,
               state,  
               action, 
               reward, 
               next_state, 
               done):
        """
        Q-테이블 업데이트
        """
        current_q = self.q_table[state, action] ## 현재 Q값

        if done:
            max_next_q = 0 ## 종료 상태에서는 다음 Q값이 0
        else:
            max_next_q = np.max(self.q_table[next_state]) ## 다음 상태에서 최대 Q값 계산
        
        ## Q-값 업데이트 공식 > Q(s,a) = Q(s,a) + α * (r + γ * max_a' Q(s',a') - Q(s,a))
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q ## Q-테이블 갱신

#train 
def train(): 
    env = SimpleGridworld() ## 환경 생성
    agent = QLearning(n_states=16, 
                      n_actions=4,
                      learning_rate=0.01, 
                      discount_factor=0.99,
                      epsilon=0.1) ## 에이전트 생성 # 상태 16(4*4), 행동 4(상하좌우)
    
    n_episodes = 1000 ## 에피소드 수
    rewards_per_episode = [] ## 에피소드별 총 보상 기록 리스트
 
    for episode in tqdm.tqdm(range(n_episodes)): ## 에피소드 반복
        state = env.reset() ## 환경 초기화
        total_reward = 0 ## 총 보상 초기화
        done = False ## 종료 여부 초기화
        step = 0 ## 스텝 수 초기화
        max_step = 100 ## 최대 스텝 수 설정

        while not done and step < max_step: ## 에피소드 진행
            action = agent.get_action(state) ## 행동 선택
            next_state, reward, done = env.step(action) ## 환경에서 한 스텝 진행
            agent.update(state, action, reward, next_state, done) ## Q-테이블 업데이트

            state = next_state ## 상태 업데이트
            total_reward += reward ## 총 보상 누적
            step += 1 ## 스텝 수 증가

        rewards_per_episode.append(total_reward) ## 에피소드별 총 보상 기록

        ## 100 에피소드마다 평균 보상 출력
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
    
        
    ## 최종 Q-테이블 출력
    print("Final Q-Table:")
    print(agent.q_table)

    ## 최적 정책 테스트 
    print('최적 정책으로 경로 찾기: ')
    state = env.reset()
    done = False
    path = [(0,0)] ## 시작점 추가
    steps = 0 

    while not done and steps < 20:
        action = np.argmax(agent.q_table[state]) ## 최적 행동 선택
        next_state, reward, done = env.step(action) ## 환경에서 한 스텝 진행
        path.append(env.state) # 현재 좌표 추가
        state = next_state ## 상태 업데이트
        steps += 1 ## 스텝 수 증가

    print("경로: ", path)
    print(f'목표 도달: {env.state == env.goal}')

if __name__ == "__main__":
    train()
    