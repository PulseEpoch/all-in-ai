import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from ppo_agent import PPOAgent

def train_ppo_agent(env, agent, episodes=500, max_t=1000, update_timestep=200):
    scores = []
    timestep = 0
    for i_episode in range(1, episodes + 1):
        state = env.reset()
        # 对于gym的新版本，reset()返回元组(state, info)
        if isinstance(state, tuple):
            state = state[0]
        score = 0
        
        for t in range(max_t):
            timestep += 1
            action, action_log_prob = agent.act(state)
            # 处理不同版本的gym step返回值
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, done, info, _ = step_result
            else:
                next_state, reward, done, _ = step_result
            
            agent.store_transition(state, action, action_log_prob, reward, next_state, done)
            state = next_state
            score += reward

            # Update PPO agent every update_timestep steps
            if timestep % update_timestep == 0 and len(agent.states) > 0:
                actor_loss, critic_loss = agent.learn()
                timestep = 0

            if done:
                break
        
        scores.append(score)
        
        # 打印训练进度
        if i_episode % 10 == 0:
            print(f'Episode {i_episode}, Average Score: {np.mean(scores[-10:]):.2f}')
        
        # 检查是否解决了问题（CartPole-v1要求平均得分>=475）
        if i_episode >= 100 and np.mean(scores[-100:]) >= 475:
            print(f'Environment solved in {i_episode-100} episodes! Average Score: {np.mean(scores[-100:]):.2f}')
            # 保存模型
            torch.save(agent.actor.state_dict(), 'ppo_actor.pth')
            torch.save(agent.critic.state_dict(), 'ppo_critic.pth')
            break
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title('PPO Agent Training on CartPole')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.savefig('training_scores.png')
    plt.close()
    
    return scores

def test_agent(env, agent, episodes=5, render=True):
    scores = []
    for i in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        score = 0
        
        for t in range(1000):  # 最多运行1000步
            if render:
                env.render()  # 显示环境
                
            action, _ = agent.act(state)
            # 处理不同版本的gym step返回值
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, done, info, _ = step_result
            else:
                next_state, reward, done, _ = step_result
            
            state = next_state
            score += reward
            
            if done:
                break
        
        print(f'Test Episode {i+1}, Score: {score}')
        scores.append(score)
    
    print(f'Average Test Score: {np.mean(scores):.2f}')
    return scores

def main():
    # 检查命令行参数，控制是否渲染GUI
    render = True
    if len(sys.argv) > 1 and sys.argv[1] == '--no-render':
        render = False
    
    # 根据是否渲染选择合适的render_mode
    render_mode = 'human' if render else None
    
    try:
        # 创建CartPole环境
        env = gym.make('CartPole-v1', render_mode=render_mode)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        print(f'State size: {state_size}, Action size: {action_size}')
        
        # 创建PPO Agent
        agent = PPOAgent(state_size, action_size)
        
        # 训练Agent
        print("Starting training...")
        scores = train_ppo_agent(env, agent)
        
        # 测试训练好的Agent
        print("\nTesting trained agent...")
        test_agent(env, agent, render=render)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 确保关闭环境
        try:
            env.close()
        except:
            pass

if __name__ == '__main__':
    main()