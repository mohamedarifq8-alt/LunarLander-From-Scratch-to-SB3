import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
import torch.optim as optim
import os

# ==========================================
# 1. المعمارية المستقرة (التي أثبتت نجاحها)
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor_fc1 = nn.Linear(state_dim, 256)
        self.actor_fc2 = nn.Linear(256, 256)
        self.mu_layer = nn.Linear(256, action_dim)
        
        # عشوائية مستقلة لضمان استمرار الاستكشاف بشكل صحي
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        self.critic_fc1 = nn.Linear(state_dim, 256)
        self.critic_fc2 = nn.Linear(256, 256)
        self.value_layer = nn.Linear(256, 1)

    def get_action_and_value(self, state, action=None):
        x_actor = F.relu(self.actor_fc1(state))
        x_actor = F.relu(self.actor_fc2(x_actor))
        mu = torch.tanh(self.mu_layer(x_actor)) 
        
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        x_critic = F.relu(self.critic_fc1(state))
        x_critic = F.relu(self.critic_fc2(x_critic))
        value = self.value_layer(x_critic)
        
        return action, log_prob, entropy, value.squeeze(-1)

# ==========================================
# 2. الوكيل (باستخدام التقييم الآمن والمفصول)
# ==========================================
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.ppo_epochs = 4
        self.entropy_coef = 0.01 # نبدأ بعشوائية صحية
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=3e-4)
        
        self.memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': []}

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(state)
            
        self.memory['states'].append(state.cpu().numpy()[0])
        self.memory['actions'].append(action.cpu().numpy()[0])
        self.memory['log_probs'].append(log_prob.cpu().numpy()[0])
        self.memory['values'].append(value.cpu().numpy()[0])
        return action.cpu().numpy()[0]

    def store_reward(self, reward, done):
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)

    def learn(self):
        states = torch.tensor(np.array(self.memory['states']), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(self.memory['actions']), dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(np.array(self.memory['log_probs']), dtype=torch.float32).to(self.device)
        old_values = torch.tensor(np.array(self.memory['values']), dtype=torch.float32).to(self.device)
        rewards = self.memory['rewards']
        dones = self.memory['dones']
        
        # التقييم الآمن الذي يفصل المحاولات تماماً (if done: G = 0)
        returns = []
        G = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done: G = 0
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.ppo_epochs):
            _, new_log_probs, entropy, new_values = self.network.get_action_and_value(states, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_values, returns)
            
            total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy.mean()
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
        for key in self.memory:
            self.memory[key].clear()

# ==========================================
# 3. حلقة التدريب مع التلاشي الذكي (Decay)
# ==========================================
env = gym.make("LunarLanderContinuous-v3")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = PPOAgent(state_dim, action_dim)
EPISODES = 3000
scores = []
best_score = 150.0 # نبحث عن الهبوط المثالي
update_every_n_episodes = 4 

print("🚀 بدء التدريب المستقر نحو الهبوط المثالي...")

for e in range(EPISODES):
    state, _ = env.reset()
    total_score = 0
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.store_reward(reward, done)
        state = next_state
        total_score += reward
        
    scores.append(total_score)
    
    if (e + 1) % update_every_n_episodes == 0:
        agent.learn()
        
    # التلاشي التدريجي للعشوائية: يقل بنسبة بسيطة جداً ليصبح الوكيل دقيقاً في النهاية
    agent.entropy_coef = max(0.0005, agent.entropy_coef * 0.998)
    
    if (e + 1) % 25 == 0:
        avg_score = np.mean(scores[-25:])
        print(f"المحاولة: {e+1}/{EPISODES} | متوسط السكور: {avg_score:.2f} | العشوائية: {agent.entropy_coef:.5f}")
        
        if avg_score > best_score:
            best_score = avg_score
            torch.save(agent.network.state_dict(), "perfect_ppo_lunar.pth")
            print(f"🌟 تم حفظ نموذج مثالي! السكور: {best_score:.2f} 🌟")

env.close()


import gymnasium as gym
import imageio
from IPython.display import Video
import torch
import torch.nn.functional as F

# 1. إنشاء البيئة
env_test = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
state, _ = env_test.reset()
done = False
frames = []

# 2. تحميل أفضل أوزان (تأكد من وجود الوكيل agent في الذاكرة)
agent.network.load_state_dict(torch.load("perfect_ppo_lunar.pth"))
agent.network.eval() # تجميد الشبكة

# 3. حلقة الاختبار بالقرار الحتمي (Deterministic)
while not done:
    frames.append(env_test.render())
    
    # 🌟 الحل السحري: تجاوز دالة act واستخراج القرار الصافي (mu) مباشرة
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        x = F.relu(agent.network.actor_fc1(state_tensor))
        x = F.relu(agent.network.actor_fc2(x))
        # نأخذ قيمة mu فقط وهي القرار الأمثل بدون أي عشوائية
        action = torch.tanh(agent.network.mu_layer(x)).cpu().numpy()[0]
        
    next_state, reward, terminated, truncated, _ = env_test.step(action)
    done = terminated or truncated
    state = next_state

env_test.close()

# 4. حفظ وعرض الفيديو
video_path = './perfect_landing.mp4'
imageio.mimsave(video_path, frames, fps=30)
print(f"تم تسجيل الهبوط الحتمي! عدد الإطارات: {len(frames)}")

Video(video_path, embed=True)