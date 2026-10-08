import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
import torch.optim as optim
import os

# ==========================================
# 1. المعمارية المستقرة
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor_fc1 = nn.Linear(state_dim, 256)
        self.actor_fc2 = nn.Linear(256, 256)
        self.mu_layer = nn.Linear(256, action_dim)
        
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
# 2. الوكيل المتوازي (يدعم مصفوفات الـ 16 بيئة معاً)
# ==========================================
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.ppo_epochs = 4
        self.entropy_coef = 0.01 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=3e-4)
        
        self.memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': []}

    def act(self, states):
        # لم نعد نستخدم unsqueeze لأن input هو أصلاً مصفوفة بـ 16 بيئة
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            actions, log_probs, _, values = self.network.get_action_and_value(states)
            
        self.memory['states'].append(states.cpu().numpy())
        self.memory['actions'].append(actions.cpu().numpy())
        self.memory['log_probs'].append(log_probs.cpu().numpy())
        self.memory['values'].append(values.cpu().numpy())
        
        return actions.cpu().numpy()

    def store_reward(self, rewards, dones):
        # تخزين مصفوفات الجوائز والنهايات للـ 16 بيئة
        self.memory['rewards'].append(rewards)
        self.memory['dones'].append(dones)

    def learn(self):
        # الأبعاد هنا هي: (عدد الخطوات، عدد البيئات، عدد الميزات)
        states = torch.tensor(np.array(self.memory['states']), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(self.memory['actions']), dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(np.array(self.memory['log_probs']), dtype=torch.float32).to(self.device)
        old_values = torch.tensor(np.array(self.memory['values']), dtype=torch.float32).to(self.device)
        
        rewards = np.array(self.memory['rewards']) 
        dones = np.array(self.memory['dones']) 
        
        # حساب العوائد لكل بيئة على حدة بشكل متوازٍ
        returns = np.zeros_like(rewards, dtype=np.float32)
        G = np.zeros(rewards.shape[1], dtype=np.float32) # مصفوفة بـ 16 صفر
        
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G * (1 - dones[t])
            returns[t] = G
            
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # تسطيح (Flatten) المصفوفات لتدريب الشبكة العصبية بكتلة واحدة
        states = states.view(-1, states.shape[-1])
        actions = actions.view(-1, actions.shape[-1])
        old_log_probs = old_log_probs.view(-1)
        old_values = old_values.view(-1)
        returns = returns.view(-1)

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
# 3. حلقة التدريب الصاروخية (Vectorized)
# ==========================================
# إنشاء 16 بيئة متوازية (تستهلك قوة المعالج لإنهاء التدريب في ثوانٍ)
NUM_ENVS = 16
envs = gym.make_vec("LunarLanderContinuous-v3", num_envs=NUM_ENVS)

state_dim = envs.single_observation_space.shape[0]
action_dim = envs.single_action_space.shape[0]

agent = PPOAgent(state_dim, action_dim)

# الإعدادات الجديدة للوقت:
# الإعدادات:
NUM_ENVS = 16
envs = gym.make_vec("LunarLanderContinuous-v3", num_envs=NUM_ENVS)

state_dim = envs.single_observation_space.shape[0]
action_dim = envs.single_action_space.shape[0]

agent = PPOAgent(state_dim, action_dim)
initial_lr = 3e-4 # السرعة الابتدائية

N_STEPS = 250
UPDATES = 300 

scores = []
best_score = 100.0 # 🌟 سنبدأ الحفظ من 100 وكلما جاب أعلى نحفظه

current_episode_returns = np.zeros(NUM_ENVS) 

print(f"🚀 بدء التدريب المتوازي مع التبريد الذكي (Annealing)...")

states, _ = envs.reset()

for update in range(1, UPDATES + 1):
    for step in range(N_STEPS):
        actions = agent.act(states)
        next_states, rewards, terminated, truncated, _ = envs.step(actions)
        dones = terminated | truncated
        
        agent.store_reward(rewards, dones)
        
        current_episode_returns += rewards
        for i in range(NUM_ENVS):
            if dones[i]:
                scores.append(current_episode_returns[i])
                current_episode_returns[i] = 0 
                
        states = next_states
        
    agent.learn()
    
    # ==========================================
    # 🌟 التعديلات الهندسية الذكية 🌟
    # ==========================================
    # 1. إبطاء تلاشي العشوائية لتبقي فضوله حياً لفترة أطول (0.98 بدلاً من 0.95)
    agent.entropy_coef = max(0.0005, agent.entropy_coef * 0.98) 
    
    # 2. التبريد التدريجي لسرعة التعلم (Linear LR Decay)
    # السرعة ستقل تدريجياً لتصل إلى الصفر تقريباً في آخر تحديث
    frac = 1.0 - (update - 1.0) / UPDATES
    current_lr = initial_lr * frac
    for param_group in agent.optimizer.param_groups:
        param_group['lr'] = max(1e-5, current_lr) # لا تقل عن 1e-5
    # ==========================================
    
    if len(scores) >= 20:
        avg_score = np.mean(scores[-20:])
        print(f"التحديث: {update}/{UPDATES} | متوسط السكور: {avg_score:.2f} | العشوائية: {agent.entropy_coef:.5f} | السرعة: {current_lr:.6f}")
        
        # 🌟 حفظ ذكي: إذا جاب سكور أعلى من best_score (الذي يبدأ بـ 100)، نحفظه ونرفع العتبة
        if avg_score > best_score:
            best_score = avg_score
            torch.save(agent.network.state_dict(), "perfect_ppo_lunar_vectorized.pth")
            print(f"🏆 تم تسجيل وحفظ رقم قياسي جديد! السكور: {best_score:.2f} 🏆")

envs.close()
# ==========================================
# 4. حفظ وعرض الفيديو بالهبوط الحتمي
# ==========================================
import imageio
from IPython.display import Video

env_test = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
state, _ = env_test.reset()
done = False
frames = []

# تحميل أوزان التدريب المتوازي
try:
    agent.network.load_state_dict(torch.load("perfect_ppo_lunar_vectorized.pth"))
except FileNotFoundError:
    print("النموذج لم يصل لـ 150 بعد، يتم عرض آخر مستوى وصل إليه...")

agent.network.eval() 

while not done:
    frames.append(env_test.render())
    
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        x = F.relu(agent.network.actor_fc1(state_tensor))
        x = F.relu(agent.network.actor_fc2(x))
        action = torch.tanh(agent.network.mu_layer(x)).cpu().numpy()[0]
        
    next_state, reward, terminated, truncated, _ = env_test.step(action)
    done = terminated or truncated
    state = next_state

env_test.close()

video_path = './perfect_landing_vectorized.mp4'
imageio.mimsave(video_path, frames, fps=30)
print(f"تم تسجيل الهبوط الحتمي! عدد الإطارات: {len(frames)}")

Video(video_path, embed=True)