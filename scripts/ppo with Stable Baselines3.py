pip install stable-baselines3[extra]

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# ==========================================
# 1. إنشاء 16 بيئة متوازية بأسطر قليلة
# ==========================================
env = make_vec_env("LunarLanderContinuous-v3", n_envs=16)

# ==========================================
# 2. بناء العقل (PPO) 
# ==========================================
# جميع المعاملات التي كتبناها يدوياً موجودة هنا كمتغيرات جاهزة
model = PPO(
    "MlpPolicy",         # استخدم شبكة عصبية قياسية (الممثل والناقد)
    env,                 # البيئة
    learning_rate=3e-4,  # سرعة التعلم
    n_steps=250,         # عدد الخطوات لكل سفينة
    ent_coef=0.005,      # معامل العشوائية (الإنتروبيا)
    verbose=1,           # طباعة السجلات بشكل جميل
    device="cuda"        # استخدم كرت الشاشة
)

# ==========================================
# 3. حلقة التدريب الصاروخية
# ==========================================
print("🚀 بدء التدريب باستخدام محرك Stable Baselines3...")
# بدلاً من كتابة حلقات for معقدة، سطر واحد يقوم بكل شيء!
# 600 ألف خطوة تعادل الـ 150 تحديث التي قمنا بها سابقاً
model.learn(total_timesteps=600_000, progress_bar=True) 

# ==========================================
# 4. الحفظ والاختبار
# ==========================================
model.save("ppo_lunar_sb3_production")
print("🌟 تم حفظ النموذج بنجاح!")

# دالة جاهزة لاختبار الوكيل 10 مرات وإعطائك متوسط السكور بدقة
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"متوسط السكور النهائي: {mean_reward:.2f} +/- {std_reward:.2f}")


import gymnasium as gym
import imageio
from IPython.display import Video
from stable_baselines3 import PPO

# 1. إنشاء بيئة الاختبار
env_test = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
state, _ = env_test.reset()
done = False
frames = []

# 2. تحميل نموذج الإنتاج الذي صنعناه
model = PPO.load("ppo_lunar_sb3_production")

# 3. حلقة التصوير
while not done:
    frames.append(env_test.render())
    
    # السحر هنا: نطلب من SB3 القرار الحتمي (بدون عشوائية) بهدوء تام
    action, _states = model.predict(state, deterministic=True)
    
    next_state, reward, terminated, truncated, _ = env_test.step(action)
    done = terminated or truncated
    state = next_state

env_test.close()

# 4. حفظ وعرض الفيديو
video_path = './sb3_perfect_landing.mp4'
imageio.mimsave(video_path, frames, fps=30)
print(f"تم تسجيل الهبوط الإنتاجي! عدد الإطارات: {len(frames)}")

Video(video_path, embed=True)