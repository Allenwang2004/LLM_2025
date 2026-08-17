import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

# 載入預訓練模型和 tokenizer
model_name = "gpt2"  # 可以使用更大的模型，如 "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 創建模型 with value head for PPO
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

# PPO 配置
config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-5,
    batch_size=128,
    mini_batch_size=128,
    gradient_accumulation_steps=1,
    optimize_cuda_cache=True,
    seed=42,
)

# 創建 PPO trainer
ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer)

# 定義一個簡單的獎勵函數 (在真實場景中，這會是一個從人類回饋訓練的獎勵模型)
def reward_function(response):
    # 簡單的獎勵：基於回應長度 (長回應得更高分)
    return len(response) * 0.01

# 訓練數據：一些提示
prompts = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a short story about a robot.",
    "What are the benefits of exercise?",
    "Describe your favorite food."
]

# 訓練循環
for epoch in range(10):  # 簡單的幾個 epoch
    for prompt in prompts:
        # 將提示編碼
        query_tensor = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # 生成回應
        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens": 20,
        }
        response_tensor = ppo_trainer.generate(query_tensor, **generation_kwargs)
        response = tokenizer.decode(response_tensor[0])

        # 計算獎勵
        reward = reward_function(response)

        # PPO 步驟
        train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], [reward])

    print(f"Epoch {epoch + 1} completed")

# 保存微調後的模型
model.save_pretrained("ppo_finetuned_model")
tokenizer.save_pretrained("ppo_finetuned_model")

print("PPO fine-tuning completed!")
