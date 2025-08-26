### 第四周作业

-   尝试使用本地部署小规模参数量化大模型运行，证明是可以运行的
-   生成的数据集效果不好
-   尝试了qwen3:4b、deepseek:8b、gpt-oss:20b三个，相对而言，gpt-oss:20b生成数据较好
-   最后使用线上deepseek V3.1，生成数据可用

#### 为隐藏api_key，安装dotenv管理环境变量，并将.env文件加入.gitignore中


```bash
conda activate peft # 进入python工作环境
pip install dotenv

```

#### 训练时的参数设置

```python
# 定义全局变量和参数
model_name_or_path = 'THUDM/chatglm3-6b'  # 模型ID或本地路径
train_data_path = 'data/zhouyi_dataset_handmade.csv'    # 训练数据路径
eval_data_path = None                     # 验证数据路径，如果没有则设置为None
seed = 8                                 # 随机种子
max_input_length = 512                    # 输入的最大长度
max_output_length = 1536                  # 输出的最大长度
lora_rank = 8                             # LoRA秩 值原来是16
lora_alpha = 32                           # LoRA alpha值
lora_dropout = 0.05                       # LoRA Dropout率
prompt_text = ''                          # 所有数据前的指令文本


training_args = TrainingArguments( # 训练参数
    output_dir=output_dir,                            # 输出目录
    per_device_train_batch_size=2,                     # 每个设备的训练批量大小, 原来是8，gpu显存不够，改为2
    gradient_accumulation_steps=1,                     # 梯度累积步数
    learning_rate=1e-3,                                # 学习率
    num_train_epochs=train_epochs,                     # 训练轮数
    lr_scheduler_type="linear",                        # 学习率调度器类型
    warmup_ratio=0.1,                                  # 预热比例, 值原来是0.1
    logging_steps=1,                                 # 日志记录步数
    save_strategy="steps",                             # 模型保存策略
    save_steps=10,                                    # 模型保存步数
    optim="adamw_torch",                               # 优化器类型
    fp16=False,                                        # 是否使用混合精度训练,原来是True,改为False，使用bf16
```

重点是`per_device_train_batch_size=2` 大于此设置，16GB显存均显示不足

#### 提交作业清单：

```sh
https://github.com/wang-fuxin/LLM-quickstart-exec/blob/main/chatglm/qlora_chatglm3_timestamp.ipynb
https://github.com/wang-fuxin/LLM-quickstart-exec/blob/main/chatglm/gen_dataset.ipynb
https://github.com/wang-fuxin/LLM-quickstart-exec/blob/main/chatglm/qlora_chatglm3_timestamp.ipynb
https://github.com/wang-fuxin/LLM-quickstart-exec/blob/main/chatglm/qlora_chatglm3.ipynb
https://github.com/wang-fuxin/LLM-quickstart-exec/blob/main/chatglm/qlora_chatglm3_handmade.ipynb
```
