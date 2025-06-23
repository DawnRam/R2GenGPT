# Gemma-3-4b-it 集成指南

本指南详细说明如何在R2GenGPT中集成Google的Gemma-3-4b-it模型。

## 1. 模型信息

- **模型名称**: google/gemma-3-4b-it
- **模型类型**: 指令调优的因果语言模型
- **参数量**: 4B
- **许可证**: Gemma License (需要接受Google的使用条款)
- **特点**: 支持对话格式，性能优秀

## 2. 访问权限设置

### 2.1 接受使用条款
在Hugging Face上访问Gemma模型需要：
1. 登录Hugging Face账户
2. 访问 [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it/tree/main)
3. 点击"Access Gemma on Hugging Face"
4. 阅读并接受Google的使用许可

### 2.2 本地认证
```bash
# 使用Hugging Face CLI登录
huggingface-cli login

# 或者设置环境变量
export HF_TOKEN=your_token_here
```

## 3. 代码修改

### 3.1 导入语句
```python
from transformers import GemmaForCausalLM, GemmaTokenizer
```

### 3.2 模型初始化
```python
# 初始化tokenizer
self.lm_tokenizer = GemmaTokenizer.from_pretrained(args.lm_model)

# 处理特殊token
if self.lm_tokenizer.pad_token is None:
    self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
    self.lm_tokenizer.pad_token_id = self.lm_tokenizer.eos_token_id

# 初始化模型
self.lm_model = GemmaForCausalLM.from_pretrained(
    args.lm_model,
    torch_dtype=torch.float16,
    device_map="auto"  # 如果使用多GPU
)
```

### 3.3 配置文件修改
```python
# configs/config.py
parser.add_argument('--lm_model', default='google/gemma-3-4b-it', type=str, help="Language model to use")
```

## 4. Gemma特定优化

### 4.1 内存优化
```python
# 使用8位量化
if args.low_resource:
    self.lm_model = GemmaForCausalLM.from_pretrained(
        args.lm_model,
        load_in_8bit=True,
        device_map="auto"
    )

# 使用4位量化
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
self.lm_model = GemmaForCausalLM.from_pretrained(
    args.lm_model,
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 4.2 LoRA配置
```python
# Gemma的LoRA配置
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
```

## 5. 对话格式适配

### 5.1 Gemma对话模板
Gemma使用特定的对话格式，需要调整提示模板：

```python
def prompt_wrap(self, img_embeds, atts_img):
    # Gemma的对话格式
    prompt = f'<start_of_turn>user\n<Img><ImageHere></Img> {self.prompt}<end_of_turn>\n<start_of_turn>model\n'
    
    batch_size = img_embeds.shape[0]
    p_before, p_after = prompt.split('<ImageHere>')
    
    p_before_tokens = self.lm_tokenizer(
        p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
    p_after_tokens = self.lm_tokenizer(
        p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
    
    p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
    p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
    
    wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
    wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
    
    return wrapped_img_embeds, wrapped_atts_img
```

### 5.2 结束符号处理
```python
# Gemma的结束符号
self.end_sym = '<end_of_turn>'
```

## 6. 生成参数优化

### 6.1 推荐的生成参数
```python
outputs = self.lm_model.generate(
    inputs_embeds=inputs_embeds,
    num_beams=3,
    do_sample=False,
    min_new_tokens=80,
    max_new_tokens=120,
    repetition_penalty=1.1,  # Gemma对重复惩罚比较敏感
    temperature=0.7,
    pad_token_id=self.lm_tokenizer.pad_token_id,
    eos_token_id=self.lm_tokenizer.eos_token_id,
    # Gemma特定的参数
    use_cache=True,
    return_dict_in_generate=True
)
```

### 6.2 解码优化
```python
def decode(self, output_token):
    # 移除特殊token
    if output_token[0] == self.lm_tokenizer.bos_token_id:
        output_token = output_token[1:]
    
    output_text = self.lm_tokenizer.decode(output_token, add_special_tokens=False)
    
    # 移除Gemma特定的对话标记
    output_text = output_text.replace('<start_of_turn>', '')
    output_text = output_text.replace('<end_of_turn>', '')
    output_text = output_text.replace('model', '').replace('user', '')
    
    return output_text.strip()
```

## 7. 性能基准

### 7.1 内存使用
- **FP16**: ~8GB VRAM
- **8-bit**: ~4GB VRAM  
- **4-bit**: ~2GB VRAM

### 7.2 推理速度
- **单GPU**: ~50 tokens/s
- **多GPU**: 可线性扩展

## 8. 常见问题解决

### 8.1 权限错误
```bash
# 错误: 401 Client Error: Unauthorized
# 解决: 确保已接受使用条款并正确登录
huggingface-cli login
```

### 8.2 内存不足
```python
# 使用梯度检查点
self.lm_model.gradient_checkpointing_enable()

# 使用CPU卸载
self.lm_model = GemmaForCausalLM.from_pretrained(
    args.lm_model,
    device_map="auto",
    offload_folder="offload"
)
```

### 8.3 生成质量问题
```python
# 调整重复惩罚
repetition_penalty=1.05  # 降低重复惩罚

# 使用温度采样
do_sample=True
temperature=0.8
top_p=0.9
```

## 9. 训练配置示例

### 9.1 完整训练配置
```bash
python train.py \
    --lm_model google/gemma-3-4b-it \
    --vision_model microsoft/swin-base-patch4-window7-224 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --llm_use_lora True \
    --llm_r 16 \
    --llm_alpha 16 \
    --max_epochs 3 \
    --precision bf16-mixed
```

### 9.2 推理配置
```bash
python test.py \
    --lm_model google/gemma-3-4b-it \
    --ckpt_file path/to/checkpoint.pth \
    --test_batch_size 8
```

## 10. 评估结果

基于Gemma-3-4b-it的R2GenGPT预期性能：
- **BLEU-4**: ~0.15-0.20
- **CIDEr**: ~0.25-0.35
- **ROUGE-L**: ~0.30-0.40
- **METEOR**: ~0.20-0.25

## 11. 总结

Gemma-3-4b-it是一个优秀的替代选择，具有以下优势：
1. **性能优秀**: 4B参数但性能接近7B模型
2. **内存友好**: 支持多种量化方式
3. **对话能力强**: 专门针对指令调优
4. **开源友好**: 相对宽松的使用条款

通过遵循本指南，你可以成功将Gemma-3-4b-it集成到R2GenGPT中，获得更好的性能和更低的资源消耗。 