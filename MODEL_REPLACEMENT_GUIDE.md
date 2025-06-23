# R2GenGPT 模型替换指南

本指南详细说明如何将R2GenGPT中的LLaMA模型替换为其他语言模型。

## 1. 支持的模型类型

### 1.1 因果语言模型 (Causal LM)
- **LLaMA系列**: `LlamaForCausalLM`
- **Qwen系列**: `Qwen2ForCausalLM`
- **InternLM系列**: `InternLMForCausalLM`
- **Baichuan系列**: `BaichuanForCausalLM`
- **Yi系列**: `YiForCausalLM`
- **StableLM系列**: `AutoModelForCausalLM` (推荐使用Auto类)

### 1.2 条件生成模型 (Conditional Generation)
- **ChatGLM系列**: `ChatGLMForConditionalGeneration`
- **BLOOM系列**: `BloomForCausalLM`
- **T5系列**: `T5ForConditionalGeneration`

## 2. 替换步骤

### 2.1 修改导入语句

```python
# 原始代码
from transformers import LlamaForCausalLM, LlamaTokenizer

# 替换为ChatGLM3
from transformers import ChatGLMForConditionalGeneration, ChatGLMTokenizer

# 替换为Qwen2
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer

# 替换为InternLM2
from transformers import InternLMForCausalLM, InternLMTokenizer

# 替换为StableLM (推荐使用Auto类)
from transformers import AutoModelForCausalLM, AutoTokenizer
```

### 2.2 修改模型初始化

```python
# 原始代码
self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
self.llama_tokenizer.pad_token_id = 0
self.llama_model = LlamaForCausalLM.from_pretrained(args.llama_model, ...)

# 替换为ChatGLM3
self.lm_tokenizer = ChatGLMTokenizer.from_pretrained(args.lm_model, trust_remote_code=True)
if self.lm_tokenizer.pad_token is None:
    self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
    self.lm_tokenizer.pad_token_id = self.lm_tokenizer.eos_token_id
self.lm_model = ChatGLMForConditionalGeneration.from_pretrained(
    args.lm_model, 
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 替换为Qwen2
self.lm_tokenizer = Qwen2Tokenizer.from_pretrained(args.lm_model, trust_remote_code=True)
self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
self.lm_model = Qwen2ForCausalLM.from_pretrained(
    args.lm_model,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 替换为StableLM (推荐方式)
self.lm_tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
if self.lm_tokenizer.pad_token is None:
    self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
    self.lm_tokenizer.pad_token_id = self.lm_tokenizer.eos_token_id
self.lm_model = AutoModelForCausalLM.from_pretrained(
    args.lm_model,
    torch_dtype="auto",
    device_map="auto"
)
```

### 2.3 修改变量名

将所有`llama_`前缀的变量名修改为通用的`lm_`：

```python
# 修改前 -> 修改后
self.llama_tokenizer -> self.lm_tokenizer
self.llama_model -> self.lm_model
self.llama_proj -> self.lm_proj
inputs_llama -> inputs_lm
atts_llama -> atts_lm
```

### 2.4 修改配置文件

```python
# configs/config.py
# 修改前
parser.add_argument('--llama_model', default='meta-llama/Llama-2-7b-chat-hf', type=str, help="LLM model to use")

# 修改后
parser.add_argument('--lm_model', default='stabilityai/stablelm-3b-4e1t', type=str, help="Language model to use")
```

## 3. 模型特定注意事项

### 3.1 ChatGLM系列

**特殊要求：**
- 需要`trust_remote_code=True`
- 特殊token处理：pad_token通常设置为eos_token
- 某些版本可能需要特殊的输入格式

**示例配置：**
```python
# 模型路径
--lm_model THUDM/chatglm3-6b

# 特殊处理
if self.lm_tokenizer.pad_token is None:
    self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
```

### 3.2 Qwen系列

**特殊要求：**
- 需要`trust_remote_code=True`
- 支持8位量化
- 某些版本可能需要特殊的attention mask处理

**示例配置：**
```python
# 模型路径
--lm_model Qwen/Qwen2-7B-Instruct

# 特殊处理
self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
```

### 3.3 InternLM系列

**特殊要求：**
- 某些版本可能需要特殊的tokenizer配置
- 支持多种量化方式

**示例配置：**
```python
# 模型路径
--lm_model internlm/internlm2-7b-chat

# 特殊处理
self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
```

### 3.4 StableLM系列

**特殊要求：**
- 推荐使用`AutoTokenizer`和`AutoModelForCausalLM`
- 使用`torch_dtype="auto"`而不是固定类型
- 开源友好，无访问限制

**示例配置：**
```python
# 模型路径
--lm_model stabilityai/stablelm-3b-4e1t

# 特殊处理
from transformers import AutoTokenizer, AutoModelForCausalLM
self.lm_tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
self.lm_model = AutoModelForCausalLM.from_pretrained(
    args.lm_model,
    torch_dtype="auto"
)
```

## 4. 生成参数调整

不同模型可能支持不同的生成参数：

```python
# 通用参数（大多数模型都支持）
outputs = self.lm_model.generate(
    inputs_embeds=inputs_embeds,
    num_beams=self.hparams.beam_size,
    do_sample=self.hparams.do_sample,
    min_new_tokens=self.hparams.min_new_tokens,
    max_new_tokens=self.hparams.max_new_tokens,
    repetition_penalty=self.hparams.repetition_penalty,
    temperature=self.hparams.temperature,
    pad_token_id=self.lm_tokenizer.pad_token_id,
    eos_token_id=self.lm_tokenizer.eos_token_id,
)

# 某些模型可能不支持length_penalty，需要注释掉
# length_penalty=self.hparams.length_penalty,
```

## 5. 测试和验证

### 5.1 基本功能测试

```python
# 测试tokenizer
test_text = "Hello world"
tokens = self.lm_tokenizer(test_text, return_tensors="pt")
decoded = self.lm_tokenizer.decode(tokens.input_ids[0])
assert decoded.strip() == test_text

# 测试模型前向传播
with torch.no_grad():
    outputs = self.lm_model(**tokens)
    assert outputs.logits.shape[-1] == self.lm_tokenizer.vocab_size
```

### 5.2 生成测试

```python
# 测试生成功能
test_inputs = torch.randn(1, 10, self.lm_model.config.hidden_size)
outputs = self.lm_model.generate(
    inputs_embeds=test_inputs,
    max_new_tokens=10,
    pad_token_id=self.lm_tokenizer.pad_token_id,
)
assert len(outputs[0]) > 10
```

## 6. 常见问题解决

### 6.1 维度不匹配错误

```python
# 错误：RuntimeError: size mismatch
# 解决：检查投影层维度
print(f"Visual features: {self.visual_encoder.num_features}")
print(f"LM hidden size: {self.lm_model.config.hidden_size}")
self.lm_proj = nn.Linear(self.visual_encoder.num_features, self.lm_model.config.hidden_size)
```

### 6.2 Tokenizer错误

```python
# 错误：KeyError: 'pad_token'
# 解决：设置pad_token
if self.lm_tokenizer.pad_token is None:
    self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
```

### 6.3 生成参数错误

```python
# 错误：TypeError: generate() got an unexpected keyword argument
# 解决：检查模型支持的参数
# 移除不支持的参数或使用try-except包装
try:
    outputs = self.lm_model.generate(..., length_penalty=2.0)
except TypeError:
    outputs = self.lm_model.generate(...)  # 移除length_penalty
```

## 7. 性能优化建议

### 7.1 内存优化

```python
# 使用8位量化
if args.low_resource:
    self.lm_model = AutoModel.from_pretrained(
        args.lm_model,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True
    )

# 使用梯度检查点
self.lm_model.gradient_checkpointing_enable()
```

### 7.2 推理优化

```python
# 使用半精度
self.lm_model.half()

# 使用torch.compile (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    self.lm_model = torch.compile(self.lm_model)
```

## 8. 完整替换示例

以StableLM为例的完整替换：

```python
# 1. 修改导入
from transformers import AutoModelForCausalLM, AutoTokenizer

# 2. 修改初始化
self.lm_tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
if self.lm_tokenizer.pad_token is None:
    self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
    self.lm_tokenizer.pad_token_id = self.lm_tokenizer.eos_token_id

self.lm_model = AutoModelForCausalLM.from_pretrained(
    args.lm_model,
    torch_dtype="auto",
    device_map="auto"
)

# 3. 修改所有变量名
self.lm_proj = nn.Linear(self.visual_encoder.num_features, self.lm_model.config.hidden_size)
self.embed_tokens = self.lm_model.get_input_embeddings()

# 4. 修改配置文件
# --lm_model stabilityai/stablelm-3b-4e1t
```

通过遵循这个指南，你可以成功将R2GenGPT中的LLaMA替换为其他语言模型，同时保持模型的核心功能不变。 