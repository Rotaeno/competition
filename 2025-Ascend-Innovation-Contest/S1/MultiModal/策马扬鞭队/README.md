# 昇思模型开发挑战赛（S1赛季)--MultiModal赛题

## 图像预处理优化

+ 问题：原来的图像预处理操作全在cpu上进行，处理完后返回numpy张量，并统一通过框架来申请npu侧张量，导致速度慢，且后面在npu申请张量时阻塞了后面的计算，一定程度拖延了时间。
+ 方案：在算子支持的情况下，提前将读入的图片张量载入npu，并在npu上进行计算，尽可能多地代替原来的numpy计算。
+ 收益：减少了prefill计算时间，也提前了prefill张量申请时间，从而大大提高速度。
例如task1/mindnlp/mindnlp/transformers/models/qwen2_vl/image_processing_qwen2_vl.py中的_preprocess：
```python
def _preprocess(
    self,
    images: Union[ImageInput, VideoInput],
    do_resize: bool = None,
    resample: PILImageResampling = None,
    do_rescale: bool = None,
    rescale_factor: float = None,
    do_normalize: bool = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
    do_convert_rgb: bool = None,
    data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    images = make_list_of_images(images)
    # 原版里，本函数所有操作均在cpu侧执行
    # 此处转为numpy再转到npu上的tensor
    image=  to_numpy_array(images[0])
    img_tensor=mindspore.Tensor(image)
    #使用npu算子进行计算，相比于直接使用numpy，速度提升非常大
    height=img_tensor.shape[0]
    width =img_tensor.shape[1]
    resized_height, resized_width = height, width
    if do_rescale:#还是HWC
        img_tensor=img_tensor.mul(rescale_factor)
            
    if do_normalize:
        img_tensor=(img_tensor-image_mean)/image_std
        img_tensor=mindspore.ops.transpose(img_tensor,(2,0,1))
    img_tensor=mindspore.ops.tile(img_tensor.unsqueeze(0),(2,1,1,1))

    ...

    # 最后函数返回npu侧张量而非numpy张量
        
```

## 融合算子替换

### rotary_pos_emb计算优化

+ 问题：手写的旋转位置编码太慢
+ 方案：使用融合算子rotary_position_embedding替换
+ 收益：减少计算时间

例如task1/mindnlp/mindnlp/transformers/models/llama/modeling_llama.py：

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # 原版：
    # cos = cos.unsqueeze(unsqueeze_dim)
    # sin = sin.unsqueeze(unsqueeze_dim)
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)

    q_embed = mindspore.ops.rotary_position_embedding(q, cos, sin, 0)   
    k_embed = mindspore.ops.rotary_position_embedding(k, cos, sin, 0)   
    return q_embed, k_embed
```

### RMSNorm计算优化

+ 问题：手写的RMSNorm下发大量小算子，效率低
+ 方案：使用融合算子rms_norm替换
+ 收益：减少计算时间

例如task1/mindnlp/mindnlp/transformers/models/qwen2_vl/modeling_qwen2_vl.py中的Qwen2RMSNorm：

```python
def forward(self, hidden_states):
    # 原版：
    # input_dtype = hidden_states.dtype
    # hidden_states = hidden_states.to(mindspore.float32)
    # variance = ops.mean(hidden_states.pow(2), -1, keepdim=True)
    # hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
    # return self.weight * hidden_states.to(input_dtype)
    return F.rms_norm(hidden_states, self.weight, self.variance_epsilon)
```


## Qwen2-VL 视觉语言模型


### rotary_pos_emb计算优化

+ 问题：decode会重复拼接cos和sin。
+ 方案：将cos和sin拼接计算提出到apply_rotary_pos_emb函数外，比如Qwen2VLModel中运行
+ 收益：减少计算时间

```python
def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    # 原版：
    # mrope_section = mrope_section * 2
    # cos = ops.cat([m[i % 3] for i, m in enumerate(ops.split(cos, mrope_section, dim=-1))], dim=-1).unsqueeze(
    #     unsqueeze_dim
    # )
    # sin = ops.cat([m[i % 3] for i, m in enumerate(ops.split(sin, mrope_section, dim=-1))], dim=-1).unsqueeze(
    #     unsqueeze_dim
    # )

    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = mindspore.ops.rotary_position_embedding(q, cos, sin, 0)   
    k_embed = mindspore.ops.rotary_position_embedding(k, cos, sin, 0)
    return q_embed, k_embed
    return q_embed, k_embed
```
### rotate_half计算优化

+ 问题：原始rotatehalf函数使用低效的张量切片操作。
+ 方案：改用算子操作代替
+ 收益：减少计算时间

```python
def rotate_half(self, x: mindspore.Tensor) -> mindspore.Tensor:
    """Rotates half the hidden dims of the input."""
    x1, x2 = ops.split(x, x.shape[-1] // 2, dim=-1)
    return ops.cat((-x2, x1), dim=-1)
```





## Janus-Pro 多模态模型

### self.tokenizer.vocab.get缓存优化

+ 问题：VLChatProcessor中的几个@property会调用self.tokenizer.vocab.get获取数据，这个操作非常耗时
+ 方案：对@property做缓存，只在第一次访问的时候使用self.tokenizer.vocab.get
+ 收益：大幅度减少prefill计算时间

路径：task1/mindnlp/llm/inference/janus_pro/janus/models/processing_vlm.py中的VLChatProcessor

```python
@property
def image_start_id(self):
    if self.my_image_start_id == None:
        self.my_image_start_id = self.tokenizer.vocab.get(self.image_start_tag)
    return self.my_image_start_id
```



## 最终收益

| model_name            | memory_reserved | memory_allocated | avg_prefill_latency | avg_decode_latency   |
| :-------------------- | :-------------- | :--------------- | :------------------ | :------------------- |
| Qwen2-VL-2B-Instruct  | 8.589934592      | 7.129296896      | 0.5031684637069702 | 0.10141904354095459  |
| deepseek-moe-16b-chat | 17.179869184     | 15.287114752      | 0.135939359664917 | 0.05353169441223145 |


## 评测结果

| 评测指标        | 平均得分     |
| --------------- | ------------ |
| 峰值显存得分    |   100     |
| Prefill时延得分 | 315.3732     |
| Decode时延得分  | 109.7689     |
| **总分**        | **175.0474** |
