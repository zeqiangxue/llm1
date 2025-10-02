

复旦大学研究生课程-AI design作业（姜老师讲授） 
作业：GPT2训练
完成时间：模型训练结束2025年9月24日凌晨3点22分（详见下方图片）
          模型推理结束2025年9月24日早上8点03分（详见下方图片）
早于作业截止时间2025年9月24日晚上课程时间。
==

1.Description:
---
根据网上资料，训练一个82M的中文GPT2模型，使用BERT的Tokenizer.中文语料采用《剑来》小说的前部分章节。训练20个周期，batchsize=8。最终可以续写10句以上的小说。
![1d4da755c634ca387c7602f22124aa46](https://github.com/user-attachments/assets/dde7f295-0a6c-457c-ba91-14107054cd4a)


2.Start:
----
(1)***environment***

首先下载依赖。pip install -r requirements.txt


(2)***dataset***

准备中文语料，放置在./data/文件夹下，将语料由.txt文件更改为input.json文件
《剑来》小说语料来源于微信公众号
按照参考样例./train.json更改input.json文件格式,由于数据集内容为原始的小说内容，包含着大量的非法字符和json读取不支持的控制字符，因此我们对原始数据集文件进行处理，去除其中非法字符，生成预处理好的数据集文件train.json。
```bash
python clr_ctrl.py
```

(3)***Model***

在model_config 定义初始GPT-2模型的超参数配置，
- "initializer_range": 0.02 ： 定义了模型参数（如权重矩阵）在初始化时的标准差，权重会在均值为0，标准差为0.02的正态分布中进行随机初始化。
- "layer_norm_epsilon": 1e-05 ： 用于层归一化的常数，用于避免在归一化过程中出现除以零的情况。设置值为1e-05，用于稳定训练。
- "n_ctx": 1024 ： 表示模型上下文窗口的大小，GPT-2 在生成文本时会考虑的最大序列长度。最大长度设为1024，即模型一次最多能处理1024个 token。
- "n_embd": 768 ： 表示每个token的嵌入维度大小，即模型中词向量的维度。设置为768，即每个词汇的表示向量是768维的。
- "n_head": 12 ： 表示自注意力机制中的注意力头的数量。设置为12，即模型的多头注意力机制中有12个独立的头。
- "n_layer": 10 ： 表示 Transformer 编码器中的层数。在这里，设置为 12，即模型有 12 层堆叠的 Transformer 块。
- "n_positions": 1024 ： 表示模型可以处理的最大位置索引，即序列中的最大位置数。最大位置数为 1024，和 n_ctx一致，表示模型最多能处理1024个位置的token。
- "vocab_size": 13317 ： 表示词汇表的大小，即模型可以识别和生成的词汇数量。在这里，词汇表大小为 21128，表示该模型可以处理的词汇量为21128个不同的 token。
![155c65699a7d8b24f5eab8c663205760](https://github.com/user-attachments/assets/3fe26f72-2ca2-4091-bcc0-b3c6876db397)


(4)***Training***

现在，我们可以使用我们处理好的数据集来训练我们的初始gpt2模型，使用如下命令：
```bash
python train.py   --model_config config/model_config_small.json   --tokenized_data_path data/tokenized/   --tokenizer_path cache/vocab_small.txt   --raw_data_path data/train.json   --epochs 15   --log_step 200   --stride 512   --output_dir model/   --device 0,1   --num_pieces 100   --raw
```

在这个过程中，我们可以看到命令窗口打印出模型的config文件，定义了模型的结构；同时也打印出了模型的参数量，为81894144，约82M

Print Model config
config:
{
  "attn_pdrop": 0.1,
  "embd_pdrop": 0.1,
  "finetuning_task": null,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 10,
  "n_positions": 1024,
  "num_labels": 1,
  "output_attentions": false,
  "output_hidden_states": false,
  "output_past": true,
  "pruned_heads": {},
  "resid_pdrop": 0.1,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "torchscript": false,
  "use_bfloat16": false,
  "vocab_size": 13317
}
number of parameters: 81894144

训练过程中，每个epoch对应的模型都将存储在./model/目录下，最终训练好的模型将存储在./model/final_model/路径中，但因为GitHub上传文件最大限制为100M，故而push时未将final_model参数推至此仓库。

![925b1463e5d61f7a874193633e86836a](https://github.com/user-attachments/assets/74812685-f2e2-42ff-927d-c03a24f906c6)


(5)***Generate***

现在，我们可以使用我们用目标语料训练生成的模型来进行文字生成，使用如下命令：
```bash
python generate.py   --device 1   --length 1000   --tokenizer_path cache/vocab_small.txt   --model_path model/final_model   --prefix "[CLS]陈平安回到家乡"   --topp 1   --temperature 1.0 --save_samples --save_samples_path ./mnt/
```
![f3aba3380c74034bb5f004c9d6df3ce6](https://github.com/user-attachments/assets/8c03d206-2bc7-409b-b385-b723fee39c5e)


3.Result
--
最终会生成10个文字样本，存储在./mnt/目录下，其中之一如下：

======================================== SAMPLE  ========================================

陈平安回到家乡的杨家铺子，找到杨老头，说了一句在外边说的，当然他不愿意与外人多说半句。陈平安听得认真当然没有想到杨老头在那边絮絮叨叨叨。一样米养百样人，不是只是修行境界的，而是这么个青萍剑宗的首席供奉米裕，就是最好的例子。老人也有自己的弟子当中，也有自己的道侣，才会被道侣称为未来宗门的贺小凉。陈平安说道:"回头我们再议论事。"老人与那孩子一起在草席上，在草席上，低头哈腰，眺望南方，神采飞扬。轻声笑道:"白玄他们几个，就这么喜欢钻牛角尖的存在。"孩子双手叉腰，哈哈大笑，"么的么的么的么的事情。"白玄眼睛一臺、白玄双王又照，太笑道:“孩子，都不是剑仙，要不是剑们，白玄你怎么跟我说话，你当我傻贸，"孩子睡了白玄立即笑开了花。小哑巴倒是认真想说，白玄当然说不出话来。陈平安转移话题，"白玄你这孩子，真会装啊，白玄老爷子，就是那个老王八蛋的孙子，白玄他们几个，都能跟我抢弟弟谈心。"白玄双手负后，挺起胸膛沉默片刻，突然一板一眼教训道:"白玄，你是不知道孙春王他们几个了。不过我是很希望白玄以小师弟的身份,与你认错，白玄你这孩子最要好的，就是那个一。"陈平安说道:"白玄他爷爷，你跟着不走，我们不放心。"白玄嗤笑道:"白玄怎么不飞剑啊，你不也不放心啊。"陈平安说道:"白玄你到底是谁，都打算输了曹慈，还不错。"白玄双手负后，仰头望向天幕，哺喃道:"不管白玄怎么说，你都是我们仨都认识的那个白玄吗，还不知道在哪个史书上，白玄就不会是个死字吗?"陈平安说道:"白玄这孩子，就没觉得不自在。"白玄转头望向天幕，喃喃道:"」爷我是不知道的那个大道了，我是真的会输。你是不知道啊。"陈平安转过头，望向天幕。陈灵均双手负后，仰头望天。如今陈平安身边还站着一个背影呢。白玄立即跟着。陈平安问道:"白玄啊，什么时候回家?

==========================================================================================
