import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BartConfig, BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder_pretrained_model="facebook/esm2_t6_8M_UR50D", decoder_embed_size=512, decoder_layers=6,
                 decoder_heads=8, decoder_ffn_dim=2048, num_classes=20):
        super(Seq2SeqModel, self).__init__()

        # Encoder
        self.encoder = AutoModel.from_pretrained(encoder_pretrained_model)

        # Decoder
        self.decoder_embed = nn.Embedding(num_classes, decoder_embed_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=decoder_embed_size, nhead=decoder_heads,
                                                   dim_feedforward=decoder_ffn_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # Output projection layer
        self.output_proj = nn.Linear(decoder_embed_size, num_classes)

    def forward(self, input_ids, decoder_input_ids):
        # Encoder forward pass
        encoder_outputs = self.encoder(input_ids=input_ids).last_hidden_state

        # Prepare decoder inputs
        decoder_embeds = self.decoder_embed(decoder_input_ids)

        # Decoder forward pass
        decoder_outputs = self.decoder(tgt=decoder_embeds, memory=encoder_outputs)

        # Output layer
        logits = self.output_proj(decoder_outputs)

        return logits


class Esm2BartSeq2Seq(nn.Module):
    def __init__(self, encoder_model_name="facebook/esm2_t6_8M_UR50D", decoder_model_name="facebook/bart-base",
                 encoder_output_dim=320, decoder_input_dim=768):
        super(Esm2BartSeq2Seq, self).__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model_name)
        self.decoder = BartForConditionalGeneration.from_pretrained(decoder_model_name)
        # 实例化 tokenizer
        self.tokenizer_out = BartTokenizer.from_pretrained(decoder_model_name)
        self.encoder_to_decoder_linear = nn.Linear(encoder_output_dim, decoder_input_dim)

    def forward(self, input_ids, labels):
        # 生成编码器的注意力掩码，如果未提供，则基于 input_ids
        encoder_attention_mask = input_ids.ne(self.encoder.config.pad_token_id).int()

        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=encoder_attention_mask).last_hidden_state
        # 将encoder 的输出维度映射到 decoder 的维度
        encoder_outputs = self.encoder_to_decoder_linear(encoder_outputs)

        # 准备 decoder 的输入：对 labels 进行右移
        decoder_input_ids = self._shift_right(labels)

        # 如果没有提供 labels_attention_mask，我们需要创建它
        decoder_attention_mask = decoder_input_ids.ne(self.decoder.config.pad_token_id).int()

        # 使用 encoder 的输出和处理后的 decoder 的输入和注意力掩码来进行解码
        outputs = self.decoder(
            attention_mask=encoder_attention_mask,  # 注意这里使用的是 labels_attention_mask
            encoder_outputs=(encoder_outputs,),  # 注意 encoder_outputs 要作为元组传入
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,  # 为解码器提供正确的注意力掩码
            labels=labels  # labels 用于计算损失
        )

        return outputs.loss, outputs.logits

    def _shift_right(self, input_ids):
        # 0   4 3 eos  :labels
        # bos 0 4 3    :shift
        # [[0.43,0.34,0.0543],[0.43,0.34,0.0543],[0.43,0.34,0.0543],[0.43,0.34,0.0543]] pred : [0,0,0,0] [seq_len * vocab_size]
        # [1,2,5,9] label
        """Shift input ids one token to the right for decoder input."""
        start_token_id = self.decoder.config.decoder_start_token_id
        return torch.cat([torch.full_like(input_ids[:, :1], start_token_id), input_ids[:, :-1]], dim=-1)

    def generate(self, input_ids, max_length=512):
        # 生成编码器的注意力掩码，如果未提供，则基于 input_ids
        encoder_attention_mask = input_ids.ne(self.encoder.config.pad_token_id).int()
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=encoder_attention_mask).last_hidden_state
        # 将 encoder 的输出通过一个线性层来调整维度，使之匹配 decoder 的维度
        encoder_outputs = self.encoder_to_decoder_linear(encoder_outputs)
        # 重要的修正: 使用 BaseModelOutput 封装 encoder_outputs
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs)

        # 调用 decoder 的 generate 方法，传入封装后的 encoder_outputs
        generated_ids = self.decoder.generate(encoder_outputs=encoder_outputs, max_length=max_length)

        return generated_ids

    def generate_custom(self, input_ids, max_length=512):
        # 生成编码器的注意力掩码
        encoder_attention_mask = input_ids.ne(self.encoder.config.pad_token_id).int()
        # 获取编码器输出
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=encoder_attention_mask).last_hidden_state
        # 将encoder的输出通过一个线性层来调整维度，使之匹配decoder的维度
        encoder_outputs = self.encoder_to_decoder_linear(encoder_outputs)

        # 使用BaseModelOutput封装encoder_outputs，准备传递给decoder
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs)
        pad_token_id = self.decoder.config.pad_token_id
        # 创建decoder_input_ids，以pad_token_id作为起始token
        # 这里假设decoder_input_ids的长度为max_length，你可以根据需要调整
        decoder_input_ids = torch.full((input_ids.shape[0], max_length), pad_token_id, dtype=torch.long,
                                       device=input_ids.device)

        # 设置decoder的起始token为pad_token_id
        decoder_attention_mask = decoder_input_ids.ne(self.decoder.config.pad_token_id).int()

        # 调用forward方法进行解码
        # 注意：这里我们假设labels参数不是必需的，如果你的forward方法需要labels，可能需要进行相应的调整
        outputs = self.decoder.forward(
            attention_mask=encoder_attention_mask,  # 注意这里使用的是 labels_attention_mask
            encoder_outputs=(encoder_outputs,),  # 注意 encoder_outputs 要作为元组传入
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,  # 为解码器提供正确的注意力掩码
            labels=labels  # labels 用于计算损失
        )

        # 提取logits并生成最终的token IDs
        logits = outputs.logits
        generated_ids = torch.argmax(logits, dim=-1)

        return generated_ids


if __name__ == "__main__":
    model = Esm2BartSeq2Seq(encoder_model_name="facebook/esm2_t6_8M_UR50D", decoder_model_name="facebook/bart-base")
    # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    tokenizer_in = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    tokenizer_out = AutoTokenizer.from_pretrained("facebook/bart-base")

    input_ids = tokenizer_in(
        "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG",
        return_tensors="pt").input_ids
    labels = tokenizer_out(
        "MRVKEKYQHLWRWGWRWGTMLLGMLMICSATEKLWVTVYYGVPVWKEATTTLFCASDAKAYDTEVHNVWATHACVPTDPNPQEVVLVNVTENFNMWKNDMVEQMHEDIISLWDQSLKPCVKLTPLCVSLKCTDLKNDTNTNSSSGRMIMEKGEIKNCSFNISTSIRGKVQKEYAFFYKLDIIPIDNDTTSYKLTSCNTSVITQACPKVSFEPIPIHYCAPAGFAILKCNNKTFNGTGPCTNVSTVQCTHGIRPVVSTQLLLNGSLAEEEVVIRSVNFTDNAKTIIVQLNTSVEINCTRPNNNTRKRIRIQRGPGRAFVTIGKIGNMRQAHCNISRAKWNNTLKQIASKLREQFGNNKTIIFKQSSGGDPEIVTHSFNCGGEFFYCNSTQL",
        return_tensors="pt").input_ids

    # outputs = model(input_ids, labels)
    # print(outputs)
    # print(outputs.shape)  # 输出形状应为 (batch_size, sequence_length, vocab_size)

    generated_seq = model.generate(input_ids)
    print(generated_seq)
