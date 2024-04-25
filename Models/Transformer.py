import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
import torch
import torch.nn as nn
from positional_encodiong import PositionalEncoding
from EncoderLayer import EncoderLayer


class Transformer(pl.LightningModule):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()

        # The constructor takes the following parameters:

        # src_vocab_size: Source vocabulary size.
        # tgt_vocab_size: Target vocabulary size.
        # d_model: The dimensionality of the model's embeddings.
        # num_heads: Number of attention heads in the multi-head attention mechanism.
        # num_layers: Number of layers for both the encoder and the decoder.
        # d_ff: Dimensionality of the inner layer in the feed-forward network.
        # max_seq_length: Maximum sequence length for positional encoding.
        # dropout: Dropout rate for regularization.

        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.fc = nn.Linear(d_model, tgt_vocab_size)

        self.output_activation = nn.Sigmoid()
          
        self.dropout = nn.Dropout(dropout)
        # for validation/testing
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.f1=torchmetrics.classification.BinaryF1Score()
        self.precision=torchmetrics.classification.BinaryPrecision()
        self.recall=torchmetrics.classification.BinaryRecall()
        self.MCC=torchmetrics.MatthewsCorrCoef(task="binary")
        self.AUC=torchmetrics.AUROC(num_classes=2, task= 'binary')
        self.lr = 3.9810717055349735e-05
        self.save_hyperparameters()



    def forward(self, src, tcg, src_mask):

        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        #print("Output shape pre fc:" + str(enc_output.shape))
        cls_output = enc_output[:,0]

        #print("CLS output shape: "+str(cls_output.shape))

        output = self.fc(cls_output)
        #print("Output shape after:" + str(output.shape))

        output = self.output_activation(output.squeeze())

    
        return output
    
    def cross_entropy_loss(self, logits, labels):
      return F.binary_cross_entropy(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y, att = train_batch
        
        logits = self.forward(x,y,att)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss


    def validation_step(self, val_batch, batch_idx):
        x, y, att = val_batch
        logits = self.forward(x,y,att)
        loss = self.cross_entropy_loss(logits, y)
        acc = self.accuracy(logits,y)
        f1_val=self.f1(logits,y)
        precision_val=self.precision(logits,y)
        recall_val=self.recall(logits,y)
        mcc = self.MCC(logits, y)
        auroc = self.AUC(logits, y)
        self.log('val_loss', loss)
        self.log('val_accuracy', acc)
        self.log('val_f1', f1_val)
        self.log('val_precision', precision_val)
        self.log('val_recall', recall_val)
        self.log("val_MCC",mcc)
        self.log("val_AUC",auroc)

    def test_step(self, test_batch, batch_idx):
        x, y, att = test_batch
        logits = self.forward(x,y,att)
        loss = self.cross_entropy_loss(logits, y)
        acc = self.accuracy(logits,y)
        f1_val=self.f1(logits,y)
        precision_val=self.precision(logits,y)
        recall_val=self.recall(logits,y)
        mcc = self.MCC(logits, y)
        auroc = self.AUC(logits, y)
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)
        self.log('test_f1', f1_val)
        self.log('test_precision', precision_val)
        self.log('test_recall', recall_val)
        self.log("test_MCC",mcc)
        self.log("test_AUC",auroc)

    def configure_optimizers(self):
      optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
      return optimizer
    
    