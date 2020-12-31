import torch
import random
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter

import math

from transformer import (
    TransformerLayer,
    SinusoidalPositionalEmbedding,
    Embedding,
    MultiheadAttention,
)

import argparse

PAD, BOS, EOS, UNK = "<_>", "<bos>", "<eos>", "<unk>"


class Vocab(object):
    def __init__(self, filename, with_SE):
        with open(filename) as f:
            if with_SE:
                self.itos = [PAD, BOS, EOS, UNK] + [
                    token.strip() for token in f.readlines()
                ]
            else:
                self.itos = [PAD, UNK] + [
                    token.strip() for token in f.readlines()
                ]
        self.stoi = dict(zip(self.itos, range(len(self.itos))))
        self._size = len(self.stoi)
        self._padding_idx = self.stoi[PAD]
        self._unk_idx = self.stoi[UNK]
        self._start_idx = self.stoi.get(BOS, -1)
        self._end_idx = self.stoi.get(EOS, -1)

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self.itos[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self.stoi.get(x, self.unk_idx)

    @property
    def size(self):
        return self._size

    @property
    def padding_idx(self):
        return self._padding_idx

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def start_idx(self):
        return self._start_idx

    @property
    def end_idx(self):
        return self._end_idx


def ListsToTensor(xs, vocab, with_S=False, with_E=False):

    batch_size = len(xs)
    lens = [len(x) + (1 if with_S else 0) + (1 if with_E else 0) for x in xs]
    mx_len = max(max(lens), 1)
    ys = []
    for i, x in enumerate(xs):
        y = (
            ([vocab.start_idx] if with_S else [])
            + [vocab.token2idx(w) for w in x]
            + ([vocab.end_idx] if with_E else [])
            + ([vocab.padding_idx] * (mx_len - lens[i]))
        )
        ys.append(y)

    # lens = torch.LongTensor([ max(1, x) for x in lens])
    data = torch.LongTensor(ys).t_().contiguous()
    return data.cuda()


def batchify(data, vocab_src, vocab_tgt):
    src = ListsToTensor([x[0] for x in data], vocab_src)
    tgt = ListsToTensor([x[1] for x in data], vocab_tgt)
    return src, tgt


class DataLoader(object):
    def __init__(self, filename, vocab_src, vocab_tgt, batch_size, for_train):
        all_data = [
            [x.split() for x in line.strip().split("|")]
            for line in open(filename).readlines()
        ]
        self.data = []
        for d in all_data:
            skip = not (len(d) == 4)
            for j, i in enumerate(d):
                if not for_train:
                    d[j] = i[:500]
                    if len(d[j]) == 0:
                        d[j] = [UNK]
                if len(i) == 0 or len(i) > 500:
                    skip = True
            if not (skip and for_train):
                self.data.append(d)

        self.batch_size = batch_size
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.train = for_train

    def __iter__(self):
        idx = list(range(len(self.data)))
        if self.train:
            random.shuffle(idx)
        cur = 0
        while cur < len(idx):
            data = [self.data[i] for i in idx[cur : cur + self.batch_size]]
            cur += self.batch_size
            yield batchify(data, self.vocab_src, self.vocab_tgt)
        raise StopIteration


def label_smoothed_nll_loss(log_probs, target, eps):
    # log_probs: N x C
    # target: N
    nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
    if eps == 0.0:
        return nll_loss
    smooth_loss = -log_probs.sum(dim=-1)
    eps_i = eps / log_probs.size(-1)
    loss = (1.0 - eps) * nll_loss + eps_i * smooth_loss
    return loss


class Ranker(nn.Module):
    def __init__(
        self,
        vocab_src,
        vocab_tgt,
        embed_dim,
        ff_embed_dim,
        num_heads,
        dropout,
        num_layers,
    ):
        super(Ranker, self).__init__()
        self.transformer_src = nn.ModuleList()
        self.transformer_tgt = nn.ModuleList()
        for i in range(num_layers):
            self.transformer_src.append(
                TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout)
            )
            self.transformer_tgt.append(
                TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout)
            )
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.embed_src_layer_norm = nn.LayerNorm(embed_dim)
        self.embed_tgt_layer_norm = nn.LayerNorm(embed_dim)
        self.embed_src = Embedding(
            vocab_src.size, embed_dim, vocab_src.padding_idx
        )
        self.embed_tgt = Embedding(
            vocab_tgt.size, embed_dim, vocab_tgt.padding_idx
        )
        self.absorber_src = Parameter(torch.Tensor(embed_dim))
        self.absorber_tgt = Parameter(torch.Tensor(embed_dim))
        self.attention_src = MultiheadAttention(
            embed_dim, 1, dropout, weights_dropout=False
        )
        self.attention_tgt = MultiheadAttention(
            embed_dim, 1, dropout, weights_dropout=False
        )
        self.scorer = nn.Linear(embed_dim, embed_dim)
        self.baseline_transformer = nn.Linear(embed_dim, embed_dim)
        self.baseline_scorer = nn.Linear(embed_dim, 1)
        self.dropout = dropout
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.absorber_src, mean=0, std=self.embed_dim ** -0.5)
        nn.init.normal_(self.absorber_tgt, mean=0, std=self.embed_dim ** -0.5)
        nn.init.xavier_uniform_(self.scorer.weight)
        nn.init.xavier_uniform_(self.baseline_transformer.weight)
        nn.init.constant_(self.scorer.bias, 0.0)
        nn.init.constant_(self.baseline_transformer.bias, 0.0)
        nn.init.constant_(self.baseline_scorer.weight, 0.0)
        nn.init.constant_(self.baseline_scorer.bias, 0.0)

    def work(self, src_input, tgt_input):
        beta, s, m = self.forward(src_input, tgt_input, work=True)
        return beta.tolist(), s.tolist(), m.tolist()

    def forward(self, src_input, tgt_input, work=False):
        _, bsz = src_input.size()
        src_emb = self.embed_src_layer_norm(
            self.embed_src(src_input) * self.embed_scale
            + self.embed_positions(src_input)
        )
        tgt_emb = self.embed_tgt_layer_norm(
            self.embed_tgt(tgt_input) * self.embed_scale
            + self.embed_positions(tgt_input)
        )

        src = F.dropout(src_emb, p=self.dropout, training=self.training)
        tgt = F.dropout(tgt_emb, p=self.dropout, training=self.training)

        # seq_len x bsz x embed_dim
        absorber = self.embed_scale * self.absorber_src.unsqueeze(
            0
        ).unsqueeze(0).expand(1, bsz, self.embed_dim)
        src = torch.cat([absorber, src], 0)

        absorber = self.embed_scale * self.absorber_tgt.unsqueeze(
            0
        ).unsqueeze(0).expand(1, bsz, self.embed_dim)
        tgt = torch.cat([absorber, tgt], 0)

        src_padding_mask = src_input.eq(self.vocab_src.padding_idx)
        tgt_padding_mask = tgt_input.eq(self.vocab_tgt.padding_idx)

        absorber = src_padding_mask.data.new(1, bsz).zero_()
        src_padding_mask = torch.cat([absorber, src_padding_mask], 0)
        tgt_padding_mask = torch.cat([absorber, tgt_padding_mask], 0)

        for layer in self.transformer_src:
            src, _, _ = layer(src, self_padding_mask=src_padding_mask)
        for layer in self.transformer_tgt:
            tgt, _, _ = layer(tgt, self_padding_mask=tgt_padding_mask)

        src, src_all = src[:1], src[1:]
        tgt, tgt_all = tgt[:1], tgt[1:]
        src_baseline = self.baseline_scorer(
            torch.tanh(self.baseline_transformer(src.squeeze(0)))
        ).squeeze(1)
        src_padding_mask = src_padding_mask[1:]
        tgt_padding_mask = tgt_padding_mask[1:]

        _, (src_weight, src_v) = self.attention_src(
            src, src_all, src_all, src_padding_mask, need_weights=True
        )
        _, (tgt_weight, tgt_v) = self.attention_tgt(
            tgt, tgt_all, tgt_all, tgt_padding_mask, need_weights=True
        )
        # v: bsz x seq_len x dim
        src_v = src_v + src_emb.transpose(0, 1)
        tgt_v = tgt_v + tgt_emb.transpose(0, 1)
        # w: 1 x bsz x seq_len
        src = torch.bmm(src_weight.transpose(0, 1), src_v).squeeze(1)
        tgt = torch.bmm(tgt_weight.transpose(0, 1), tgt_v).squeeze(1)
        if work:
            # bsz x dim  bsz x seq_len x dim
            s = torch.bmm(tgt_v, self.scorer(src).unsqueeze(2)).squeeze(2)
            max_len = tgt_padding_mask.size(0)
            m = max_len - tgt_padding_mask.float().sum(dim=0).to(
                dtype=torch.int
            )
            beta = tgt_weight.squeeze(0)
            return beta, s, m  # bsz x seq_len, bsz

        src = F.dropout(src, p=self.dropout, training=self.training)
        tgt = F.dropout(tgt, p=self.dropout, training=self.training)
        scores = torch.mm(self.scorer(src), tgt.transpose(0, 1))  # bsz x bsz
        baseline_mse = F.mse_loss(
            src_baseline, scores.mean(dim=1), reduction="mean"
        )

        log_probs = F.log_softmax(scores, -1)
        gold = torch.arange(bsz).cuda()
        _, pred = torch.max(log_probs, -1)
        acc = torch.sum(torch.eq(gold, pred).float()) / bsz
        loss = label_smoothed_nll_loss(log_probs, gold, 0.1)
        loss = loss.mean()

        return loss + baseline_mse, acc


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_src", type=str)
    parser.add_argument("--vocab_tgt", type=str)

    parser.add_argument("--embed_dim", type=int)
    parser.add_argument("--ff_embed_dim", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--dropout", type=float)

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--dev_batch_size", type=int)
    parser.add_argument("--print_every", type=int)
    parser.add_argument("--eval_every", type=int)
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--dev_data", type=str)
    parser.add_argument("--which_ranker", type=str)
    return parser.parse_args()


def update_lr(optimizer, coefficient):
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * coefficient


if __name__ == "__main__":
    random.seed(19940117)
    torch.manual_seed(19940117)
    args = parse_config()
    vocab_src = Vocab(args.vocab_src, with_SE=False)
    vocab_tgt = Vocab(args.vocab_tgt, with_SE=False)

    model = Ranker(
        vocab_src,
        vocab_tgt,
        args.embed_dim,
        args.ff_embed_dim,
        args.num_heads,
        args.dropout,
        args.num_layers,
    )
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    train_data = DataLoader(
        args.train_data, vocab_src, vocab_tgt, args.train_batch_size, True
    )
    dev_data = DataLoader(
        args.dev_data, vocab_src, vocab_tgt, args.dev_batch_size, True
    )

    model.train()
    loss_accumulated = 0.0
    acc_accumulated = 0.0
    batches_processed = 0
    best_dev_acc = 0
    for epoch in range(args.epochs):
        for src_input, tgt_input in train_data:
            optimizer.zero_grad()
            loss, acc = model(src_input, tgt_input)

            loss_accumulated += loss.item()
            acc_accumulated += acc
            batches_processed += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if batches_processed % args.print_every == -1 % args.print_every:
                print(
                    "Batch %d, loss %.5f, acc %.5f"
                    % (
                        batches_processed,
                        loss_accumulated / batches_processed,
                        acc_accumulated / batches_processed,
                    )
                )
            if batches_processed % args.eval_every == -1 % args.eval_every:
                model.eval()
                dev_acc = 0.0
                dev_batches = 0
                for src_input, tgt_input in dev_data:
                    _, acc = model(src_input, tgt_input)
                    dev_acc += acc
                    dev_batches += 1
                dev_acc = dev_acc / dev_batches
                if best_dev_acc < dev_acc:
                    best_dev_acc = dev_acc
                    torch.save(
                        {"args": args, "model": model.state_dict()},
                        "ckpt_persona/epoch%d_batch%d_acc_%.3f"
                        % (epoch, batches_processed, dev_acc),
                    )

                print("Dev Batch %d, acc %.5f" % (batches_processed, dev_acc))
                model.train()
