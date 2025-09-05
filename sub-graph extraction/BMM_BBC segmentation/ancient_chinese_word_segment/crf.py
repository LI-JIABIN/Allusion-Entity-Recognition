# -*- encoding: utf-8 -*-
 
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def log_sum_exp_batch(log_Tensor, axis=-1): # shape (batch_size,n,m)
    #计算对数概率值的和，并对结果进行一个批次内的归一化操作，避免了由于概率值较小或较大导致的数值溢出或下溢等问题
    return torch.max(log_Tensor, axis)[0] + \
        torch.log(torch.exp(log_Tensor-torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0],-1,1)).sum(axis))

class Bert_BiLSTM_CRF(nn.Module):
    def __init__(self, tag_to_ix, hidden_dim=768):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.lstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=hidden_dim//2, batch_first=True)
        self.transitions = nn.Parameter(torch.randn(
            self.tagset_size, self.tagset_size
        ))  #在PyTorch中，可以使用nn.Parameter()将转移矩阵封装成参数，以便在反向传播过程中更新参数值。
        # 先初始化 hidden_dim
        self.hidden_dim = hidden_dim
        # self.hidden = self.init_hidden()
        self.start_label_id = self.tag_to_ix['[CLS]']
        self.end_label_id = self.tag_to_ix['[SEP]']


        self.fc = nn.Linear(hidden_dim, self.tagset_size)
        self.bert = BertModel.from_pretrained(
            r"H:\Pycharm Project\Local\data_process\auto_annotate_poetry\ancient_chinese_word_segment\bert-ancient-chinese")
        
        self.transitions.data[self.start_label_id, :] = -10000
        self.transitions.data[:, self.end_label_id] = -10000
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.transitions.to(self.device)
        


    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))


    def _forward_alg(self, feats):
        '''
        this also called alpha-recursion or forward recursion, to calculate log_prob of all barX 
        '''
        #_forward_alg(self, feats)函数是CRF模型中前向传播算法的实现。它的作用是计算所有可能的标注序列（路径）的概率，并返回这些概率的对数值。
        
        # T = self.max_seq_length
        T = feats.shape[1]  
        batch_size = feats.shape[0]
        
        # alpha_recursion,forward, alpha(zt)=p(zt,bar_x_1:t)
        log_alpha = torch.Tensor(batch_size, 1, self.tagset_size).fill_(-10000.).to(self.device)  #[batch_size, 1, 16]
        #log_alpha是一个动态规划表格，由于需要对动态规划表格进行赋值操作，因此需要先将整个表格初始化为一个较小的负数，以确保在计算时不会引入无效的路径。
        # normal_alpha_0 : alpha[0]=Ot[0]*self.PIs
        # self.start_label has all of the score. it is log,0 is p=1
        log_alpha[:, 0, self.start_label_id] = 0
        
        # feats: sentances -> word embedding -> lstm -> MLP -> feats
        # feats is the probability of emission, feat.shape=(1,tag_size)
        for t in range(1, T):  #得到以当前位置为终点的所有可能路径的概率，最终得到用于计算动态规划表格中每个元素的值。
            log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)  #转移概率+发射概率

        # log_prob of all barX
        log_prob_all_barX = log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

        
    def _score_sentence(self, feats, label_ids):
        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size,self.tagset_size,self.tagset_size)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0],1)).to(self.device)
        # the 0th node is start_label->start_word,the probability of them=1. so t begin with 1.
        for t in range(1, T):
            score = score + \
                batch_transitions.gather(-1, (label_ids[:, t]*self.tagset_size+label_ids[:, t-1]).view(-1,1)) \
                    + feats[:, t].gather(-1, label_ids[:, t].view(-1,1)).view(-1,1)
        return score

    def _bert_enc(self, x):
        """
        x: [batchsize, sent_len]
        enc: [batch_size, sent_len, 768]
        """
        with torch.no_grad():
            encoded_layer, _  = self.bert(x)
            enc = encoded_layer[-1]
        return enc

    def _viterbi_decode(self, feats):
        '''
        Max-Product Algorithm or viterbi algorithm, argmax(p(z_0:t|x_0:t))
        '''
        
        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # batch_transitions=self.transitions.expand(batch_size,self.tagset_size,self.tagset_size)

        log_delta = torch.Tensor(batch_size, 1, self.tagset_size).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0.
        
        # psi is for the vaule of the last latent that make P(this_latent) maximum.
        psi = torch.zeros((batch_size, T, self.tagset_size), dtype=torch.long)  # psi[0]=0000 useless
        for t in range(1, T):
            # delta[t][k]=max_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # delta[t] is the max prob of the path from  z_t-1 to z_t[k]
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            # psi[t][k]=argmax_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # psi[t][k] is the path choosed from z_t-1 to z_t[k],the value is the z_state(is k) index of z_t-1
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = torch.zeros((batch_size, T), dtype=torch.long)

        # max p(z1:t,all_x|theta)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T-2, -1, -1):
            # choose the state of z_t according the state choosed of z_t+1.
            path[:, t] = psi[:, t+1].gather(-1,path[:, t+1].view(-1,1)).squeeze()
        return max_logLL_allz_allx, path


    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)  #[batch_size, max_len, 16]
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return torch.mean(forward_score - gold_score)


    def _get_lstm_features(self, sentence):
        """sentence is the ids"""
        # self.hidden = self.init_hidden()
        embeds = self._bert_enc(sentence)  # [8, 75, 768]
        # 过lstm
        enc, _ = self.lstm(embeds)
        lstm_feats = self.fc(enc)
        # lstm_feats = self.fc(embeds)
        return lstm_feats  # [8, 75, 16]

    def _compute_marginal_probabilities(self, feats):
        T = feats.size(1)
        batch_size = feats.size(0)
        tag_size = self.tagset_size

        # 初始化 alpha
        alpha = torch.full((batch_size, tag_size), -10000., device=self.device)
        alpha[:, self.start_label_id] = 0.

        transitions = self.transitions.unsqueeze(0)  # [1, tag_size, tag_size]

        # 前向算法
        forward_vars = []
        for t in range(T):
            emit_score = feats[:, t, :]  # [batch_size, tag_size]
            alpha_expanded = alpha.unsqueeze(2)  # [batch_size, tag_size, 1]
            emit_score_expanded = emit_score.unsqueeze(1)  # [batch_size, 1, tag_size]
            next_tag_var = alpha_expanded + transitions + emit_score_expanded  # [batch_size, tag_size, tag_size]
            alpha = torch.logsumexp(next_tag_var, dim=1)  # [batch_size, tag_size]
            forward_vars.append(alpha)

        log_Z = torch.logsumexp(alpha, dim=1)  # [batch_size]

        # 后向算法
        beta = torch.full((batch_size, tag_size), -10000., device=self.device)
        beta[:, self.end_label_id] = 0.

        backward_vars = []
        for t in reversed(range(T)):
            emit_score = feats[:, t, :]  # [batch_size, tag_size]
            beta_expanded = beta.unsqueeze(1)  # [batch_size, 1, tag_size]
            emit_score_expanded = emit_score.unsqueeze(2)  # [batch_size, tag_size, 1]
            next_tag_var = beta_expanded + transitions + emit_score_expanded  # [batch_size, tag_size, tag_size]
            beta = torch.logsumexp(next_tag_var, dim=2)  # [batch_size, tag_size]
            backward_vars.insert(0, beta)

        # 计算边缘概率
        log_gamma = []
        for t in range(T):
            alpha_t = forward_vars[t]  # [batch_size, tag_size]
            beta_t = backward_vars[t]  # [batch_size, tag_size]
            log_gamma_t = alpha_t + beta_t  # [batch_size, tag_size]
            log_gamma.append(log_gamma_t)

        log_gamma = torch.stack(log_gamma, dim=1)  # [batch_size, T, tag_size]
        log_marginals = log_gamma - log_Z.unsqueeze(1).unsqueeze(2)
        marginal_probabilities = torch.exp(log_marginals)
        return marginal_probabilities

    def forward(self, sentence, return_probabilities=False):
        lstm_feats = self._get_lstm_features(sentence)  # [batch_size, seq_len, tagset_size]
        score, tag_seq = self._viterbi_decode(lstm_feats)

        if return_probabilities:
            # 计算边缘概率
            tag_probabilities = self._compute_marginal_probabilities(lstm_feats)
            return score, tag_seq, tag_probabilities
        else:
            return score, tag_seq



if __name__=="__main__":
    import torch.optim as optim
    from utils import NerDataset, pad, VOCAB, tokenizer, tag2idx, idx2tag
    from torch.utils import data


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    MAX_LEN = 256 - 2
    model = Bert_BiLSTM_CRF(tag2idx).cuda()

    train_dataset = NerDataset(
        r"H:\Pycharm Project\Local\bert-bilstm-crf\diangu-bert-bilstm-crf\data\allusions\train.txt")

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=192,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr = 0.0005)
    criterion = nn.CrossEntropyLoss(ignore_index=0)


    print('Start Train...,')
    best_f1 = -1
    best_epoch = -1

    model.train()
    for i, batch in enumerate(train_iter):
        words, x, heads,tags, y, seqlens = batch
        x = x.to(device)
        y = y.to(device)
        _y = y  # for monitoring
        optimizer.zero_grad()
        loss = model.neg_log_likelihood(x, y)  # logits: (N, T, VOCAB), y: (N, T)
        loss.backward()

        optimizer.step()

        if i % 10 == 0:  # monitoring
            print(f"step: {i}, loss: {loss.item()}")
