import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init


dim = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, neg_penalty):
        super(GCN, self).__init__()
        self.in_dim = in_dim  # 输入的维度
        self.out_dim = out_dim  # 输出的维度
        self.neg_penalty = neg_penalty  # 负值
        self.kernel = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.c = 0.85
        self.losses = []

    def forward(self, x, adj):
        # GCN-node
        feature_dim = int(adj.shape[-1])
        eye = torch.eye(feature_dim).to(
            device
        )  # 生成对角矩阵 feature_dim * feature_dim
        if x is None:  # 如果没有初始特征
            AXW = torch.tensordot(
                adj, self.kernel, [[-1], [0]]
            )  # batch_size * num_node * feature_dim
        else:
            XW = torch.tensordot(
                x, self.kernel, [[-1], [0]]
            )  # batch *  num_node * feature_dim
            AXW = torch.matmul(adj, XW)  # batch *  num_node * feature_dim
        # I_cAXW = eye+self.c*AXW
        I_cAXW = eye + self.c * AXW
        y_relu = F.relu(I_cAXW)
        temp = torch.mean(input=y_relu, dim=-2, keepdim=True) + 1e-6
        col_mean = temp.repeat([1, feature_dim, 1])
        y_norm = torch.divide(y_relu, col_mean)  # 正则化后的值
        output = F.softplus(y_norm)
        # output = y_relu
        # 做个尝试
        if self.neg_penalty != 0:
            neg_loss = torch.multiply(
                torch.tensor(self.neg_penalty),
                torch.sum(F.relu(1e-6 - self.kernel)),
            )
            self.losses.append(neg_loss)
        return output


class model_gnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(model_gnn, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        ###################################
        self.gcn1_p = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_p = GCN(hidden_dim, hidden_dim, 0.2)
        # self.gcn3_p = GCN(in_dim, hidden_dim, 0.2)
        self.gcn1_n = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_n = GCN(hidden_dim, hidden_dim, 0.2)
        # self.gcn3_n = GCN(in_dim, hidden_dim, 0.2)
        # ----------------------------------
        self.gcn1_p_1 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_p_1 = GCN(hidden_dim, hidden_dim, 0.2)
        # self.gcn3_p_1 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn1_n_1 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_n_1 = GCN(hidden_dim, hidden_dim, 0.2)
        # self.gcn3_n_1 = GCN(in_dim, hidden_dim, 0.2)
        # ----------------------------------
        self.gcn1_p_2 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_p_2 = GCN(hidden_dim, hidden_dim, 0.2)
        # self.gcn3_p_2 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn1_n_2 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_n_2 = GCN(hidden_dim, hidden_dim, 0.2)
        # self.gcn3_n_2 = GCN(in_dim, hidden_dim, 0.2)
        # ---------------------------------
        self.gcn_p_shared = GCN(in_dim, hidden_dim, 0.2)
        self.gcn_n_shared = GCN(in_dim, hidden_dim, 0.2)
        self.gcn_p_shared1 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn_n_shared1 = GCN(in_dim, hidden_dim, 0.2)
        # --------------------------------- ATT score
        self.Wp_1 = nn.Linear(self.hidden_dim, 1)
        self.Wp_2 = nn.Linear(self.hidden_dim, 1)
        self.Wp_3 = nn.Linear(self.hidden_dim, 1)
        self.Wn_1 = nn.Linear(self.hidden_dim, 1)
        self.Wn_2 = nn.Linear(self.hidden_dim, 1)
        self.Wn_3 = nn.Linear(self.hidden_dim, 1)
        ###################################
        self.kernel_p = nn.Parameter(torch.FloatTensor(dim, in_dim))  #
        self.kernel_n = nn.Parameter(torch.FloatTensor(dim, in_dim))
        self.kernel_p1 = nn.Parameter(torch.FloatTensor(dim, in_dim))  #
        self.kernel_n1 = nn.Parameter(torch.FloatTensor(dim, in_dim))
        self.kernel_p2 = nn.Parameter(torch.FloatTensor(dim, in_dim))  #
        self.kernel_n2 = nn.Parameter(torch.FloatTensor(dim, in_dim))
        # print(self.kernel_p)
        # self.kernel_p = Variable(torch.randn(116, 5)).cuda()  # 116 5
        # self.kernel_n = Variable(torch.randn(116, 5)).cuda()   # 116 5
        ################################################
        self.lin1 = nn.Linear(2 * in_dim * in_dim, 16)
        self.lin2 = nn.Linear(16, self.out_dim)
        self.lin1_1 = nn.Linear(2 * in_dim * in_dim, 16)
        self.lin2_1 = nn.Linear(16, self.out_dim)
        self.lin1_2 = nn.Linear(2 * in_dim * in_dim, 16)
        self.lin2_2 = nn.Linear(16, self.out_dim)
        self.losses = []
        self.losses1 = []
        self.losses2 = []
        self.mseLoss = nn.MSELoss()
        self.reset_weigths()
        self.nums = 3
        # 1 666 3 663
        with open("regions2.txt", "r") as f:
            counts = 0
            tmp_list = []
            for line in f:  # 116
                if counts == 0:
                    counts += 1
                    continue
                tmp = np.zeros(self.nums)
                line.strip("\n")
                line = line.split()

                for columns in range(self.nums):
                    # if columns != 2:
                    #     break
                    tmp[columns] = line[columns]

                tmp_list.append(tmp)
                counts += 1

        self.R = np.array(tmp_list).transpose((1, 0))
        self.R = torch.FloatTensor(self.R)
        self.ij = []
        print(self.R.shape)  # 6*116
        for ri in range(self.nums):
            tmp_sum = 0
            temp = []
            for i in range(dim):
                for j in range(i + 1, dim):
                    if self.R[ri][i] != 0 and self.R[ri][j] != 0:
                        temp.append((i, j))
            self.ij.append(temp)

    def dim_reduce(
        self,
        adj_matrix,
        num_reduce,
        ortho_penalty,
        variance_penalty,
        neg_penalty,
        kernel,
        tell=None,
    ):
        kernel_p = F.relu(kernel)
        batch_size = int(adj_matrix.shape[0])
        AF = torch.tensordot(adj_matrix, kernel_p, [[-1], [0]])
        reduced_adj_matrix = torch.transpose(
            torch.tensordot(kernel_p, AF, [[0], [1]]),  # num_reduce*batch*num_reduce
            1,
            0,
        )  # num_reduce*batch*num_reduce*num_reduce
        kernel_p_tran = kernel_p.transpose(-1, -2)  # num_reduce * column_dim
        gram_matrix = torch.matmul(kernel_p_tran, kernel_p)
        diag_elements = gram_matrix.diag()

        if tell == "A":
            if ortho_penalty != 0:
                ortho_loss_matrix = torch.square(
                    gram_matrix - torch.diag(diag_elements)
                )
                ortho_loss = torch.multiply(
                    torch.tensor(ortho_penalty), torch.sum(ortho_loss_matrix)
                )
                self.losses.append(ortho_loss)

            if variance_penalty != 0:
                variance = diag_elements.var()
                variance_loss = torch.multiply(torch.tensor(variance_penalty), variance)
                self.losses.append(variance_loss)

            if neg_penalty != 0:
                neg_loss = torch.multiply(
                    torch.tensor(neg_penalty),
                    torch.sum(F.relu(torch.tensor(1e-6) - kernel)),
                )
                self.losses.append(neg_loss)
            self.losses.append(0.05 * torch.sum(torch.abs(kernel_p)))
        elif tell == "A1":
            if ortho_penalty != 0:
                ortho_loss_matrix = torch.square(
                    gram_matrix - torch.diag(diag_elements)
                )
                ortho_loss = torch.multiply(
                    torch.tensor(ortho_penalty), torch.sum(ortho_loss_matrix)
                )
                self.losses1.append(ortho_loss)

            if variance_penalty != 0:
                variance = diag_elements.var()
                variance_loss = torch.multiply(torch.tensor(variance_penalty), variance)
                self.losses1.append(variance_loss)

            if neg_penalty != 0:
                neg_loss = torch.multiply(
                    torch.tensor(neg_penalty),
                    torch.sum(F.relu(torch.tensor(1e-6) - kernel)),
                )
                self.losses1.append(neg_loss)
            self.losses1.append(0.05 * torch.sum(torch.abs(kernel_p)))
        elif tell == "A2":
            if ortho_penalty != 0:
                ortho_loss_matrix = torch.square(
                    gram_matrix - torch.diag(diag_elements)
                )
                ortho_loss = torch.multiply(
                    torch.tensor(ortho_penalty), torch.sum(ortho_loss_matrix)
                )
                self.losses2.append(ortho_loss)

            if variance_penalty != 0:
                variance = diag_elements.var()
                variance_loss = torch.multiply(torch.tensor(variance_penalty), variance)
                self.losses2.append(variance_loss)

            if neg_penalty != 0:
                neg_loss = torch.multiply(
                    torch.tensor(neg_penalty),
                    torch.sum(F.relu(torch.tensor(1e-6) - kernel)),
                )
                self.losses2.append(neg_loss)
            self.losses2.append(0.05 * torch.sum(torch.abs(kernel_p)))
        return reduced_adj_matrix

    def reset_weigths(self):
        """reset weights"""
        stdv = 1.0 / math.sqrt(dim)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, A, A1, A2):
        ##############################3
        A = torch.transpose(A, 1, 0)
        s_feature_p = A[0]
        s_feature_n = A[1]
        A1 = torch.transpose(A1, 1, 0)
        s_feature_p1 = A1[0]
        s_feature_n1 = A1[1]
        A2 = torch.transpose(A2, 1, 0)
        s_feature_p2 = A2[0]
        s_feature_n2 = A2[1]
        ###############################

        ###############################
        p_reduce = self.dim_reduce(
            s_feature_p, 10, 0.2, 0.3, 0.1, self.kernel_p, tell="A"
        )
        p_conv1_1_shared = self.gcn_p_shared(None, p_reduce)  # shared GCN
        p_conv1 = self.gcn1_p(None, p_reduce)
        p_conv1 = p_conv1 + p_conv1_1_shared  # sum
        # p_conv1 = torch.cat((p_conv1, p_conv1_1_shared), -1) # concat
        p_conv2_1_shared = self.gcn_p_shared1(p_conv1, p_reduce)
        p_conv2 = self.gcn2_p(p_conv1, p_reduce)
        # p_conv2_1_shared = self.gcn_p_shared1(p_conv2, p_reduce)
        p_conv2 = p_conv2 + p_conv2_1_shared
        # p_conv3 = self.gcn3_p(p_conv2, p_reduce)
        n_reduce = self.dim_reduce(
            s_feature_n, 10, 0.2, 0.5, 0.1, self.kernel_n, tell="A"
        )
        n_conv1_1_shared = self.gcn_n_shared(None, n_reduce)  # shared GCN
        n_conv1 = self.gcn1_n(None, n_reduce)
        n_conv1 = n_conv1 + n_conv1_1_shared  # sum
        # n_conv1 = torch.cat((n_conv1, n_conv1_1_shared), -1) # concat
        n_conv2 = self.gcn2_n(n_conv1, n_reduce)
        n_conv2_1_shared = self.gcn_n_shared1(n_conv1, n_reduce)
        # n_conv2_1_shared = self.gcn_n_shared1(n_conv1, n_reduce)
        n_conv2 = n_conv2 + n_conv2_1_shared
        # n_conv3 = self.gcn3_n(n_conv2, n_reduce)
        # ---------------------------------
        p_reduce1 = self.dim_reduce(
            s_feature_p1, 10, 0.2, 0.3, 0.1, self.kernel_p1, tell="A1"
        )
        p_conv1_2_shared = self.gcn_p_shared(None, p_reduce1)  # shared GCN
        p_conv1_1 = self.gcn1_p_1(None, p_reduce1)
        p_conv1_1 = p_conv1_1 + p_conv1_2_shared  # sum
        # p_conv1_1 = torch.cat((p_conv1_1, p_conv1_2_shared), -1) # concat
        p_conv2_1 = self.gcn2_p_1(p_conv1_1, p_reduce1)
        p_conv2_2_shared = self.gcn_p_shared1(p_conv1_1, p_reduce1)
        # p_conv2_2_shared = self.gcn_p_shared1(p_conv1_1, p_reduce1)
        p_conv2_1 = p_conv2_1 + p_conv2_2_shared
        # p_conv3 = self.gcn3_p(p_conv2, p_reduce)
        n_reduce1 = self.dim_reduce(
            s_feature_n1, 10, 0.2, 0.5, 0.1, self.kernel_n1, tell="A1"
        )
        n_conv1_2_shared = self.gcn_n_shared(None, n_reduce1)  # shared GCN
        n_conv1_1 = self.gcn1_n_1(None, n_reduce1)
        n_conv1_1 = n_conv1_1 + n_conv1_2_shared  # sum
        # n_conv1_1 = torch.cat((n_conv1_1, n_conv1_2_shared), -1) #concat
        n_conv2_1 = self.gcn2_n_1(n_conv1_1, n_reduce1)
        n_conv2_2_shared = self.gcn_n_shared1(n_conv1_1, n_reduce1)
        # n_conv2_2_shared = self.gcn_n_shared1(n_conv1_1, n_reduce1)
        n_conv2_1 = n_conv2_1 + n_conv2_2_shared
        # n_conv3 = self.gcn3_n(n_conv2, n_reduce)
        # ---------------------------------
        p_reduce2 = self.dim_reduce(
            s_feature_p2, 10, 0.2, 0.3, 0.1, self.kernel_p2, tell="A2"
        )
        p_conv1_3_shared = self.gcn_p_shared(None, p_reduce2)  # shared GCN
        p_conv1_2 = self.gcn1_p_2(None, p_reduce2)
        p_conv1_2 = p_conv1_2 + p_conv1_3_shared  # sum
        # p_conv1_2 = torch.cat((p_conv1_2, p_conv1_3_shared), -1) # concat
        p_conv2_2 = self.gcn2_p_2(p_conv1_2, p_reduce2)
        p_conv2_3_shared = self.gcn_p_shared1(p_conv1_2, p_reduce2)
        # p_conv2_3_shared = self.gcn_p_shared1(p_conv1_2, p_reduce2)
        p_conv2_2 = p_conv2_2 + p_conv2_3_shared
        # p_conv3 = self.gcn3_p(p_conv2, p_reduce)
        n_reduce2 = self.dim_reduce(
            s_feature_n2, 10, 0.2, 0.5, 0.1, self.kernel_n2, tell="A2"
        )
        n_conv1_3_shared = self.gcn_n_shared(None, n_reduce2)
        n_conv1_2 = self.gcn1_n_2(None, n_reduce2)
        n_conv1_2 = n_conv1_2 + n_conv1_3_shared  # sum
        # n_conv1_2 = torch.cat((n_conv1_2, n_conv1_3_shared), -1) # concat
        n_conv2_2 = self.gcn2_n_2(n_conv1_2, n_reduce2)
        n_conv2_3_shared = self.gcn_n_shared1(n_conv1_2, n_reduce2)
        # n_conv2_3_shared = self.gcn_n_shared1(n_conv1_2, n_reduce2)
        n_conv2_2 = n_conv2_2 + n_conv2_3_shared
        # n_conv3 = self.gcn3_n(n_conv2, n_reduce)
        # ----------------------------------
        # p_conv1_1_shared = self.gcn_p_shared(None, p_reduce)
        # p_conv1_2_shared = self.gcn_p_shared(None, p_reduce1)
        # p_conv1_3_shared = self.gcn_p_shared(None, p_reduce2)
        #
        # n_conv1_1_shared = self.gcn_n_shared(None, n_reduce)
        # n_conv1_2_shared = self.gcn_n_shared(None, n_reduce1)
        # n_conv1_3_shared = self.gcn_n_shared(None, n_reduce2)
        # -----------------------------------
        # p_conv = p_conv2 + p_conv2_1 + p_conv2_2
        # n_conv = n_conv2 + n_conv2_1 + n_conv2_2
        ##################################

        # conv_concat = torch.cat([p_conv2, n_conv2], -1).reshape([-1, 2 * self.in_dim * self.in_dim])
        conv_concat = torch.cat([p_conv2, n_conv2], -1).reshape(
            [-1, 2 * self.in_dim * self.in_dim]
        )
        conv_concat1 = torch.cat([p_conv2_1, n_conv2_1], -1).reshape(
            [-1, 2 * self.in_dim * self.in_dim]
        )
        conv_concat2 = torch.cat([p_conv2_2, n_conv2_2], -1).reshape(
            [-1, 2 * self.in_dim * self.in_dim]
        )
        output = self.lin2(self.lin1(conv_concat))
        output1 = self.lin2_1(self.lin1_1(conv_concat1))
        output2 = self.lin2_2(self.lin1_2(conv_concat2))
        # output = torch.softmax(output, dim=1)

        # F loss
        simi_loss1 = self.SimiLoss(self.kernel_p, self.kernel_p1)
        simi_loss2 = self.SimiLoss(self.kernel_p, self.kernel_p2)
        simi_loss3 = self.SimiLoss(self.kernel_p1, self.kernel_p2)
        simi_loss4 = self.SimiLoss(self.kernel_n, self.kernel_n1)
        simi_loss5 = self.SimiLoss(self.kernel_n, self.kernel_n2)
        simi_loss6 = self.SimiLoss(self.kernel_n1, self.kernel_n2)
        # simi_loss = squ_p1.sum() + squ_p2.sum() + squ_p3.sum() + squ_n1.sum() + squ_n2.sum() + squ_n3.sum()
        simiLoss = 0.0 * (
            simi_loss6 + simi_loss4 + simi_loss3 + simi_loss2 + simi_loss1 + simi_loss5
        )
        # 0.2 674 0.1 683 0.3 669 0.4 668 0.5 660 0.05 674 0.08 673 0.15 673

        score, score1, score2, score_, score_1, score_2, l1, l2, l3, l4, l5, l6 = (
            self.load_s_c(
                self.kernel_p,
                self.kernel_p1,
                self.kernel_p2,
                self.kernel_n,
                self.kernel_n1,
                self.kernel_n2,
            )
        )
        # score, score1, score2, score_, score_1, score_2 = self.SimiLoss3(self.kernel_p, self.kernel_p1, self.kernel_p2,
        #                                                                  self.kernel_n, self.kernel_n1, self.kernel_n2)
        # 约束score max
        loss1 = -1 * torch.log(score + 1e-3)
        loss2 = -1 * torch.log(score1 + 1e-3)
        loss3 = -1 * torch.log(score2 + 1e-3)
        loss4 = -1 * torch.log(score_ + 1e-3)
        loss5 = -1 * torch.log(score_1 + 1e-3)
        loss6 = -1 * torch.log(score_2 + 1e-3)
        l = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        ll = l1 + l2 + l3 + l4 + l5 + l6
        # 625

        # score = self.load_s_c(self.kernel_p)
        # score1 = self.load_s_c(self.kernel_p1)
        # score2 = self.load_s_c(self.kernel_p2)
        SimiLoss1 = self.SimiLoss2(score, score1)
        SimiLoss2 = self.SimiLoss2(score, score2)
        SimiLoss3 = self.SimiLoss2(score1, score2)
        # score = score.view(6,1)
        # SimiLoss1 = F.cross_entropy(score.view(6, 1), score1.view(6, 1))

        # score_ = self.load_s_c(self.kernel_n)
        # score_1 = self.load_s_c(self.kernel_n1)
        # score_2 = self.load_s_c(self.kernel_n2)
        SimiLoss4 = self.SimiLoss2(score_, score_1)
        SimiLoss5 = self.SimiLoss2(score_, score_2)
        SimiLoss6 = self.SimiLoss2(score_1, score_2)
        simiLoss2 = 0 * (
            SimiLoss1 + SimiLoss6 + SimiLoss5 + SimiLoss4 + SimiLoss2 + SimiLoss3
        )
        simiLoss = simiLoss + l.sum() * 0.00 + ll.sum() * 0.1
        # simiLoss F相似性约束 0.1     l 子网络max    ll 子网络离散化

        # score max 0.01 683 0.001 677
        # 0.01l 0.1ll 663    0.1simiLoss 0.01l 0.1ll 662        0.01 0.01l 666           0.001 0.01 660     0.01 0.1 669
        # 0.01 1 656     0.01 0.5 655
        loss = torch.sum(torch.tensor(self.losses)) + simiLoss
        self.losses.clear()
        loss1 = torch.sum(torch.tensor(self.losses1))
        self.losses1.clear()
        loss2 = torch.sum(torch.tensor(self.losses2))
        self.losses2.clear()
        return output, output1, output2, loss, loss1, loss2

    def SimiLoss(self, F1, F2):
        f1 = F.relu(F1)
        f2 = F.relu(F2).T
        O = torch.matmul(f1, f2)
        # O = O.trace()
        O0 = O.diagonal()
        O1 = F.softmax(O0)
        O2 = torch.log(O1).sum()
        # U = F.relu(O)
        # U1 = U.trace()
        # T = U.sum()
        # simi_loss1 = -torch.log(U1)
        simi_loss1 = -O2

        ####
        # simi_loss1 = 0
        # f1 = F.relu(F1).T
        # f2 = F.relu(F2).T
        # for i in range(8):
        #     L = nn.CrossEntropyLoss()
        #     l = L(f1[i], f2[i])
        #     simi_loss1 += l
        ####
        return simi_loss1

    def SimiLoss2(self, S1, S2):
        # CE loss
        # s1 = S1.unsqueeze(0)
        # s2 = S2.unsqueeze(0).T
        # O = torch.matmul(s1, s2)
        # # O = O.trace()
        # O1 = F.softmax(O)
        # O2 = torch.log(O1)
        # # U = F.relu(O)
        # # U1 = U.trace()
        # # T = U.sum()
        # # simi_loss1 = -torch.log(U1)
        # simi_loss2 = -O2

        # MSE loss
        # s1 = S1.unsqueeze(0)
        # s2 = S2.unsqueeze(0)
        # simi_loss2 = (abs(s1 - s2)).sum()

        s1 = S1.unsqueeze(1)
        s2 = S2.unsqueeze(1)
        # s1 = torch.log(S1.unsqueeze(1) + 0.0001)
        # s2 = torch.log(S2.unsqueeze(1) + 0.0001)
        simi_loss2 = self.mseLoss(s1, s2)
        return simi_loss2

    def load_s_c(self, F, F1, F2, F3, F4, F5):
        F = F.relu(F).T
        F1 = F.relu(F1).T
        F2 = F.relu(F2).T
        F3 = F.relu(F3).T
        F4 = F.relu(F4).T
        F5 = F.relu(F5).T
        s = F  # 5 * 116
        s1 = F1  # 5 * 116
        s2 = F2  # 5 * 116
        s3 = F3  # 5 * 116
        s4 = F4  # 5 * 116
        s5 = F5  # 5 * 116
        s_MAX_INDEX = torch.argmax(s, dim=0)
        s_MAX_INDEX1 = torch.argmax(s1, dim=0)
        s_MAX_INDEX2 = torch.argmax(s2, dim=0)
        s_MAX_INDEX3 = torch.argmax(s3, dim=0)
        s_MAX_INDEX4 = torch.argmax(s4, dim=0)
        s_MAX_INDEX5 = torch.argmax(s5, dim=0)
        # print(s_MAX_INDEX)
        # ss = np.zeros((5, 116))
        ss = torch.zeros((self.in_dim, dim)).to(device)
        ss1 = torch.zeros((self.in_dim, dim)).to(device)
        ss2 = torch.zeros((self.in_dim, dim)).to(device)
        ss3 = torch.zeros((self.in_dim, dim)).to(device)
        ss4 = torch.zeros((self.in_dim, dim)).to(device)
        ss5 = torch.zeros((self.in_dim, dim)).to(device)
        for ii in range(dim):
            ss[s_MAX_INDEX[ii]][ii] = s[s_MAX_INDEX[ii]][ii]
            ss1[s_MAX_INDEX1[ii]][ii] = s1[s_MAX_INDEX1[ii]][ii]
            ss2[s_MAX_INDEX2[ii]][ii] = s2[s_MAX_INDEX2[ii]][ii]
            ss3[s_MAX_INDEX3[ii]][ii] = s3[s_MAX_INDEX3[ii]][ii]
            ss4[s_MAX_INDEX4[ii]][ii] = s4[s_MAX_INDEX4[ii]][ii]
            ss5[s_MAX_INDEX5[ii]][ii] = s5[s_MAX_INDEX5[ii]][ii]
        # s = ss
        # s1 = ss1
        # s2 = ss2
        # s3 = ss3
        # s4 = ss4
        # s5 = ss5
        R_sum = torch.sum(self.R, dim=1)
        scores = torch.zeros(self.nums).to(device)
        scores_ = torch.zeros(self.nums).to(device)
        scores1 = torch.zeros(self.nums).to(device)
        scores1_ = torch.zeros(self.nums).to(device)
        scores2 = torch.zeros(self.nums).to(device)
        scores2_ = torch.zeros(self.nums).to(device)
        scores3 = torch.zeros(self.nums).to(device)
        scores3_ = torch.zeros(self.nums).to(device)
        scores4 = torch.zeros(self.nums).to(device)
        scores4_ = torch.zeros(self.nums).to(device)
        scores5 = torch.zeros(self.nums).to(device)
        scores5_ = torch.zeros(self.nums).to(device)
        for ri in range(self.nums):
            tmp_sum = 0
            tmp_sum1 = 0
            tmp_sum2 = 0
            tmp_sum3 = 0
            tmp_sum4 = 0
            tmp_sum5 = 0

            tmp_sum_ = 0
            tmp_sum_1 = 0
            tmp_sum_2 = 0
            tmp_sum_3 = 0
            tmp_sum_4 = 0
            tmp_sum_5 = 0
            temp = self.ij[ri]
            for ij in temp:
                i = ij[0]
                j = ij[1]

                t = s[:, i] * s[:, j]
                tmp_sum_ += t.sum()
                t1 = s1[:, i] * s1[:, j]
                tmp_sum_1 += t1.sum()
                t2 = s2[:, i] * s2[:, j]
                tmp_sum_2 += t2.sum()
                t3 = s3[:, i] * s3[:, j]
                tmp_sum_3 += t3.sum()
                t4 = s4[:, i] * s4[:, j]
                tmp_sum_4 += t4.sum()
                t5 = s5[:, i] * s5[:, j]
                tmp_sum_5 += t5.sum()

                t = torch.matmul(ss[:, i].unsqueeze(1), ss[:, j].unsqueeze(1).T)
                t = t - torch.diag_embed(torch.diag(t))
                tmp_sum += t.sum()
                t1 = torch.matmul(ss1[:, i].unsqueeze(1), ss1[:, j].unsqueeze(1).T)
                t1 = t1 - torch.diag_embed(torch.diag(t1))
                tmp_sum1 += t1.sum()
                t2 = torch.matmul(ss2[:, i].unsqueeze(1), ss2[:, j].unsqueeze(1).T)
                t2 = t2 - torch.diag_embed(torch.diag(t2))
                tmp_sum2 += t2.sum()
                t3 = torch.matmul(ss3[:, i].unsqueeze(1), ss3[:, j].unsqueeze(1).T)
                t3 = t3 - torch.diag_embed(torch.diag(t3))
                tmp_sum3 += t3.sum()
                t4 = torch.matmul(ss4[:, i].unsqueeze(1), ss4[:, j].unsqueeze(1).T)
                t4 = t4 - torch.diag_embed(torch.diag(t4))
                tmp_sum4 += t4.sum()
                t5 = torch.matmul(ss5[:, i].unsqueeze(1), ss5[:, j].unsqueeze(1).T)
                t5 = t5 - torch.diag_embed(torch.diag(t5))
                tmp_sum5 += t5.sum()

            scores_[ri] = tmp_sum_
            scores1_[ri] = tmp_sum_1
            scores2_[ri] = tmp_sum_2
            scores3_[ri] = tmp_sum_3
            scores4_[ri] = tmp_sum_4
            scores5_[ri] = tmp_sum_5

            scores[ri] = (2 / (R_sum[ri] ** 2)) * tmp_sum
            scores1[ri] = (2 / (R_sum[ri] ** 2)) * tmp_sum1
            scores2[ri] = (2 / (R_sum[ri] ** 2)) * tmp_sum2
            scores3[ri] = (2 / (R_sum[ri] ** 2)) * tmp_sum3
            scores4[ri] = (2 / (R_sum[ri] ** 2)) * tmp_sum4
            scores5[ri] = (2 / (R_sum[ri] ** 2)) * tmp_sum5
        return (
            scores,
            scores1,
            scores2,
            scores3,
            scores4,
            scores5,
            scores_,
            scores1_,
            scores2_,
            scores3_,
            scores4_,
            scores5_,
        )
