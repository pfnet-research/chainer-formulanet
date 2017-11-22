import chainer
from chainer import dataset
import chainer.functions as F
import chainer.links as L
from chainer import reporter
import collections
import numpy as np
import re
import parser
import tree


DIM = 256


class FP(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.fc = L.Linear(DIM, DIM)
            self.bn = L.BatchNormalization(DIM)

    def __call__(self, x):
        return F.relu(self.bn(self.fc(x)))

class Block(chainer.Chain):
    def __init__(self, n_input):
        super().__init__()
        self._n_input = n_input
        with self.init_scope():
            self.fc1 = L.Linear(DIM*n_input, DIM)
            self.fc2 = L.Linear(DIM, DIM)
            self.bn1 = L.BatchNormalization(DIM)
            self.bn2 = L.BatchNormalization(DIM)

    def __call__(self, *args):
        assert len(args) == self._n_input
        h = F.relu(self.bn1(self.fc1(F.concat(args))))
        h = F.relu(self.bn2(self.fc2(h)))
        return h

# dirty optimization
class Block2(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.fc1a = L.Linear(DIM, DIM, nobias=True)
            self.fc1b = L.Linear(DIM, DIM, nobias=False)
            self.fc2 = L.Linear(DIM, DIM)
            self.bn1 = L.BatchNormalization(DIM)
            self.bn2 = L.BatchNormalization(DIM)

    def __call__(self, arg):
        h = F.relu(self.bn1(arg))
        h = F.relu(self.bn2(self.fc2(h)))
        return h

# dirty optimization
class Block3(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.fc1a = L.Linear(DIM, DIM, nobias=True)
            self.fc1b = L.Linear(DIM, DIM, nobias=True)
            self.fc1c = L.Linear(DIM, DIM, nobias=False)
            self.fc2 = L.Linear(DIM, DIM)
            self.bn1 = L.BatchNormalization(DIM)
            self.bn2 = L.BatchNormalization(DIM)

    def __call__(self, arg):
        h = F.relu(self.bn1(arg))
        h = F.relu(self.bn2(self.fc2(h)))
        return h

class Classifier(chainer.Chain):
    def __init__(self, conditional = True):
        super().__init__()
        self._conditional = conditional
        with self.init_scope():
            self.fc1 = L.Linear(2*DIM if conditional else DIM, DIM)
            self.bn  = L.BatchNormalization(DIM)
            self.fc2 = L.Linear(DIM, 2)

    def __call__(self, *args):
        if self._conditional:
            assert len(args) == 2
        else:
            assert len(args) == 1
        return self.fc2(F.relu(self.bn(self.fc1(F.concat(args)))))

class FormulaNet(chainer.Chain):
    def __init__(self, vocab_size, steps, order_preserving, conditional):
        super().__init__()
        self._vocab_size = vocab_size
        self._steps = steps
        self._order_preserving = order_preserving
        self._conditional = conditional

        with self.init_scope():
            self.embed_id = L.EmbedID(vocab_size, DIM)
            self.FP = FP()
            self.FI = Block2()
            self.FO = Block2()
            if order_preserving:
                self.FH = Block3()
                self.FL = Block3()
                self.FR = Block3()
            self.classifier = Classifier(conditional)

    def __call__(self, minibatch):
        predicted = []
        expected = []
        loss = 0
        for (g_conj, g, y) in minibatch:
            predicted1, loss1 = self._forward1(g_conj, g, y)
            predicted.append(predicted1[-1,:])
            expected.append(1 if y else 0)
            loss += loss1
        self.loss = loss / len(minibatch)
        reporter.report({'loss': self.loss}, self)

        predicted = F.vstack(predicted)
        with chainer.cuda.get_device_from_array(predicted.data):
            expected = self.xp.array(expected, np.int32)
        self.accuracy = F.accuracy(predicted, expected)
        reporter.report({'accuracy': self.accuracy}, self)

        return self.loss

    def _forward1(self, g_conj, g, y):
        x = self._initial_nodes_embedding(g)
        g_embeddings = [F.max(x, axis=0, keepdims=True)]
        if self._conditional:
            x_conj = self._initial_nodes_embedding(g_conj)
            g_conj_embeddings = [F.max(x_conj, axis=0, keepdims=True)]

        for i in range(self._steps):
            x = self._update_nodes_embedding(g, x)
            g_embeddings.append(F.max(x, axis=0, keepdims=True))
            if self._conditional:
                x_conj = self._update_nodes_embedding(g_conj, x_conj)
                g_conj_embeddings.append(F.max(x_conj, axis=0, keepdims=True))
        g_embeddings = F.vstack(g_embeddings)
        if self._conditional:
            g_conj_embeddings = F.vstack(g_conj_embeddings)

        if self._conditional:
            predicted = self.classifier(g_conj_embeddings, g_embeddings)
        else:
            predicted = self.classifier(g_embeddings)

        with chainer.cuda.get_device_from_array(predicted.data):
            if y:
                labels = self.xp.ones(shape=(1+self._steps,), dtype=np.int32)
            else:
                labels = self.xp.zeros(shape=(1+self._steps,), dtype=np.int32)

        return predicted, F.softmax_cross_entropy(predicted, labels)

    def predict(self, g_conj, g):
        return (F.argmax(logit(g_conj, g)) > 0)

    def logit(self, g_conj, g):
        x = self._initial_nodes_embedding(g)
        if self._conditional:
            x_conj = self._initial_nodes_embedding(g_conj)

        for i in range(self._steps):
            x = self._update_nodes_embedding(g, x)
            if self._conditional:
                x_conj = self._update_nodes_embedding(g_conj, x_conj)

        g_embedding = F.max(x, axis=0, keepdims=True)
        if self._conditional:
            return self.classifier(F.max(x_conj, axis=0, keepdims=True), g_embeddings)[0]
        else:
            return self.classifier(g_embeddings)[0]

    def _initial_nodes_embedding(self, g):
        return self.embed_id(g.labels)

    def _update_nodes_embedding(self, g, x):
        x_new = x
        with chainer.cuda.get_device_from_array(x.data):
            zeros_DIM = self.xp.zeros(DIM, dtype=np.float32)

        FI_inputs = []
        FO_inputs = []
        FI_offset = 0
        FO_offset = 0
        FI_offsets = []
        FO_offsets = []

        # dirty optimization
        FI_fc1a_x = self.FI.fc1a(x)
        FI_fc1b_x = self.FI.fc1b(x)
        FO_fc1a_x = self.FO.fc1a(x)
        FO_fc1b_x = self.FO.fc1b(x)

        for v in range(len(g.labels)):
            if len(g.in_edges[v]) > 0:
                FI_inputs.append( FI_fc1a_x[g.in_edges[v]] + F.broadcast_to(FI_fc1b_x[v], (len(g.in_edges[v]), DIM)) )
            FI_offsets.append(FI_offset)
            FI_offset += len(g.in_edges[v])

            if len(g.out_edges[v]) > 0:
                FO_inputs.append( F.broadcast_to(FO_fc1a_x[v], (len(g.out_edges[v]), DIM)) + FO_fc1b_x[g.out_edges[v]] )
            FO_offsets.append(FO_offset)
            FO_offset += len(g.out_edges[v])

        # 速度とバッチ正規化のサンプル数を増やすために、各頂点単位ではなくまとめて実行する
        FI_outputs = self.FI(F.vstack(FI_inputs))
        FO_outputs = self.FO(F.vstack(FO_inputs))

        d = []
        for v in range(len(g.labels)):
            h = F.sum(FI_outputs[FI_offsets[v] : FI_offsets[v] + len(g.in_edges[v]),  :], axis=0) + \
                F.sum(FO_outputs[FO_offsets[v] : FO_offsets[v] + len(g.out_edges[v]), :], axis=0)
            h /= (len(g.in_edges[v]) + len(g.out_edges[v]))
            d.append(h)
        x_new += F.vstack(d)

        if self._order_preserving:
            FL_inputs = []
            FH_inputs = []
            FR_inputs = []
            FL_offset = 0
            FH_offset = 0
            FR_offset = 0
            FL_offsets = []
            FH_offsets = []
            FR_offsets = []

            # dirty optimization
            FL_fc1a_x = self.FL.fc1a(x)
            FL_fc1b_x = self.FL.fc1b(x)
            FL_fc1c_x = self.FL.fc1c(x)
            FH_fc1a_x = self.FH.fc1a(x)
            FH_fc1b_x = self.FH.fc1b(x)
            FH_fc1c_x = self.FH.fc1c(x)
            FR_fc1a_x = self.FR.fc1a(x)
            FR_fc1b_x = self.FR.fc1b(x)
            FR_fc1c_x = self.FR.fc1c(x)

            for v in range(len(g.labels)):
                tbl = g.treeletsL[v]
                if len(tbl) > 0:
                    FL_inputs.append( F.broadcast_to(FL_fc1a_x[v], (tbl.shape[0], DIM)) + FL_fc1b_x[tbl[:,0]] + FL_fc1c_x[tbl[:,1]] )
                FL_offsets.append(FL_offset)
                FL_offset += len(tbl)

                tbl = g.treeletsH[v]
                if len(tbl) > 0:
                    FH_inputs.append( FH_fc1a_x[tbl[:,0]] + F.broadcast_to(FH_fc1b_x[v], (tbl.shape[0], DIM)) + FH_fc1c_x[tbl[:,1]] )
                FH_offsets.append(FH_offset)
                FH_offset += len(tbl)

                tbl = g.treeletsR[v]
                if len(tbl) > 0:
                    FR_inputs.append( FR_fc1a_x[tbl[:,0]] + FR_fc1b_x[tbl[:,1]] + F.broadcast_to(FR_fc1c_x[v], (tbl.shape[0], DIM)) )
                FR_offsets.append(FR_offset)
                FR_offset += len(tbl)

            # 速度とバッチ正規化のサンプル数を増やすために、各頂点単位ではなくまとめて実行する
            FL_outputs = self.FL(F.vstack(FL_inputs))
            FH_outputs = self.FH(F.vstack(FH_inputs))
            FR_outputs = self.FR(F.vstack(FR_inputs))

            d = []
            for v in range(len(g.labels)):
                den = len(g.treeletsL[v]) + len(g.treeletsH[v]) + len(g.treeletsR[v])
                if den == 0:
                    d.append(zeros_DIM)
                else:
                    h = F.sum(FL_outputs[FL_offsets[v] : FL_offsets[v] + len(g.treeletsL[v]), :], axis=0) + \
                        F.sum(FH_outputs[FH_offsets[v] : FH_offsets[v] + len(g.treeletsH[v]), :], axis=0) + \
                        F.sum(FR_outputs[FH_offsets[v] : FR_offsets[v] + len(g.treeletsR[v]), :], axis=0)
                    d.append(h / den)
            x_new += F.vstack(d)

        return self.FP(x_new)


class Dataset(dataset.DatasetMixin):
    def __init__(self, names, datafiles):
        super().__init__()
        self._name_to_id = {name: i for (i, name) in enumerate(names)}
        self._examples = []
        for df in datafiles:
            g1 = self._build_graph(df.conjecture.text)
            for (s,y) in zip(df.examples, df.labels):
                g2 = self._build_graph(s.text)
                self._examples.append((g1,g2,y))

    def __len__(self):
        return len(self._examples)

    def get_example(self, i):
        return self._examples[i]

    def _symbol_to_id(self, sym):
        if re.fullmatch(r'_\d+', sym):
            sym = "_"
        elif re.fullmatch(r'GEN%PVAR%\d+', sym):
            sym = "GEN%PVAR"
        if sym not in self._name_to_id:
            sym = "UNKNOWN"
        return self._name_to_id[sym]

    def _build_graph(self, text):
        g = tree.tree_to_graph(tree.thm_to_tree(parser.thm.parse(text)))
        labels, edges, treelets = g
        nv = len(labels)

        labels = np.array([self._symbol_to_id(l) for l in labels], dtype=np.int32)

        out_edges = [[] for _ in range(nv)]
        in_edges  = [[] for _ in range(nv)]
        for (u,v) in edges:
            out_edges[u].append(v)
            in_edges[v].append(u)
        out_edges = [np.array(vs, dtype=np.int32) for vs in out_edges]
        in_edges  = [np.array(vs, dtype=np.int32) for vs in in_edges]

        treeletsL = [[] for _ in range(nv)]
        treeletsH = [[] for _ in range(nv)]
        treeletsR = [[] for _ in range(nv)]
        for (u,v,w) in treelets:
            treeletsL[u].append((v,w))
            treeletsH[v].append((u,w))
            treeletsR[w].append((u,v))
        treeletsL = [np.array(xs, dtype=np.int32) for xs in treeletsL]
        treeletsH = [np.array(xs, dtype=np.int32) for xs in treeletsH]
        treeletsR = [np.array(xs, dtype=np.int32) for xs in treeletsR]

        return GraphData(
            labels=labels,
            in_edges=in_edges,
            out_edges=out_edges,
            treeletsL=treeletsL,
            treeletsH=treeletsH,
            treeletsR=treeletsR
        )

GraphData = collections.namedtuple(
    "GraphData",
    ["labels", "in_edges", "out_edges", "treeletsL", "treeletsH", "treeletsR"]
)

def convert(minibatch, device = None):
    return [(convert_GraphData(conj, device), convert_GraphData(stmt, device), y) for (conj,stmt,y) in minibatch]

def convert_GraphData(gd, device):
    return GraphData(
        labels  = chainer.dataset.convert.to_device(device, gd.labels),
        in_edges  = [chainer.dataset.convert.to_device(device, ns) for ns in gd.in_edges],
        out_edges = [chainer.dataset.convert.to_device(device, ns) for ns in gd.out_edges],
        treeletsL = [chainer.dataset.convert.to_device(device, tl) for tl in gd.treeletsL],
        treeletsH = [chainer.dataset.convert.to_device(device, tl) for tl in gd.treeletsH],
        treeletsR = [chainer.dataset.convert.to_device(device, tl) for tl in gd.treeletsR],
    )
