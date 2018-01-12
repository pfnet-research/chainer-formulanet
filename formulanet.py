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

    def __call__(self, gs, minibatch):
        #import time
        #from datetime import datetime
        #start = time.time()
        #print("__call__ enter: " + str(datetime.now()))
        
        predicted, loss = self._forward(gs, minibatch)
        self.loss = loss
        reporter.report({'loss': self.loss}, self)

        with chainer.cuda.get_device_from_array(predicted.data):
            expected = self.xp.array([1 if y else 0 for (conj, stmt, y) in minibatch], np.int32)
        self.accuracy = F.accuracy(predicted, expected)
        reporter.report({'accuracy': self.accuracy}, self)

        #print("__call__ exit: " + str(datetime.now()))
        #print("elapsed_time: {0} sec".format(time.time() - start))

        return loss

    def _forward(self, gs, minibatch):
        stmt_embeddings = []
        conj_embeddings = []
        labels = []

        def collect_embedding():
            es = [self._compute_graph_embedding(gs, x, i) for i in range(len(gs.node_ranges))]
            for (conj, stmt, y) in minibatch:
                stmt_embeddings.append(es[stmt])
                if self._conditional:
                    conj_embeddings.append(es[conj])
                labels.append(1 if y else 0)

        x = self._initial_nodes_embedding(gs)
        collect_embedding()
        for i in range(self._steps):
            # print("step " + str(i))
            x = self._update_nodes_embedding(gs, x)
            collect_embedding()

        stmt_embeddings = F.vstack(stmt_embeddings)
        if self._conditional:
            conj_embeddings = F.vstack(conj_embeddings)

        if self._conditional:
            predicted = self.classifier(conj_embeddings, stmt_embeddings)
        else:
            predicted = self.classifier(stmt_embeddings)

        with chainer.cuda.get_device_from_array(predicted.data):
            labels = self.xp.array(labels, dtype=np.int32)

        return predicted[-len(minibatch):], F.softmax_cross_entropy(predicted, labels)

    def predict(self, gs, conj, stmt):
        return (F.argmax(logit(gs, conj, stmt)) > 0)

    def logit(self, gs, conj, stmt):
        x = self._initial_nodes_embedding(gs)
        for i in range(self._steps):
            x = self._update_nodes_embedding(gs, x)

        stmt_embedding = self._compute_graph_embedding(gs, x, stmt)
        if self._conditional:
            conj_emedding = self._compute_graph_embedding(gs, x, conj)
            return self.classifier(conj_embedding, stmt_embedding)[0]
        else:
            return self.classifier(stmt_embedding)[0]

    def _initial_nodes_embedding(self, gs):
        return self.embed_id(gs.labels)

    def _update_nodes_embedding(self, gs, x):
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

        for v in range(len(gs.labels)):
            if len(gs.in_edges[v]) > 0:
                FI_inputs.append( FI_fc1a_x[gs.in_edges[v]] + F.broadcast_to(FI_fc1b_x[v], (len(gs.in_edges[v]), DIM)) )
            FI_offsets.append(FI_offset)
            FI_offset += len(gs.in_edges[v])

            if len(gs.out_edges[v]) > 0:
                FO_inputs.append( F.broadcast_to(FO_fc1a_x[v], (len(gs.out_edges[v]), DIM)) + FO_fc1b_x[gs.out_edges[v]] )
            FO_offsets.append(FO_offset)
            FO_offset += len(gs.out_edges[v])

        # 速度とバッチ正規化のサンプル数を増やすために、各頂点単位ではなくまとめて実行する
        FI_outputs = self.FI(F.vstack(FI_inputs))
        FO_outputs = self.FO(F.vstack(FO_inputs))

        d = []
        for v in range(len(gs.labels)):
            h = F.sum(FI_outputs[FI_offsets[v] : FI_offsets[v] + len(gs.in_edges[v]),  :], axis=0) + \
                F.sum(FO_outputs[FO_offsets[v] : FO_offsets[v] + len(gs.out_edges[v]), :], axis=0)
            h /= (len(gs.in_edges[v]) + len(gs.out_edges[v]))
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

            for v in range(len(gs.labels)):
                tbl = gs.treeletsL[v]
                if len(tbl) > 0:
                    FL_inputs.append( F.broadcast_to(FL_fc1a_x[v], (tbl.shape[0], DIM)) + FL_fc1b_x[tbl[:,0]] + FL_fc1c_x[tbl[:,1]] )
                FL_offsets.append(FL_offset)
                FL_offset += len(tbl)

                tbl = gs.treeletsH[v]
                if len(tbl) > 0:
                    FH_inputs.append( FH_fc1a_x[tbl[:,0]] + F.broadcast_to(FH_fc1b_x[v], (tbl.shape[0], DIM)) + FH_fc1c_x[tbl[:,1]] )
                FH_offsets.append(FH_offset)
                FH_offset += len(tbl)

                tbl = gs.treeletsR[v]
                if len(tbl) > 0:
                    FR_inputs.append( FR_fc1a_x[tbl[:,0]] + FR_fc1b_x[tbl[:,1]] + F.broadcast_to(FR_fc1c_x[v], (tbl.shape[0], DIM)) )
                FR_offsets.append(FR_offset)
                FR_offset += len(tbl)

            # 速度とバッチ正規化のサンプル数を増やすために、各頂点単位ではなくまとめて実行する
            FL_outputs = self.FL(F.vstack(FL_inputs))
            FH_outputs = self.FH(F.vstack(FH_inputs))
            FR_outputs = self.FR(F.vstack(FR_inputs))

            d = []
            for v in range(len(gs.labels)):
                den = len(gs.treeletsL[v]) + len(gs.treeletsH[v]) + len(gs.treeletsR[v])
                if den == 0:
                    d.append(zeros_DIM)
                else:
                    h = F.sum(FL_outputs[FL_offsets[v] : FL_offsets[v] + len(gs.treeletsL[v]), :], axis=0) + \
                        F.sum(FH_outputs[FH_offsets[v] : FH_offsets[v] + len(gs.treeletsH[v]), :], axis=0) + \
                        F.sum(FR_outputs[FH_offsets[v] : FR_offsets[v] + len(gs.treeletsR[v]), :], axis=0)
                    d.append(h / den)
            x_new += F.vstack(d)

        return self.FP(x_new)

    def _compute_graph_embedding(self, gs, x, stmt):
        (beg,end) = gs.node_ranges[stmt]
        return F.max(x[beg:end], axis=0, keepdims=True)


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


GraphsData = collections.namedtuple(
    "GraphsData",
    ["node_ranges", "labels", "in_edges", "out_edges", "treeletsL", "treeletsH", "treeletsR"]
)

def convert(minibatch, device = None):
    node_offset = 0
    node_ranges = []
    table = {}

    labels = []
    in_edges = []
    out_edges = []
    treeletsL = []
    treeletsH = []
    treeletsR = []

    def f(gd):
        nonlocal node_offset
        if id(gd) in table:
            return table[id(gd)]

        labels.append(gd.labels)
        for ns in gd.in_edges:
            in_edges.append(chainer.dataset.convert.to_device(device, ns + node_offset))
        for ns in gd.out_edges:
            out_edges.append(chainer.dataset.convert.to_device(device, ns + node_offset))
        for tl in gd.treeletsL:
            treeletsL.append(chainer.dataset.convert.to_device(device, tl + node_offset))
        for tl in gd.treeletsH:
            treeletsH.append(chainer.dataset.convert.to_device(device, tl + node_offset))
        for tl in gd.treeletsR:
            treeletsR.append(chainer.dataset.convert.to_device(device, tl + node_offset))

        ret = len(node_ranges)
        node_ranges.append( (node_offset, node_offset+len(gd.labels)) )
        node_offset += len(gd.labels)
        table[id(gd)] = ret

        return ret

    minibatch = [(f(conj), f(stmt), y) for (conj,stmt,y) in minibatch]

    gs = GraphsData(
        node_ranges = node_ranges,
        labels      = chainer.dataset.convert.to_device(device, np.concatenate(labels)),
        in_edges    = in_edges,
        out_edges   = out_edges,
        treeletsL   = treeletsL,
        treeletsH   = treeletsH,
        treeletsR   = treeletsR,
    )

    return (gs, minibatch)
