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
            self.FI = Block(2)
            self.FO = Block(2)
            if order_preserving:
                self.FH = Block(3)
                self.FL = Block(3)
                self.FR = Block(3)
            self.classifier = Classifier(conditional)

    def __call__(self, batch):
        loss = 0
        for (g_conj, g, y) in batch:
            loss += self._forward1(g_conj, g, y)
        self.loss = loss / len(batch)
        reporter.report({'loss': self.loss}, self)
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

        if y:
            labels = self.xp.ones(shape=(1+self._steps,), dtype=np.int32)
        else:
            labels = self.xp.zeros(shape=(1+self._steps,), dtype=np.int32)

        return F.softmax_cross_entropy(predicted, labels)

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
        d = []
        for v in range(len(g.labels)):
            h = self.xp.zeros(DIM, dtype=np.float32)
            if len(g.in_edges[v]) > 0:
                h += F.sum(self.FI(x[g.in_edges[v],:], F.broadcast_to(x[v], (len(g.in_edges[v]), DIM))), axis=0)
            if len(g.out_edges[v]) > 0:
                h += F.sum(self.FO(F.broadcast_to(x[v], (len(g.out_edges[v]), DIM)), x[g.out_edges[v]]), axis=0)
            h /= g.degrees[v]
            d.append(h)
        x += F.vstack(d)

        if self._order_preserving:
            d = []
            for v in range(len(g.labels)):
                h = self.xp.zeros(DIM, dtype=np.float32)
                if g.treelets_degree[v] > 0:
                    tbl = g.treeletsL[v]
                    if len(tbl) > 0:
                        h += F.sum(self.FL(F.broadcast_to(x[v], (tbl.shape[0], DIM)), x[tbl[:,0]], x[tbl[:,1]]), axis=0)
                    tbl = g.treeletsH[v]
                    if len(tbl) > 0:
                        h += F.sum(self.FH(x[tbl[:,0]], F.broadcast_to(x[v], (tbl.shape[0], DIM)), x[tbl[:,1]]), axis=0)
                    tbl = g.treeletsR[v]
                    if len(tbl) > 0:
                        h += F.sum(self.FR(x[tbl[:,0]], x[tbl[:,1]], F.broadcast_to(x[v], (tbl.shape[0], DIM))), axis=0)
                    h /= g.treelets_degree[v]
                d.append(h)
            x += F.vstack(d)

        return self.FP(x)


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
        degrees = [0 for _ in range(nv)]
        for (u,v) in edges:
            out_edges[u].append(v)
            in_edges[v].append(u)
            degrees[u] += 1
            degrees[v] += 1
        out_edges = [np.array(vs, dtype=np.int32) for vs in out_edges]
        in_edges  = [np.array(vs, dtype=np.int32) for vs in in_edges]
        degrees   = np.array(degrees, dtype=np.int32)

        treeletsL = [[] for _ in range(nv)]
        treeletsH = [[] for _ in range(nv)]
        treeletsR = [[] for _ in range(nv)]
        treelets_degree = [0 for _ in range(nv)]
        for (u,v,w) in treelets:
            treeletsL[u].append((v,w))
            treeletsH[v].append((u,w))
            treeletsR[w].append((u,v))
            treelets_degree[u] += 1
            treelets_degree[v] += 1
            treelets_degree[w] += 1
        treeletsL = [np.array(xs, dtype=np.int32) for xs in treeletsL]
        treeletsH = [np.array(xs, dtype=np.int32) for xs in treeletsH]
        treeletsR = [np.array(xs, dtype=np.int32) for xs in treeletsR]
        treelets_degree = np.array(treelets_degree, dtype=np.int32)

        return GraphData(
            labels=labels,
            degrees=degrees,
            in_edges=in_edges,
            out_edges=out_edges,
            treelets_degree=treelets_degree,
            treeletsL=treeletsL,
            treeletsH=treeletsH,
            treeletsR=treeletsR
        )

GraphData = collections.namedtuple(
    "GraphData",
    ["labels", "degrees", "in_edges", "out_edges", "treelets_degree", "treeletsL", "treeletsH", "treeletsR"]
)

def convert(batch, device = None):
    return [(convert_GraphData(conj, device), convert_GraphData(stmt, device), y) for (conj,stmt,y) in batch]

def convert_GraphData(gd, device):
    return GraphData(
        labels  = chainer.dataset.convert.to_device(device, gd.labels),
        degrees = chainer.dataset.convert.to_device(device, gd.degrees),
        in_edges  = [chainer.dataset.convert.to_device(device, ns) for ns in gd.in_edges],
        out_edges = [chainer.dataset.convert.to_device(device, ns) for ns in gd.out_edges],
        treelets_degree = chainer.dataset.convert.to_device(device, gd.treelets_degree),
        treeletsL = [chainer.dataset.convert.to_device(device, tl) for tl in gd.treeletsL],
        treeletsH = [chainer.dataset.convert.to_device(device, tl) for tl in gd.treeletsH],
        treeletsR = [chainer.dataset.convert.to_device(device, tl) for tl in gd.treeletsR],
    )

