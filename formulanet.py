import chainer
from chainer import dataset
from chainer import function_node
import chainer.functions as F
import chainer.links as L
from chainer import reporter
import collections
import h5py
import numpy as np
import re

import holstep
import parser_funcparselib
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

        # dirty optimization
        FI_fc1a_x = self.FI.fc1a(x)
        FI_fc1b_x = self.FI.fc1b(x)
        FO_fc1a_x = self.FO.fc1a(x)
        FO_fc1b_x = self.FO.fc1b(x)
        FI_inputs = FI_fc1a_x[gs.edges[:,0]] + FI_fc1b_x[gs.edges[:,1]]
        FO_inputs = FO_fc1a_x[gs.edges[:,0]] + FO_fc1b_x[gs.edges[:,1]]

        # 速度とバッチ正規化のサンプル数を増やすために、各頂点単位ではなくまとめて実行する
        FI_outputs = self.FI(FI_inputs)
        FO_outputs = self.FO(FO_inputs)

        d = gather_edges_to_vertex(gs, FI_outputs, FO_outputs)

        x_new += d

        if self._order_preserving:
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
            FL_inputs = FL_fc1a_x[gs.treelets[:,0]] + FL_fc1b_x[gs.treelets[:,1]] + FL_fc1c_x[gs.treelets[:,2]]
            FH_inputs = FH_fc1a_x[gs.treelets[:,0]] + FH_fc1b_x[gs.treelets[:,1]] + FH_fc1c_x[gs.treelets[:,2]]
            FR_inputs = FR_fc1a_x[gs.treelets[:,0]] + FR_fc1b_x[gs.treelets[:,1]] + FR_fc1c_x[gs.treelets[:,2]]

            # 速度とバッチ正規化のサンプル数を増やすために、各頂点単位ではなくまとめて実行する
            FL_outputs = self.FL(FL_inputs)
            FH_outputs = self.FH(FH_inputs)
            FR_outputs = self.FR(FR_inputs)

            d = gather_treelets_to_vertex(gs, FL_outputs, FH_outputs, FR_outputs)

            x_new += d

        return self.FP(x_new)

    def _compute_graph_embedding(self, gs, x, stmt):
        (beg,end) = gs.node_ranges[stmt]
        return F.max(x[beg:end], axis=0, keepdims=True)


class GatherEdgesToVertex(function_node.FunctionNode):
    def __init__(self, gs):
        self.gs = gs

    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(
            in_types.size() == 2,
            in_types[0].dtype.kind == 'f',
            in_types[1].dtype.kind == 'f',
            in_types[0].shape == (len(self.gs.edges), DIM),
            in_types[1].shape == (len(self.gs.edges), DIM),
        )

    def forward(self, inputs):
        xp = chainer.cuda.get_array_module(*inputs)
        FI_outputs, FO_outputs = inputs
        ret = xp.zeros((len(self.gs.labels), DIM), np.float32)
        for v in range(len(self.gs.labels)):
            den = len(self.gs.in_edges[v]) + len(self.gs.out_edges[v])
            xp.sum(FI_outputs[self.gs.in_edges[v],  :], axis=0, out=ret[v,:])
            xp.sum(FO_outputs[self.gs.out_edges[v], :], axis=0, out=ret[v,:])
            xp.divide(ret[v,:], den, out=ret[v,:])
        return ret,

    # XXX: This is not differentiable
    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        gy = gy.data
        xp = chainer.cuda.get_array_module(gy)
        gFI = xp.zeros((len(self.gs.edges), DIM), np.float32)
        gFO = xp.zeros((len(self.gs.edges), DIM), np.float32)
        for v in range(len(self.gs.labels)):
            den = len(self.gs.in_edges[v]) + len(self.gs.out_edges[v])
            g = gy[v] / den
            if xp is np:
                np.add.at(gFI, self.gs.in_edges[v], g)
                np.add.at(gFO, self.gs.out_edges[v], g)
            else:
                gFI.scatter_add(self.gs.in_edges[v], g)
                gFO.scatter_add(self.gs.out_edges[v], g)
        return chainer.Variable(gFI), chainer.Variable(gFO)

def gather_edges_to_vertex(gs, FI_outputs, FO_outputs):
    if True:
        y, = GatherEdgesToVertex(gs).apply((FI_outputs, FO_outputs))
        return y
    else:
        d = []
        for v in range(len(self.gs.labels)):
            h = F.sum(FI_outputs[self.gs.in_edges[v], :], axis=0) + \
                F.sum(FO_outputs[self.gs.out_edges[v],:], axis=0)
            h /= (len(self.gs.in_edges[v]) + len(self.gs.out_edges[v]))
            d.append(h)
        return F.vstack(d)


class GatherTreeletsToVertex(function_node.FunctionNode):
    def __init__(self, gs):
        self.gs = gs

    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(
            in_types.size() == 3,
            in_types[0].dtype.kind == 'f',
            in_types[1].dtype.kind == 'f',
            in_types[2].dtype.kind == 'f',
            in_types[0].shape == (len(self.gs.treelets), DIM),
            in_types[1].shape == (len(self.gs.treelets), DIM),
            in_types[2].shape == (len(self.gs.treelets), DIM),
        )

    def forward(self, inputs):
        xp = chainer.cuda.get_array_module(*inputs)
        FL_outputs, FH_outputs, FR_outputs = inputs
        ret = xp.zeros((len(self.gs.labels), DIM), np.float32)
        for v in range(len(self.gs.labels)):
            den = len(self.gs.treeletsL[v]) + len(self.gs.treeletsH[v]) + len(self.gs.treeletsR[v])
            if den != 0:
                xp.sum(FL_outputs[self.gs.treeletsL[v], :], axis=0, out=ret[v,:])
                xp.sum(FH_outputs[self.gs.treeletsH[v], :], axis=0, out=ret[v,:])
                xp.sum(FR_outputs[self.gs.treeletsR[v], :], axis=0, out=ret[v,:])
                xp.divide(ret[v,:], den, out=ret[v,:])
        return ret,

    # XXX: This is not differentiable
    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        gy = gy.data
        xp = chainer.cuda.get_array_module(gy)
        gFL = xp.zeros((len(self.gs.treelets), DIM), np.float32)
        gFH = xp.zeros((len(self.gs.treelets), DIM), np.float32)
        gFR = xp.zeros((len(self.gs.treelets), DIM), np.float32)
        for v in range(len(self.gs.labels)):
            d = len(self.gs.in_edges[v]) + len(self.gs.out_edges[v])
            g = gy[v] / d
            if xp is np:
                np.add.at(gFL, self.gs.treeletsL[v], g)
                np.add.at(gFH, self.gs.treeletsH[v], g)
                np.add.at(gFR, self.gs.treeletsR[v], g)
            else:
                gFL.scatter_add(self.gs.treeletsL[v], g)
                gFH.scatter_add(self.gs.treeletsH[v], g)
                gFR.scatter_add(self.gs.treeletsR[v], g)
        return chainer.Variable(gFL), chainer.Variable(gFH), chainer.Variable(gFR)

def gather_treelets_to_vertex(gs, FL_outputs, FH_outputs, FR_outputs):
    if True:
        y, = GatherTreeletsToVertex(gs).apply((FL_outputs, FH_outputs, FR_outputs))
        return y
    else:
        with chainer.cuda.get_device_from_array(FL_outputs.data):
            zeros_DIM = self.xp.zeros(DIM, dtype=np.float32)
        d = []
        for v in range(len(gs.labels)):
            den = len(gs.treeletsL[v]) + len(gs.treeletsH[v]) + len(gs.treeletsR[v])
            if den == 0:
                d.append(zeros_DIM)
            else:
                h = F.sum(FL_outputs[gs.treeletsL[v], :], axis=0) + \
                    F.sum(FH_outputs[gs.treeletsH[v], :], axis=0) + \
                    F.sum(FR_outputs[gs.treeletsR[v], :], axis=0)
                d.append(h / den)
        return F.vstack(d)


class Dataset(dataset.DatasetMixin):
    def __init__(self, names, h5f):
        super().__init__()
        self._name_to_id = {name: i for (i, name) in enumerate(names)}
        self._h5f = h5f
        self._dt_vstr = h5py.special_dtype(vlen=str)
        self._dt_vint = h5py.special_dtype(vlen=np.int32)

    def init_db(self):
        self._h5f.create_dataset("examples_conjecture", (0,), maxshape=(None,), dtype=self._dt_vstr, compression="gzip")
        self._h5f.create_dataset("examples_statement", (0,), maxshape=(None,), dtype=np.int32, compression="gzip")

    def add_file(self, name, fname):
        df = holstep.read_file(fname)
        grp = self._h5f.create_group(name)

        grp_conjecture = grp.create_group("conjecture")
        self._set_graph(grp_conjecture, self._build_graph(df.conjecture.text))

        grp.create_dataset("labels", data=np.array(df.labels, dtype=np.bool), compression="gzip")

        grp_statements = grp.create_group("statements")
        for (i,s) in enumerate(df.examples):
            grp_statement = grp_statements.create_group("%05d" % i)
            self._set_graph(grp_statement, self._build_graph(s.text))

        n = len(self._h5f["examples_conjecture"])
        self._h5f["examples_conjecture"].resize((n + len(df.examples),))
        for i in range(n,n+len(df.examples)):
            self._h5f["examples_conjecture"][i] = name
        self._h5f["examples_statement" ].resize((n + len(df.examples),))
        self._h5f["examples_statement" ][n:] = np.arange(len(df.examples), dtype=np.int32)

    def _set_graph(self, grp, g):
        grp.create_dataset("labels", data=g.labels, compression="gzip")
        grp.create_dataset("edges", data=g.edges, compression="gzip")
        grp.create_dataset("in_edges", data=g.in_edges, dtype=self._dt_vint, compression="gzip")
        grp.create_dataset("out_edges", data=g.out_edges, dtype=self._dt_vint, compression="gzip")
        grp.create_dataset("treelets", data=g.treelets, compression="gzip")
        grp.create_dataset("treeletsL", data=g.treeletsL, dtype=self._dt_vint, compression="gzip")
        grp.create_dataset("treeletsH", data=g.treeletsH, dtype=self._dt_vint, compression="gzip")
        grp.create_dataset("treeletsR", data=g.treeletsR, dtype=self._dt_vint, compression="gzip")

    def _get_graph(self, grp):
        return GraphData(
            labels=grp["labels"],
            edges=grp["edges"],
            in_edges=grp["in_edges"],
            out_edges=grp["out_edges"],
            treelets=grp["treelets"],
            treeletsL=grp["treeletsL"],
            treeletsH=grp["treeletsH"],
            treeletsR=grp["treeletsR"],
        )

    def __len__(self):
        return len(self._h5f["examples_conjecture"])

    def get_example(self, i):
        name = self._h5f["examples_conjecture"][i]
        j = self._h5f["examples_statement"][i]
        grp = self._h5f[name]
        g_conj = self._get_graph(grp["conjecture"])
        g_stmt = self._get_graph(grp["statements"]["%05d" % j])
        label  = grp["labels"][j]
        return (g_conj, g_stmt, label)

    def _symbol_to_id(self, sym):
        if re.fullmatch(r'_\d+', sym):
            sym = "_"
        elif re.fullmatch(r'GEN%PVAR%\d+', sym):
            sym = "GEN%PVAR"
        if sym not in self._name_to_id:
            sym = "UNKNOWN"
        return self._name_to_id[sym]

    def _build_graph(self, text):
        tokens = parser_funcparselib.tokenize(text)
        g = tree.tree_to_graph(tree.thm_to_tree(parser_funcparselib.thm.parse(tokens)))
        labels, edges, treelets = g
        nv = len(labels)

        out_edges = [[] for _ in range(nv)]
        in_edges  = [[] for _ in range(nv)]
        for (i,(u,v)) in enumerate(edges):
            out_edges[u].append(i)
            in_edges[v].append(i)
        out_edges = [np.array(es, dtype=np.int32) for es in out_edges]
        in_edges  = [np.array(es, dtype=np.int32) for es in in_edges]

        treeletsL = [[] for _ in range(nv)]
        treeletsH = [[] for _ in range(nv)]
        treeletsR = [[] for _ in range(nv)]
        for (i,(u,v,w)) in enumerate(treelets):
            treeletsL[u].append(i)
            treeletsH[v].append(i)
            treeletsR[w].append(i)
        treeletsL = [np.array(ts, dtype=np.int32) for ts in treeletsL]
        treeletsH = [np.array(ts, dtype=np.int32) for ts in treeletsH]
        treeletsR = [np.array(ts, dtype=np.int32) for ts in treeletsR]

        return GraphData(
            labels=np.array([self._symbol_to_id(l) for l in labels], dtype=np.int32),
            edges=np.array(edges, dtype=np.int32),
            in_edges=in_edges,
            out_edges=out_edges,
            treelets=np.array(treelets, dtype=np.int32),
            treeletsL=treeletsL,
            treeletsH=treeletsH,
            treeletsR=treeletsR
        )

GraphData = collections.namedtuple(
    "GraphData",
    ["labels", "edges", "in_edges", "out_edges", "treelets", "treeletsL", "treeletsH", "treeletsR"]
)


GraphsData = collections.namedtuple(
    "GraphsData",
    ["node_ranges", "labels", "edges", "in_edges", "out_edges", "treelets", "treeletsL", "treeletsH", "treeletsR"]
)

def convert(minibatch, device = None):
    node_offset = 0
    node_ranges = []
    edge_offset = 0
    treelet_offset = 0

    table = {}

    labels = []
    edges = []
    in_edges = []
    out_edges = []
    treelets = []
    treeletsL = []
    treeletsH = []
    treeletsR = []

    def f(gd):
        nonlocal node_offset
        nonlocal edge_offset
        nonlocal treelet_offset
        if id(gd) in table:
            return table[id(gd)]

        labels.append(gd.labels)

        edges.append(np.array(gd.edges) + node_offset)
        for es in gd.in_edges:
            in_edges.append(chainer.dataset.convert.to_device(device, np.array(es) + edge_offset))
        for es in gd.out_edges:
            out_edges.append(chainer.dataset.convert.to_device(device, np.array(es) + edge_offset))

        treelets.append(np.array(gd.treelets) + node_offset)
        for tl in gd.treeletsL:
            treeletsL.append(chainer.dataset.convert.to_device(device, np.array(tl) + treelet_offset))
        for tl in gd.treeletsH:
            treeletsH.append(chainer.dataset.convert.to_device(device, np.array(tl) + treelet_offset))
        for tl in gd.treeletsR:
            treeletsR.append(chainer.dataset.convert.to_device(device, np.array(tl) + treelet_offset))

        ret = len(node_ranges)
        node_ranges.append( (node_offset, node_offset+len(gd.labels)) )

        node_offset += len(gd.labels)
        edge_offset += len(gd.edges)
        treelet_offset += len(gd.treelets)

        table[id(gd)] = ret

        return ret

    minibatch = [(f(conj), f(stmt), y) for (conj,stmt,y) in minibatch]

    gs = GraphsData(
        node_ranges = node_ranges,
        labels      = chainer.dataset.convert.to_device(device, np.concatenate(labels)),
        edges       = chainer.dataset.convert.to_device(device, np.concatenate(edges)),
        in_edges    = in_edges,
        out_edges   = out_edges,
        treelets    = chainer.dataset.convert.to_device(device, np.concatenate(treelets)),
        treeletsL   = treeletsL,
        treeletsH   = treeletsH,
        treeletsR   = treeletsR,
    )

    return (gs, minibatch)
