import chainer
from chainer import dataset, Variable
import chainer.functions as F
import chainer.links as L
from chainer import reporter
import h5py
import numpy as np
from pathlib import Path
import re
import sparse_matmul
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import holstep
import parser_funcparselib
import tree


try:
    import cupy
    Array = Union[np.ndarray, cupy.ndarray]
except ImportError:
    Array = np.ndarray
VariableOrArray = Union[Variable, Array]


DIM = 256


GraphData = NamedTuple(
    "GraphData",
    [("labels", Array),
     ("edges", Array),
     ("in_edges", List[Array]),
     ("out_edges", List[Array]),
     ("treelets", Array),
     ("treeletsL", List[Array]),
     ("treeletsH", List[Array]),
     ("treeletsR", List[Array]),
    ]
)


GraphsData = NamedTuple(
    "GraphsData",
    [("node_ranges", Array),
     ("labels", Array),
     ("edges", Array),
     ("treelets", Array),
     ("MI", sparse_matmul.sparse_coo_matrix),
     ("MO", sparse_matmul.sparse_coo_matrix),
     ("ML", sparse_matmul.sparse_coo_matrix),
     ("MH", sparse_matmul.sparse_coo_matrix),
     ("MR", sparse_matmul.sparse_coo_matrix),
    ]
)


class FP(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.fc = L.Linear(DIM, DIM)
            self.bn = L.BatchNormalization(DIM)

    def __call__(self, x: VariableOrArray) -> Variable:
        return F.relu(self.bn(self.fc(x)))

class Block(chainer.Chain):
    def __init__(self, n_input: int) -> None:
        super().__init__()
        self._n_input = n_input
        with self.init_scope():
            self.fc1 = L.Linear(DIM*n_input, DIM)
            self.fc2 = L.Linear(DIM, DIM)
            self.bn1 = L.BatchNormalization(DIM)
            self.bn2 = L.BatchNormalization(DIM)

    def __call__(self, *args: VariableOrArray):
        assert len(args) == self._n_input
        h = F.relu(self.bn1(self.fc1(F.concat(args))))
        h = F.relu(self.bn2(self.fc2(h)))
        return h

# dirty optimization
class Block2(chainer.Chain):
    def __init__(self) -> None:
        super().__init__()
        with self.init_scope():
            self.fc1a = L.Linear(DIM, DIM, nobias=True)
            self.fc1b = L.Linear(DIM, DIM, nobias=False)
            self.fc2 = L.Linear(DIM, DIM)
            self.bn1 = L.BatchNormalization(DIM)
            self.bn2 = L.BatchNormalization(DIM)

    def __call__(self, arg: VariableOrArray) -> Variable:
        h = F.relu(self.bn1(arg))
        h = F.relu(self.bn2(self.fc2(h)))
        return h

# dirty optimization
class Block3(chainer.Chain):
    def __init__(self) -> None:
        super().__init__()
        with self.init_scope():
            self.fc1a = L.Linear(DIM, DIM, nobias=True)
            self.fc1b = L.Linear(DIM, DIM, nobias=True)
            self.fc1c = L.Linear(DIM, DIM, nobias=False)
            self.fc2 = L.Linear(DIM, DIM)
            self.bn1 = L.BatchNormalization(DIM)
            self.bn2 = L.BatchNormalization(DIM)

    def __call__(self, arg: VariableOrArray) -> Variable:
        h = F.relu(self.bn1(arg))
        h = F.relu(self.bn2(self.fc2(h)))
        return h

class Step(chainer.Chain):
    def __init__(self, order_preserving: bool) -> None:
        super().__init__()
        self._order_preserving = order_preserving
        with self.init_scope():
            self.FP = FP()
            self.FI = Block2()
            self.FO = Block2()
            if order_preserving:
                self.FH = Block3()
                self.FL = Block3()
                self.FR = Block3()

    def __call__(self, gs: GraphsData, x: VariableOrArray) -> Variable:
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

        d = sparse_matmul.sparse_matmul(gs.MI, FI_outputs) + \
            sparse_matmul.sparse_matmul(gs.MO, FO_outputs)

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

            d = sparse_matmul.sparse_matmul(gs.ML, FL_outputs) + \
                sparse_matmul.sparse_matmul(gs.MH, FH_outputs) + \
                sparse_matmul.sparse_matmul(gs.MR, FR_outputs)

            x_new += d

        return self.FP(x_new)

class Classifier(chainer.Chain):
    def __init__(self, conditional: bool = True) -> None:
        super().__init__()
        self._conditional = conditional
        with self.init_scope():
            self.fc1 = L.Linear(2*DIM if conditional else DIM, DIM)
            self.bn  = L.BatchNormalization(DIM)
            self.fc2 = L.Linear(DIM, 2)

    def __call__(self, *args: VariableOrArray) -> Variable:
        if self._conditional:
            assert len(args) == 2
        else:
            assert len(args) == 1
        return self.fc2(F.relu(self.bn(self.fc1(F.concat(args)))))

class FormulaNet(chainer.Chain):
    def __init__(self, vocab_size: int, steps: int, order_preserving: bool, conditional: bool) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._steps = steps
        self._order_preserving = order_preserving
        self._conditional = conditional

        with self.init_scope():
            self.embed_id = L.EmbedID(vocab_size, DIM)
            self.steps = chainer.ChainList(*[Step(order_preserving) for _ in range(steps)])
            self.classifier = Classifier(conditional)

    def __call__(self, gs: GraphsData, minibatch: VariableOrArray) -> Variable:
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

    def _forward(self, gs: GraphsData, minibatch: VariableOrArray) -> Tuple[Array, Array]:
        stmt_embeddings = []
        conj_embeddings = []
        labels = []

        def collect_embedding() -> None:
            es = [self._compute_graph_embedding(gs, x, i) for i in range(len(gs.node_ranges))]
            for (conj, stmt, y) in minibatch:
                stmt_embeddings.append(es[stmt])
                if self._conditional:
                    conj_embeddings.append(es[conj])
                labels.append(1 if y else 0)

        x = self._initial_nodes_embedding(gs)
        collect_embedding()
        for (i,step) in enumerate(self.steps):
            # print("step " + str(i))
            x = step(gs, x)
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

    def predict(self, gs: GraphsData, conj: int, stmt: int) -> bool:
        return (F.argmax(self.logit(gs, conj, stmt)) > 0)

    def logit(self, gs: GraphsData, conj: int, stmt: int) -> Variable:
        x = self._initial_nodes_embedding(gs)
        for (i, step) in enumerate(self.steps):
            x = step(gs, x)

        stmt_embedding = self._compute_graph_embedding(gs, x, stmt)
        if self._conditional:
            conj_embedding = self._compute_graph_embedding(gs, x, conj)
            return self.classifier(conj_embedding, stmt_embedding)[0]
        else:
            return self.classifier(stmt_embedding)[0]

    def _initial_nodes_embedding(self, gs: GraphsData) -> Variable:
        return self.embed_id(gs.labels)

    def _compute_graph_embedding(self, gs: GraphsData, x: Array, stmt: int) -> Variable:
        (beg,end) = gs.node_ranges[stmt]
        return F.max(x[beg:end], axis=0, keepdims=True)


class Dataset(dataset.DatasetMixin):
    def __init__(self, names: List[str], h5f) -> None:
        super().__init__()
        self._name_to_id = {name: i for (i, name) in enumerate(names)}
        self._h5f = h5f
        self._len = int(len(self._h5f["examples_conjecture"]))

    def init_db(self) -> None:
        self._h5f.create_dataset("examples_conjecture", (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str), compression="gzip")
        self._h5f.create_dataset("examples_statement", (0,), maxshape=(None,), dtype=np.int32, compression="gzip")

    def add_file(self, name: str, fname: Union[Path, str]) -> None:
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
        self._len += len(df.examples)

    def _set_graph(self, grp: h5py.Group, g: GraphData) -> None:
        dt_vint = h5py.special_dtype(vlen=np.int32)
        grp.create_dataset("labels", data=g.labels, compression="gzip")
        grp.create_dataset("edges", data=g.edges, compression="gzip")
        grp.create_dataset("in_edges", data=g.in_edges, dtype=dt_vint, compression="gzip")
        grp.create_dataset("out_edges", data=g.out_edges, dtype=dt_vint, compression="gzip")
        grp.create_dataset("treelets", data=g.treelets, compression="gzip")
        grp.create_dataset("treeletsL", data=g.treeletsL, dtype=dt_vint, compression="gzip")
        grp.create_dataset("treeletsH", data=g.treeletsH, dtype=dt_vint, compression="gzip")
        grp.create_dataset("treeletsR", data=g.treeletsR, dtype=dt_vint, compression="gzip")

    def _get_graph(self, grp: h5py.Group) -> GraphData:
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

    def __len__(self) -> int:
        return self._len

    def get_example(self, i: int) -> Tuple[GraphData, GraphData, bool]:
        name = self._h5f["examples_conjecture"][i]
        j = self._h5f["examples_statement"][i]
        grp = self._h5f[name]
        g_conj = self._get_graph(grp["conjecture"])
        g_stmt = self._get_graph(grp["statements"]["%05d" % j])
        label  = grp["labels"][j]
        return (g_conj, g_stmt, label)

    def _symbol_to_id(self, sym: str) -> int:
        if re.fullmatch(r'_\d+', sym):
            sym = "_"
        elif re.fullmatch(r'GEN%PVAR%\d+', sym):
            sym = "GEN%PVAR"
        if sym not in self._name_to_id:
            sym = "UNKNOWN"
        return self._name_to_id[sym]

    def _build_graph(self, text: str) -> GraphData:
        tokens = parser_funcparselib.tokenize(text)
        g = tree.tree_to_graph(tree.thm_to_tree(parser_funcparselib.thm.parse(tokens)))
        labels, edges, treelets = g
        nv = len(labels)

        out_edges_l = [[] for _ in range(nv)] # type: List[List[int]]
        in_edges_l  = [[] for _ in range(nv)] # type: List[List[int]]
        for (i,(u,v)) in enumerate(edges):
            out_edges_l[u].append(i)
            in_edges_l[v].append(i)
        out_edges = [np.array(es, dtype=np.int32) for es in out_edges_l]
        in_edges  = [np.array(es, dtype=np.int32) for es in in_edges_l]

        treeletsL_l = [[] for _ in range(nv)] # type: List[List[int]]
        treeletsH_l = [[] for _ in range(nv)] # type: List[List[int]]
        treeletsR_l = [[] for _ in range(nv)] # type: List[List[int]]
        for (i,(u,v,w)) in enumerate(treelets):
            treeletsL_l[u].append(i)
            treeletsH_l[v].append(i)
            treeletsR_l[w].append(i)
        treeletsL = [np.array(ts, dtype=np.int32) for ts in treeletsL_l]
        treeletsH = [np.array(ts, dtype=np.int32) for ts in treeletsH_l]
        treeletsR = [np.array(ts, dtype=np.int32) for ts in treeletsR_l]

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


def convert(minibatch: List[Tuple[GraphData, GraphData, bool]], device: Optional[int] = None) -> Tuple[GraphsData, List[Tuple[int,int,bool]]]:
    node_offset = 0
    node_ranges = [] # type: List[Tuple[int,int]]
    edge_offset = 0
    treelet_offset = 0

    table = {} # type: Dict[int,int]

    labels = []
    edges = []
    treelets = []

    MI_data = []
    MI_row = []
    MI_col = []
    MO_data = []
    MO_row = []
    MO_col = []
    ML_data = []
    ML_row = []
    ML_col = []
    MH_data = []
    MH_row = []
    MH_col = []
    MR_data = []
    MR_row = []
    MR_col = []

    def f(gd: GraphData) -> int:
        nonlocal node_offset
        nonlocal edge_offset
        nonlocal treelet_offset
        if id(gd) in table:
            return table[id(gd)]

        labels.append(gd.labels)

        edges.append(np.array(gd.edges) + node_offset)
        for v in range(len(gd.labels)):
            den = (len(gd.in_edges[v]) + len(gd.out_edges[v]))
            for e in gd.in_edges[v]:
                MI_data.append(1.0 / den)
                MI_row.append(node_offset + v)
                MI_col.append(edge_offset + e)
            for e in gd.out_edges[v]:
                MO_data.append(1.0 / den)
                MO_row.append(node_offset + v)
                MO_col.append(edge_offset + e)

        treelets.append(np.array(gd.treelets) + node_offset)
        for v in range(len(gd.labels)):
            den = len(gd.treeletsL[v]) + len(gd.treeletsH[v]) + len(gd.treeletsR[v])
            if den == 0:
                continue
            for t in gd.treeletsL[v]:
                ML_data.append(1.0 / den)
                ML_row.append(node_offset + v)
                ML_col.append(treelet_offset + t)
            for t in gd.treeletsH[v]:
                MH_data.append(1.0 / den)
                MH_row.append(node_offset + v)
                MH_col.append(treelet_offset + t)
            for t in gd.treeletsR[v]:
                MR_data.append(1.0 / den)
                MR_row.append(node_offset + v)
                MR_col.append(treelet_offset + t)

        ret = len(node_ranges)
        node_ranges.append( (node_offset, node_offset+len(gd.labels)) )

        node_offset += len(gd.labels)
        edge_offset += len(gd.edges)
        treelet_offset += len(gd.treelets)

        table[id(gd)] = ret

        return ret

    minibatch2 = [(f(conj), f(stmt), y) for (conj,stmt,y) in minibatch]

    def arr_f(x: List[float]) -> Array:
        return chainer.dataset.convert.to_device(device, np.array(x, dtype=np.float32))
    def arr_i(x: List[int]) -> Array:
        return chainer.dataset.convert.to_device(device, np.array(x, dtype=np.int32))

    MI = sparse_matmul.sparse_coo_matrix(
        arr_f(MI_data), arr_i(MI_row), arr_i(MI_col),
        shape=(node_offset, edge_offset))
    MO = sparse_matmul.sparse_coo_matrix(
        arr_f(MO_data), arr_i(MO_row), arr_i(MO_col),
        shape=(node_offset, edge_offset))

    ML = sparse_matmul.sparse_coo_matrix(
        arr_f(ML_data), arr_i(ML_row), arr_i(ML_col),
        shape=(node_offset, treelet_offset))
    MH = sparse_matmul.sparse_coo_matrix(
        arr_f(MH_data), arr_i(MH_row), arr_i(MH_col),
        shape=(node_offset, treelet_offset))
    MR = sparse_matmul.sparse_coo_matrix(
        arr_f(MR_data), arr_i(MR_row), arr_i(MR_col),
        shape=(node_offset, treelet_offset))

    gs = GraphsData(
        node_ranges = node_ranges,
        labels   = chainer.dataset.convert.to_device(device, np.concatenate(labels)),
        edges    = chainer.dataset.convert.to_device(device, np.concatenate(edges)),
        treelets = chainer.dataset.convert.to_device(device, np.concatenate(treelets)),
        MI = MI,
        MO = MO,
        ML = ML,
        MH = MH,
        MR = MR,
    )

    return (gs, minibatch2)
