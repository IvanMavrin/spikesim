import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import trange
from dataclasses import dataclass
import csv

class SpikeNetworkSim:
    def __init__(self, inputs_l, dt = 1):
            
        self.values = pd.DataFrame(columns=np.arange(inputs_l))
        
        input_nodes = [
            {
                "type": "input",
                "priority": 0,
                "listening": None,
                "broadcasting": None,
                "layer": -1
            } for _ in range(inputs_l)]
        
        self.nodes = pd.DataFrame(data=input_nodes, columns=["type", "priority", "listening", "broadcasting", "layer"])
        self.status = pd.DataFrame(columns=["weights", "inhibited"])
        self.dt = dt
        
        self.layer_params = {
            "tau_refractory": [],
            "tau_inhibitory": [],
            "tau_ltp": [],
            "tau_leak": [],
            "thres": [],
            "ainc": [],
            "adec": [],
            "wmin": [],
            "wmax": [],
            "learning": [],
            "wta": [],
            "layer_type": []
        }
        
        self.defaults = {
            "tau_inhibitory": 3, 
            "tau_refractory": 5, 
            "tau_leak": 10, 
            "tau_ltp": 5, 
            "thres": 200,
            "ainc": 30, 
            "adec": -15, 
            "wmax": 255,
            "wmin": 1,
            "learning": True,
            "wta": False,
            "layer_type": "default"
        }
        
        self.processors = {
            "default": self.process,
            "ttron": self.ttron_process
        }
        
    def new_layer(self, num_nodes, weights=None, passed_inputs=None, **layer_params):
        
        for param in layer_params.keys():
            if param in layer_params:
                self.layer_params[param].append(layer_params[param])
            else:
                raise Exception(f"Parameter {param} doesn't exist")
        for dparam in self.defaults.keys():
            if dparam not in layer_params:
                self.layer_params[dparam].append(self.defaults[dparam])
        
        layer = self.nodes["layer"].max()+1
        
        net = self.nodes
        
        if layer == 0: #первый слой
            input_nodes = self.nodes.loc[self.nodes['type'] == 'input', ['type', 'listening', 'broadcasting', 'priority']]
            net = None
        else:
            output_nodes = self.nodes.loc[net['type'] == 'postsynaptic'].index.tolist()
            num_inputs = len(output_nodes)
            input_nodes = pd.DataFrame({
                'type': 'buffer',
                'listening': [[output_nodes[i]] for i in range(num_inputs)],
                'broadcasting': [[] for _ in range(num_inputs)],
                'priority': 0
            }, index=np.arange(net.index.max()+1, net.index.max()+1+num_inputs))
        
        if passed_inputs is not None:
            input_nodes = pd.concat(input_nodes, passed_inputs)
        num_inputs = input_nodes.index.size

        # Создаем DataFrame для узлов ltp
        ltp_nodes = pd.DataFrame({
            'type': 'ltp',
            'listening': [[] for _ in range(num_inputs)],
            'broadcasting': [[] for _ in range(num_inputs)],
            'priority': 1
        }, index=np.arange(input_nodes.index.max()+1, input_nodes.index.max()+1+num_inputs))

        # Соединяем узлы input с узлами ltp и presynaptic
        for i in range(num_inputs):
            input_nodes.iat[i, 2] = [ltp_nodes.index[i]]
            ltp_nodes.iat[i, 1] = [input_nodes.index[i]]

        df = pd.concat([input_nodes, ltp_nodes])

        for i in range(num_nodes):
            presynaptic_node_index = df.index.max() + 1
            postsynaptic_node_index = presynaptic_node_index + 1
            potentiating_node_index = postsynaptic_node_index + 1

            nodes_data = {
                'presynaptic': {
                    'listening': [input_nodes.index.tolist()],
                    'broadcasting': [postsynaptic_node_index], 
                    'index': presynaptic_node_index,
                    'priority': 2
                },
                'postsynaptic': {
                    'listening': [presynaptic_node_index],
                    'broadcasting': [], 
                    'index': postsynaptic_node_index,
                    'priority': 3
                },
                'potentiating': {
                    'listening': [ltp_nodes.index.tolist()],
                    'broadcasting': [presynaptic_node_index], 
                    'index': potentiating_node_index,
                    'priority': 4
                }
            }

            # Создаем DataFrame для всех узлов
            nodes = pd.DataFrame([
                {
                    'type': k,
                    **v
                } for k, v in nodes_data.items()
            ])

            nodes = nodes.set_index('index')

            # Конкатенируем все DataFrame в один
            df = pd.concat([df, nodes])

        for i in range(num_inputs):
            df.iat[num_inputs+i, 2] = df.loc[df['type'] == 'potentiating'].index.tolist()
        
        presynaptic_range = df[df.type == 'presynaptic'].index.tolist()
        for i in df[df.type == 'postsynaptic'].index:
            df.iat[i-df.index.min(), 2] = [p for p in presynaptic_range if p != i-(postsynaptic_node_index-presynaptic_node_index)]
        if weights is None:
            weights = {n: np.random.randint(self.layer_params["wmin"][-1], self.layer_params["wmax"][-1], num_inputs) for n in df.loc[df['type'] == 'presynaptic'].index.tolist()}
        
        if isinstance(weights, np.ndarray):
            weights = {n: w for n, w in zip(df.loc[df['type'] == 'presynaptic'].index.tolist(), weights)}
        df['layer'] = layer
        self.nodes = pd.concat((net, df))
        self.status = pd.concat((self.status, pd.DataFrame([
            {
                "index": n,
                "weights": w,
                "inhibited": -1
            } for n, w in weights.items()
        ]).set_index("index")))
    
    def ttron_layer(self, num_nodes, num_cat_inputs, weights=None, passed_inputs=None, delay_depth=0, genome=None, **layer_params):
        
        for param in layer_params.keys():
            if param in layer_params:
                self.layer_params[param].append(layer_params[param])
            else:
                raise Exception(f"Parameter {param} doesn't exist")
        for dparam in self.defaults.keys():
            if dparam not in layer_params:
                self.layer_params[dparam].append(self.defaults[dparam])
        
        layer = self.nodes.layer.max()+1
        
        max_priority = self.nodes.priority.max()
        
        if layer == 0: #первый слой
            input_nodes = self.nodes.loc[self.nodes['type'] == 'input', ['type', 'listening', 'broadcasting', 'priority']]
            net = None
        else:
            output_nodes = self.nodes.loc[net['type'] == 'postsynaptic'].index.tolist()
            num_inputs = len(output_nodes)
            input_nodes = pd.DataFrame({
                'type': 'buffer',
                'listening': [[output_nodes[i]] for i in range(num_inputs)],
                'broadcasting': [[] for _ in range(num_inputs)],
                'priority': 0
            }, index=np.arange(net.index.max()+1, net.index.max()+1+num_inputs))
            net = self.nodes
    
        
        if passed_inputs is not None:
            input_nodes = pd.concat(input_nodes, passed_inputs)
        num_inputs = input_nodes.index.size
            
                
        delay_nodes = input_nodes.copy()
        for i in range(delay_depth):
            delay_nodes = pd.concat((delay_nodes, pd.DataFrame({
                'type': 'buffer',
                'listening': [[delay_nodes.index[-num_inputs+n]] for n in range(num_inputs)],
                'broadcasting': [[] for _ in range(num_inputs)],
                'priority': max_priority+delay_depth-i
            }, index=np.arange(input_nodes.index.max()+1+num_inputs*i, input_nodes.index.max()+1+num_inputs*(i+1)))))
                                    
        presynaptic_input = pd.concat((input_nodes, delay_nodes.loc[delay_nodes.priority == 1]))
        presynaptic_ltp = []
        input_nodes = delay_nodes
        num_inputs = input_nodes.index.size

        # Создаем DataFrame для узлов ltp
        ltp_nodes = pd.DataFrame({
            'type': 'ltp',
            'listening': [[] for _ in range(num_inputs)],
            'broadcasting': [[] for _ in range(num_inputs)],
            'priority': 1
        }, index=np.arange(input_nodes.index.max()+1, input_nodes.index.max()+1+num_inputs))

        # Соединяем узлы input с узлами ltp и presynaptic
        for i in range(num_inputs):
            input_nodes.iat[i, 2] = [ltp_nodes.index[i]]
            ltp_nodes.iat[i, 1] = [input_nodes.index[i]]
            if input_nodes.index[i] in presynaptic_input.index:
                presynaptic_ltp.append(ltp_nodes.index[i])
            
        df = pd.concat([input_nodes, ltp_nodes])
        # добавляем категориальные входы
        cat_inputs = pd.DataFrame({
            'type': 'input',
            'listening': [[] for _ in range(num_cat_inputs)],
            'broadcasting': [[] for _ in range(num_cat_inputs)],
            'priority': 0
        }, index=np.arange(df.index.max()+1, df.index.max()+1+num_cat_inputs))
        
        df = pd.concat([df, cat_inputs])
        for i in range(num_nodes):
            
            teacher_node_index           = df.index.max() + 1
            presynaptic_node_index       = df.index.max() + 2
            max_tracker_node_index       = df.index.max() + 3
            max_tracker_timer_node_index = df.index.max() + 4
            postsynaptic_node_index      = df.index.max() + 5
            spike_timer_node_index       = df.index.max() + 6
            potentiating_node_index      = df.index.max() + 7

            nodes_data = [
                {
                    'type': 'teacher',
                    'listening': [genome[i]],
                    'broadcasting': [potentiating_node_index], 
                    'index': teacher_node_index,
                    'priority': 1
                },{
                    'type': 'presynaptic',
                    'listening': presynaptic_input.index.tolist(),
                    'broadcasting': [postsynaptic_node_index], 
                    'index': presynaptic_node_index,
                    'priority': 2
                },{
                    'type': 'max_tracker',
                    'listening': [presynaptic_node_index],
                    'broadcasting': [max_tracker_timer_node_index], 
                    'index': max_tracker_node_index,
                    'priority': 4
                },{
                    'type': 'ltp',
                    'listening': [max_tracker_node_index],
                    'broadcasting': [potentiating_node_index], 
                    'index': max_tracker_timer_node_index,
                    'priority': 3
                },{
                    'type': 'postsynaptic',
                    'listening': [presynaptic_node_index],
                    'broadcasting': [], 
                    'index': postsynaptic_node_index,
                    'priority': 3
                },{
                    'type': 'ltp',
                    'listening': [postsynaptic_node_index],
                    'broadcasting': [potentiating_node_index], 
                    'index': spike_timer_node_index,
                    'priority': 4
                },{
                    'type': 'potentiating',
                    'listening': np.concatenate(([teacher_node_index, spike_timer_node_index, max_tracker_timer_node_index, postsynaptic_node_index], presynaptic_ltp, )),
                    'broadcasting': [presynaptic_node_index], 
                    'index': potentiating_node_index,
                    'priority': 5
                }
            ]

            # Создаем DataFrame для всех узлов
            nodes = pd.DataFrame(nodes_data)

            nodes = nodes.set_index('index')

            # Конкатенируем все DataFrame в один
            df = pd.concat([df, nodes])

        for i in range(num_inputs):
            df.iat[num_inputs+i, 2] = df.loc[df['type'] == 'potentiating'].index.tolist()
        
        presynaptic_range = df[df.type == 'presynaptic'].index.tolist()
        for i in df[df.type == 'postsynaptic'].index:
            df.iat[i, 2] = [p for p in presynaptic_range if p != i-(postsynaptic_node_index-presynaptic_node_index)]
            
        df['layer'] = layer
        
        if weights is None:
            weights = {n: np.random.randint(self.layer_params["wmin"][-1], self.layer_params["wmax"][-1], presynaptic_input.index.size).astype(np.float32) for n in df.loc[df['type'] == 'presynaptic'].index.tolist()}
            
        if isinstance(weights, np.ndarray):
            weights = {n: w for n, w in zip(df.loc[df['type'] == 'presynaptic'].index.tolist(), weights)}
        self.nodes = pd.concat((net, df))
        self.status = pd.concat((self.status, pd.DataFrame([
            {
                "index": n,
                "weights": w,
                "inhibited": -1
            } for n, w in weights.items()
        ]).set_index('index')))
        
        return cat_inputs.index.tolist()
    
    def stepwise_generator(self, data):
        vals_z = np.zeros(data.shape[1])
        nodes_sorted = self.nodes.sort_values("priority").loc[:, ['type', 'listening', 'broadcasting', 'layer']]
        
        net_it = list(zip(nodes_sorted.values, nodes_sorted.index))
        
        layer_params = self.layer_params
        
        status = self.status.to_dict()
        
        for i, vals in enumerate(data):
            t=i*self.dt
            layer = None
            for (node_type, listen, cast, _layer), node in net_it:
                n_val = 0
                if node_type == "input":
                    continue
                if _layer != layer:
                    layer = _layer
                    params = {k: layer_params[k][layer] for k in layer_params.keys()}
                    params["leak"] = np.exp(-self.dt/params["tau_leak"])
                    processor = self.processors[params["layer_type"]]
                if len(cast) == 1:
                    cast = cast[0]
                if len(listen) == 1:
                    listen = listen[0]
                vals, status = processor(node_type, listen, cast, status, vals, vals_z, params, node, t)       
                        
            vals_z = vals
            yield dict(zip(self.nodes.index, vals))
        old_weights = self.status.to_dict()
        old_weights.update(status)
        self.status = pd.DataFrame(old_weights)
    
    def process(self, node_type, listen, cast, status, vals, vals_z, params, node, t):
        
        n_val = vals_z[node]
        match node_type:
            case "buffer":
                n_val = vals_z[listen]
            case "ltp":
                if vals[listen]:
                    n_val = 1
                else:
                    n_val = vals_z[node]+1
            case "buffer":
                n_val = vals_z[listen]
            case "presynaptic":
                if status["inhibited"][node] < t:
                    n_val = (vals[listen]*status["weights"][node]).sum()+vals_z[node]*params["leak"]
                if params["wta"] and status["inhibited"][node] >= t:
                    n_val = 0
            case "postsynaptic":
                n_val = int(vals[listen]>params["thres"])
                if n_val:
                    status["inhibited"][listen] = t+params["tau_refractory"]
                    for b in cast:
                        status["inhibited"][b] = max(t+params["tau_inhibitory"], status["inhibited"][b]+params["tau_inhibitory"])
            case "potentiating":
                if vals[node-1] and params["learning"]:
                    nw = status["weights"][cast] + np.where(vals[listen]<params["tau_ltp"], params["ainc"], params["adec"])
                    nw = np.where(nw>params["wmax"], params["wmax"], nw)
                    status["weights"][cast] = np.where(nw<params["wmin"], params["wmin"], nw)
        vals[node] = n_val
        return vals, status
    
    def ttron_process(self, node_type, listen, cast, status, vals, vals_z, params, node, t):
        
        def change_weights(target_node, delays, time_offset, by):
            #не учитывать слишком новые импульсы, пришедшие после пика пресинаптического потенциала
            contrib_coeff = np.nan_to_num(np.exp(-np.where(delays<time_offset, np.nan, delays-time_offset)/params["tau_ltp"]), nan=0) 
            dw = contrib_coeff*by + status["weights"][target_node]
            dw = np.where(dw>params["wmax"], params["wmax"], dw)
            status["weights"][target_node] = np.where(dw<params["wmin"], params["wmin"], dw).astype(dtype=np.float32)
            
        n_val = vals_z[node]
        match node_type:
            case "teacher":
                n_val = vals[node]
                if listen is not None:
                    n_val = vals[listen]
            case "ltp":
                if n_val is not None:
                    n_val = vals_z[node]+1
                    if listen is not None:
                        if vals[listen]:
                            n_val = 1
                        
            case "max_tracker":
                if n_val == 0:
                    vals[cast] = 1                    
                if vals[listen] > n_val:
                    n_val = vals[listen]
                    vals[cast] = 1
            case "buffer":
                n_val = vals_z[listen]
            case "presynaptic":
                if status["inhibited"][node] < t:
                    n_val = (vals[listen]*status["weights"][node]).sum()+vals_z[node]*params["leak"]
                if (params["wta"] and status["inhibited"][node] >= t) or status["inhibited"][node]-params["tau_refractory"] == t-1:
                    n_val = 0
            case "postsynaptic":
                n_val = int(vals[listen]>params["thres"])
                if n_val:
                    vals[listen] = 0
                    status["inhibited"][listen] = t+params["tau_refractory"]
                    for b in cast:
                        status["inhibited"][b] = max(t+params["tau_inhibitory"], status["inhibited"][b]+params["tau_inhibitory"])
            case "potentiating":
                if params["learning"]:
                    teacher = vals[listen[0]]
                    postsynaptic_timer = vals[listen[1]]
                    max_tracker_timer = vals[listen[2]]
                    postsynaptic = vals[listen[3]]
                    ltp = vals[listen[4:]]
                    n_val = 0
                    if teacher and not np.isnan(postsynaptic_timer):
                        change_weights(cast, ltp, max_tracker_timer, params["ainc"])
                        vals[listen[1]] = None #отключаем таймер ожидания импульса от учителя
                        n_val = 1
                        vals[listen[2]-1]=0
                    if postsynaptic_timer > params["tau_ltp"]:
                        change_weights(cast, ltp, max_tracker_timer, params["adec"])
                        vals[listen[1]] = None #отключаем таймер ожидания импульса от учителя
                        n_val = -1
                        vals[listen[2]-1]=0
        vals[node] = n_val    
        return vals, status
    
    def feed_raw(self, data_raw, out_csv=None):
        self.status["inhibited"].values[:] = -1
        self.values = None
        data = pd.DataFrame(data_raw, columns=self.nodes.index).fillna(0).values
        s = self.stepwise_generator(data)
        out = [u for u in s]
        self.values = pd.DataFrame(out)
        if not out_csv is None:
            with open(out_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.values.columns)
                writer.writeheader()
                for row in out:
                    writer.writerow(row)
        return pd.DataFrame(out)        
    
    def error(self, answer_nodes):
        if not isinstance(answer_nodes, np.ndarray):
            answer_nodes = np.array(answer_nodes)
        output_nodes = self.nodes.query("type == 'postsynaptic'").index
        outputs = self.values.loc[:, output_nodes].values
        res = {o: {} for o in output_nodes}
        
        last = 0
        bins = []
        labels = []
        for i in self.values.index:
            if any(self.values.loc[i, answer_nodes]):
                bins.append((last, i))
                last = i
                labels.append(answer_nodes[np.where(self.values.loc[i, answer_nodes])[0][0]])
        for i, b in enumerate(bins):
            for o in output_nodes:
                if any(self.values.loc[b[0]:b[1], o]):
                    if labels[i] not in res[o]:
                        res[o][labels[i]] = 0
                    res[o][labels[i]] += 1
        for out in res.keys():
            tot = sum(res[out].values())
            for ans in res[out].keys():
                pos = res[out][ans]
                res[out][ans] = (labels.count(ans)-pos)/(labels.count(ans))+(tot-pos)/(len(labels)-labels.count(ans))
        return res
            
            
    