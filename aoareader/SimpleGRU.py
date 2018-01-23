import math
import torch
import warnings

from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
from torch.nn._functions.rnn import CudnnRNN, AutogradRNN, variable_recurrent_factory, StackedRNN
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend


def SimpleGRUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):

    # if input.is_cuda:
    if False:
        gi = F.linear(input, w_ih)
        gh = F.linear(hidden, w_hh)
        state = fusedBackend.GRUFused.apply
        return state(gi, gh, hidden) if b_ih is None else state(gi, gh, hidden, b_ih, b_hh)

    gi = F.linear(input, w_ih, b_ih)
    gh = F.linear(hidden, w_hh, b_hh)
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    #resetgate = F.sigmoid(i_r + h_r)
    inputgate = F.sigmoid(i_i + h_i)
    #newgate = F.tanh(i_n + resetgate * h_n)
    newgate = F.relu(i_n + h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy


def SimpleAutogradRNN(mode, input_size, hidden_size, num_layers=1, batch_first=False,
                dropout=0, train=True, bidirectional=False, batch_sizes=None,
                dropout_state=None, flat_weight=None):

    if mode == 'RNN_RELU':
        cell = RNNReLUCell
    elif mode == 'RNN_TANH':
        cell = RNNTanhCell
    elif mode == 'LSTM':
        cell = LSTMCell
    elif mode == 'GRU':
        cell = SimpleGRUCell
    else:
        raise Exception('Unknown mode: {}'.format(mode))

    if batch_sizes is None:
        rec_factory = Recurrent
    else:
        rec_factory = variable_recurrent_factory(batch_sizes)

    if bidirectional:
        layer = (rec_factory(cell), rec_factory(cell, reverse=True))
    else:
        layer = (rec_factory(cell),)

    func = StackedRNN(layer,
                      num_layers,
                      (mode == 'LSTM'),
                      dropout=dropout,
                      train=train)

    def forward(input, weight, hidden):
        if batch_first and batch_sizes is None:
            input = input.transpose(0, 1)

        nexth, output = func(input, hidden, weight)

        if batch_first and batch_sizes is None:
            output = output.transpose(0, 1)

        return output, nexth

    return forward


def SimpleRNN(*args, **kwargs):

    def forward(input, *fargs, **fkwargs):
        #if cudnn.is_acceptable(input.data):
        if False:
            func = CudnnRNN(*args, **kwargs)
        else:
            func = SimpleAutogradRNN(*args, **kwargs)

        # Hack for the tracer that allows us to represent RNNs as single
        # nodes and export them to ONNX in this form
        # It can be also used as a decorator at the higher level
        # Check the first argument explicitly to reduce the overhead of creating
        # the lambda
        import torch
        if torch._C._jit_is_tracing(input):
            import torch.onnx.symbolic
            func = torch.onnx.symbolic.RNN_symbolic_builder(*args, **kwargs)(func)

        return func(input, *fargs, **fkwargs)

    return forward


class SimpleRNNBase(torch.nn.Module):

    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False):
        super(SimpleRNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        elif mode == 'SimpleGRU':
            gate_size = 3 * hidden_size
        else:
            gate_size = hidden_size

        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = Parameter(torch.Tensor(gate_size))
                b_hh = Parameter(torch.Tensor(gate_size))
                layer_params = (w_ih, w_hh, b_ih, b_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        self.flatten_parameters()
        self.reset_parameters()

        self._backend.register_function('SimpleRNN', SimpleRNN)

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            self._data_ptrs = []
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        unique_data_ptrs = set(p.data_ptr() for l in self.all_weights for p in l)
        if len(unique_data_ptrs) != sum(len(l) for l in self.all_weights):
            self._data_ptrs = []
            return

        with torch.cuda.device_of(any_param):
            # This is quite ugly, but it allows us to reuse the cuDNN code without larger
            # modifications. It's really a low-level API that doesn't belong in here, but
            # let's make this exception.
            from torch.backends.cudnn import rnn
            from torch.backends import cudnn
            from torch.nn._functions.rnn import CudnnRNN
            handle = cudnn.get_handle()
            with warnings.catch_warnings(record=True):
                fn = CudnnRNN(
                    self.mode,
                    self.input_size,
                    self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=self.batch_first,
                    dropout=self.dropout,
                    train=self.training,
                    bidirectional=self.bidirectional,
                    dropout_state=self.dropout_state,
                )

            # Initialize descriptors
            fn.datatype = cudnn._typemap[any_param.type()]
            fn.x_descs = cudnn.descriptor(any_param.new(1, self.input_size), 1)
            fn.rnn_desc = rnn.init_rnn_descriptor(fn, handle)

            # Allocate buffer to hold the weights
            self._param_buf_size = rnn.get_num_weights(handle, fn.rnn_desc, fn.x_descs[0], fn.datatype)
            fn.weight_buf = any_param.new(self._param_buf_size).zero_()
            fn.w_desc = rnn.init_weight_descriptor(fn, fn.weight_buf)

            # Slice off views into weight_buf
            all_weights = [[p.data for p in l] for l in self.all_weights]
            params = rnn.get_parameters(fn, handle, fn.weight_buf)

            # Copy weights and update their storage
            rnn._copyParams(all_weights, params)
            for orig_layer_param, new_layer_param in zip(all_weights, params):
                for orig_param, new_param in zip(orig_layer_param, new_layer_param):
                    orig_param.set_(new_param.view_as(orig_param))

            self._data_ptrs = list(p.data.data_ptr() for p in self.parameters())

    def _apply(self, fn):
        ret = super(SimpleRNNBase, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = batch_sizes[0]
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)

        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode == 'LSTM':
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.autograd.Variable(input.data.new(self.num_layers *
                                                        num_directions,
                                                        max_batch_size,
                                                        self.hidden_size).zero_(), requires_grad=False)
            if self.mode == 'LSTM':
                hx = (hx, hx)

        has_flat_weights = list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
        if has_flat_weights:
            first_data = next(self.parameters()).data
            assert first_data.storage().size() == self._param_buf_size
            flat_weight = first_data.new().set_(first_data.storage(), 0, torch.Size([self._param_buf_size]))
        else:
            flat_weight = None

        self.check_forward_args(input, hx, batch_sizes)
        func = self._backend.SimpleRNN(
            self.mode,
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            batch_sizes=batch_sizes,
            dropout_state=self.dropout_state,
            flat_weight=flat_weight
        )
        output, hidden = func(input, self.all_weights, hx)
        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def __setstate__(self, d):
        super(RNNBase, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:2]]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]


class SimpleGRU(SimpleRNNBase):
    def __init__(self, *args, **kwargs):
        super(SimpleGRU, self).__init__('GRU', *args, **kwargs)
