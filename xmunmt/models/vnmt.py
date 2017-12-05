# coding=utf-8
# Author: Zhixing Tan, Chong Ruan
# Contact: playinf@stu.xmu.edu.cn, pkurc@pku.edu.cn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import xmunmt.layers as layers
import xmunmt.utils.search as search

from .model import NMTModel


def _copy_through(time, length, output, new_output):
    copy_cond = (time >= length)
    return tf.where(copy_cond, output, new_output)


def _gru_encoder(cell, inputs, sequence_length, initial_state, dtype=None):
    # inputs: [batch_size, time_steps, hidden_size]
    # Assume that the underlying cell is GRUCell-like
    output_size = cell.output_size
    dtype = dtype or inputs.dtype

    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]

    zero_output = tf.zeros([batch, output_size], dtype)

    if initial_state is None:
        initial_state = cell.zero_state(batch, dtype)

    input_ta = tf.TensorArray(dtype, time_steps,
                              tensor_array_name="input_array")
    output_ta = tf.TensorArray(dtype, time_steps,
                               tensor_array_name="output_array")
    input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))

    def loop_func(t, out_ta, state):
        inp_t = input_ta.read(t)
        cell_output, new_state = cell(inp_t, state)
        cell_output = _copy_through(t, sequence_length, zero_output,
                                    cell_output)
        new_state = _copy_through(t, sequence_length, state, new_state)
        out_ta = out_ta.write(t, cell_output)
        return t + 1, out_ta, new_state

    time = tf.constant(0, dtype=tf.int32, name="time")
    loop_vars = (time, output_ta, initial_state)

    outputs = tf.while_loop(lambda t, *_: t < time_steps, loop_func,
                            loop_vars, parallel_iterations=32,
                            swap_memory=True)

    output_final_ta = outputs[1]
    final_state = outputs[2]

    all_output = output_final_ta.stack()
    all_output.set_shape([None, None, output_size])
    all_output = tf.transpose(all_output, [1, 0, 2])

    return all_output, final_state


def _encoder(cell_fw, cell_bw, inputs, sequence_length, dtype=None,
             scope=None, reuse=None):
    with tf.variable_scope(scope or "encoder",
                           values=[inputs, sequence_length], reuse=reuse):
        inputs_fw = inputs
        inputs_bw = tf.reverse_sequence(inputs, sequence_length,
                                        batch_axis=0, seq_axis=1)

        with tf.variable_scope("forward"):
            output_fw, state_fw = _gru_encoder(cell_fw, inputs_fw,
                                               sequence_length, None,
                                               dtype=dtype)

        with tf.variable_scope("backward"):
            output_bw, state_bw = _gru_encoder(cell_bw, inputs_bw,
                                               sequence_length, None,
                                               dtype=dtype)
            output_bw = tf.reverse_sequence(output_bw, sequence_length,
                                            batch_axis=0, seq_axis=1)

        # [batch_size, time_steps, hidden_size*2]
        annotation = tf.concat([output_fw, output_bw], axis=2)

        results = {
            "annotation": annotation,
            "average": tf.reduce_sum(annotation, axis=1, keep_dims=False) / # [batch_size, hidden_size*2]
                       tf.reshape(tf.cast(sequence_length, tf.float32), shape=[tf.shape(inputs)[0], 1]),
            "outputs": {
                "forward": output_fw,
                "backward": output_bw
            },
            "final_states": {
                "forward": state_fw,
                "backward": state_bw
            }
        }

        return results


def _inferer(encoder_hs, d_z, dtype=None, scope=None):
    # Args:
    # encoder_hs: encoder hidden states used to calculate Gaussian means and covariances.
    #             Could be either a Tensor or a list of tensors of shape [batch_size, input_size].
    #             e.g.: pass h_f or [h_f] for prior distribution, and [h_f, h_e] for posterior distribution.

    # Returns: a dictionary with key "mu" and "log_sigma2", which are Gaussian means and log variances
    #           and both are of shape [batch_size, d_z]
    with tf.variable_scope(scope or "inferer"):
        h_z_prime = tf.tanh(layers.nn.linear(encoder_hs, d_z, bias=True, scope="h_z_prime"))
        mu = layers.nn.linear(h_z_prime, d_z, bias=True, scope="mu")
        log_sigma2 = layers.nn.linear(h_z_prime, d_z, bias=True, scope="log_sigma2")
    return {"mu": mu, "log_sigma2":log_sigma2}


def _decoder(cell, inputs, memory, sequence_length, initial_state, h_e_prime, dtype=None,
             scope=None):
    # h_e_prime: a Tensor with shape [batch_size, params.e_prime_size], global semantics to facilitate inference

    # inputs: [batch_size, time_steps, dimension]
    # Assume that the underlying cell is GRUCell-like
    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]
    dtype = dtype or inputs.dtype
    output_size = cell.output_size
    zero_output = tf.zeros([batch, output_size], dtype)
    zero_value = tf.zeros([batch, memory.shape[-1].value], dtype)

    with tf.variable_scope(scope or "decoder", dtype=dtype):
        inputs = tf.transpose(inputs, [1, 0, 2])
        mem_mask = tf.sequence_mask(sequence_length["source"],
                                    maxlen=tf.shape(memory)[1],
                                    dtype=tf.float32)
        bias = layers.attention.attention_bias(mem_mask)
        cache = layers.attention.attention(None, memory, None, output_size)

        input_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="input_array")
        output_ta = tf.TensorArray(tf.float32, time_steps,
                                   tensor_array_name="output_array")
        value_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="value_array")
        alpha_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="alpha_array")
        input_ta = input_ta.unstack(inputs)
        initial_state = layers.nn.linear(initial_state, output_size, True,
                                         scope="s_transform")
        initial_state = tf.tanh(initial_state)

        def loop_func(t, out_ta, att_ta, val_ta, state, cache_key):
            # att_ta: attention weights
            # val_ta: context vectors
            inp_t = input_ta.read(t)
            results = layers.attention.attention(state, memory, bias,
                                                 output_size,
                                                 cache={"key": cache_key})
            alpha = results["weight"]
            context = results["value"]
            cell_input = [inp_t, context, h_e_prime]
            cell_output, new_state = cell(cell_input, state)
            cell_output = _copy_through(t, sequence_length["target"],
                                        zero_output, cell_output)
            new_state = _copy_through(t, sequence_length["target"], state,
                                      new_state)
            new_value = _copy_through(t, sequence_length["target"], zero_value,
                                      context)

            out_ta = out_ta.write(t, cell_output)
            att_ta = att_ta.write(t, alpha)
            val_ta = val_ta.write(t, new_value)
            cache_key = tf.identity(cache_key)
            return t + 1, out_ta, att_ta, val_ta, new_state, cache_key

        time = tf.constant(0, dtype=tf.int32, name="time")
        loop_vars = (time, output_ta, alpha_ta, value_ta, initial_state,
                     cache["key"])

        outputs = tf.while_loop(lambda t, *_: t < time_steps,
                                loop_func, loop_vars,
                                parallel_iterations=32,
                                swap_memory=True)

        output_final_ta = outputs[1]
        value_final_ta = outputs[3]

        final_output = output_final_ta.stack()
        final_output.set_shape([None, None, output_size])
        final_output = tf.transpose(final_output, [1, 0, 2])

        final_value = value_final_ta.stack()
        final_value.set_shape([None, None, memory.shape[-1].value])
        final_value = tf.transpose(final_value, [1, 0, 2])

        result = {
            "outputs": final_output,
            "values": final_value,
            "initial_state": initial_state
        }

    return result


def model_graph(features, labels, params):
    src_vocab_size = len(params.vocabulary["source"])
    tgt_vocab_size = len(params.vocabulary["target"])

    with tf.variable_scope("source_embedding"):
        src_emb = tf.get_variable("embedding",
                                  [src_vocab_size, params.embedding_size])
        src_bias = tf.get_variable("bias", [params.embedding_size])
        src_inputs = tf.nn.embedding_lookup(src_emb, features["source"])

    with tf.variable_scope("target_embedding"):
        tgt_emb = tf.get_variable("embedding",
                                  [tgt_vocab_size, params.embedding_size])
        tgt_bias = tf.get_variable("bias", [params.embedding_size])
        tgt_inputs = tf.nn.embedding_lookup(tgt_emb, features["target"])

    batch_size = tf.shape(src_inputs)[0]

    src_inputs = tf.nn.bias_add(src_inputs, src_bias)
    tgt_inputs = tf.nn.bias_add(tgt_inputs, tgt_bias)

    # src_inputs, tgt_inputs: [batch_size, time_steps, embedding_size]

    if params.dropout and not params.use_variational_dropout:
        src_inputs = tf.nn.dropout(src_inputs, 1.0 - params.dropout)
        tgt_inputs = tf.nn.dropout(tgt_inputs, 1.0 - params.dropout)

    # encoder
    cell_fw = layers.rnn_cell.LegacyGRUCell(params.hidden_size)
    cell_bw = layers.rnn_cell.LegacyGRUCell(params.hidden_size)

    if params.use_variational_dropout:
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(
            cell_fw,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            input_size=params.embedding_size,
            dtype=tf.float32
        )
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(
            cell_bw,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            input_size=params.embedding_size,
            dtype=tf.float32
        )

    encoder_output = _encoder(cell_fw, cell_bw, src_inputs,
                              features["source_length"])



    prior_dist = _inferer([encoder_output["average"]], d_z=params.z_size, dtype=None, scope="prior_distribution")




    # `sigma_prior` corresponds to covariance of prior distribution p(z|x), i.e.: \sigma' in paper
    sigma_prior = tf.exp(prior_dist["log_sigma2"] / 2.0)
    h_z_prior = prior_dist["mu"]


    if params.enable_KL:
        # Sample h_z from posterior distribution and calculate KL divergence
        encoder_output_tgt = _encoder(cell_fw, cell_bw, tgt_inputs,
                                      features["target_length"], reuse=True)
        posterior_dist = _inferer([encoder_output["average"], encoder_output_tgt["average"]],
                                  d_z=params.z_size, dtype=None, scope="posterior_distribution")
        # `sigma_posterior` corresponds to covariance of posterior distribution q(z|x, y), i.e.: \sigma in paper
        sigma_posterior = tf.exp(posterior_dist["log_sigma2"] / 2.0)

        # Re-parametric trick
        # normal_noise = tf.random_normal(tf.shape(encoder_output["average"]))
        h_z = posterior_dist["mu"] + sigma_posterior * features["normal_noise"]

        # KL-divergence: suppose we have two d-dimensional Gaussian: P1(x) = N(\mu1, \Sigma1), P2(x) = N(\mu2, \Sigma2),
        #   then KL(P1 || P2) = 1/2 * { log {det(\Sigma2) / det(\Sigma1)} - d +
        #                               tr{\Sigma2^{-1} \Sigma1} + (\mu2 - \mu1)^T \Sigma2^{-1} (\mu2 - \mu1) }

        # If covariance matrices are diagonal, denote diag(\Sigma) by \sigma, we have:
        #   KL(P1 || P2) = 1/2 * { log {\prod \sigma2 / \prod \sigma1} - d +
        #                               \sum_i {\sigma1 / \sigma2} + \sum{\sigma2^{-1} * (\mu2 - \mu1)^2} }

        print("sigma prior: ", sigma_prior)
        print("posterior mu: ", posterior_dist["mu"])
        # KL(posterior || prior):

        divergence = 0.5 * (tf.reduce_sum(prior_dist["log_sigma2"]) - tf.reduce_sum(posterior_dist["log_sigma2"])) \
                     + tf.reduce_sum(sigma_posterior / sigma_prior) \
                     + tf.reduce_sum(tf.square(prior_dist["mu"] - posterior_dist["mu"]) / sigma_prior)
        divergence = 0.5 * (divergence / tf.cast(batch_size, tf.float32) - tf.cast(params.e_prime_size, tf.float32))
    else:
        h_z = h_z_prior
        divergence = 0.0

    # [batch_size, params.e_prime_size]
    h_e_prime = tf.tanh(layers.nn.linear(h_z, output_size=params.e_prime_size, bias=True, scope="h_e_prime"))

    # decoder
    cell = layers.rnn_cell.LegacyGRUCell(params.hidden_size)

    if params.use_variational_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            # input + context + global latent variable information
            input_size=params.embedding_size + 2 * params.hidden_size + params.e_prime_size,
            dtype=tf.float32
        )

    length = {
        "source": features["source_length"],
        "target": features["target_length"]
    }
    initial_state = encoder_output["final_states"]["backward"]
    decoder_output = _decoder(cell, tgt_inputs, encoder_output["annotation"],
                              length, initial_state, h_e_prime)

    # Shift left
    shifted_tgt_inputs = tf.pad(tgt_inputs, [[0, 0], [1, 0], [0, 0]])
    shifted_tgt_inputs = shifted_tgt_inputs[:, :-1, :]

    all_outputs = tf.concat(
        [
            tf.expand_dims(decoder_output["initial_state"], axis=1),
            decoder_output["outputs"],
        ],
        axis=1
    )
    shifted_outputs = all_outputs[:, :-1, :]

    maxout_features = [
        shifted_tgt_inputs,
        shifted_outputs,
        decoder_output["values"]
    ]
    maxout_size = params.hidden_size // params.maxnum

    if labels is None:
        # Special case for non-incremental decoding
        maxout_features = [
            shifted_tgt_inputs[:, -1, :],
            shifted_outputs[:, -1, :],
            decoder_output["values"][:, -1, :]
        ]
        maxhid = layers.nn.maxout(maxout_features, maxout_size, params.maxnum,
                                  concat=False)
        readout = layers.nn.linear(maxhid, params.embedding_size, False,
                                   scope="deepout")

        # Prediction
        logits = layers.nn.linear(readout, tgt_vocab_size, True,
                                  scope="softmax")

        return logits

    maxhid = layers.nn.maxout(maxout_features, maxout_size, params.maxnum,
                              concat=False)
    readout = layers.nn.linear(maxhid, params.embedding_size, False,
                               scope="deepout")

    if params.dropout and not params.use_variational_dropout:
        readout = tf.nn.dropout(readout, 1.0 - params.dropout)

    # Prediction
    logits = layers.nn.linear(readout, tgt_vocab_size, True, scope="softmax")
    logits = tf.reshape(logits, [-1, tgt_vocab_size])

    ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        label_smoothing=params.label_smoothing,
        normalize=True
    )

    ce = tf.reshape(ce, tf.shape(labels))
    tgt_mask = tf.to_float(
        tf.sequence_mask(
            features["target_length"],
            maxlen=tf.shape(features["target"])[1]
        )
    )
    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask) + divergence

    return loss, divergence


class VNMT(NMTModel):
    """
    Reference:
        Variational Neural Machine Translation
    """

    def __init__(self, params, scope="vnmt"):
        super(VNMT, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer):
        def training_fn(features, params=None):
            if params is None:
                params = self.parameters
            with tf.variable_scope(self._scope, initializer=initializer):
                loss = model_graph(features, features["target"], params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)
            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0
            params.enable_KL = False

            with tf.variable_scope(self._scope):
                logits = model_graph(features, None, params)

            return logits

        return evaluation_fn

    def get_inference_func(self):
        def inference_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)
            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0
            params.enable_KL = False

            with tf.variable_scope(self._scope):
                logits = model_graph(features, None, params)

            return logits

        return inference_fn

    @staticmethod
    def get_name():
        return "vnmt"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            # vocabulary
            pad="</s>",
            unk="UNK",
            eos="</s>",
            bos="</s>",
            append_eos=False,
            # model
            rnn_cell="LegacyGRUCell",
            embedding_size=620,
            hidden_size=1000,
            maxnum=2,
            # regularization
            dropout=0.2,
            use_variational_dropout=False,
            label_smoothing=0.1,
            constant_batch_size=True,
            batch_size=128,
            max_length=80,
            clip_grad_norm=5.0,
            z_size=2000,
            e_prime_size=2000,
            enable_KL=True
        )

        return params
