import numpy
from data_iterator import DataIterator
#from model import *
import time
import random
import sys
from utils import calc_auc, calc_acc
import openvino.runtime as ov

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='./openvino/FP32/DIEN.xml', help="path to openvino IR model")
parser.add_argument("--seed", type=int, default=3, help="seed value")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--data_type", type=str, default='FP32', help="data type: FP32 or BF16")
args = parser.parse_args()

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0

def prepare_data(input, target, maxlen = None, return_neg = False):
    # x: a list of sentences
    lengths_x = [len(s[4]) for s in input]
    seqs_mid = [inp[3] for inp in input]
    seqs_cat = [inp[4] for inp in input]
    noclk_seqs_mid = [inp[5] for inp in input]
    noclk_seqs_cat = [inp[6] for inp in input]
    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_noclk_seqs_mid.append(inp[5])
                new_noclk_seqs_cat.append(inp[6])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat

        if len(lengths_x) < 1:
            return None, None, None, None
    
    n_samples = len(seqs_mid)
    maxlen_x = numpy.max(lengths_x)
    neg_samples = len(noclk_seqs_mid[0][0])


    mid_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    cat_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    noclk_mid_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    noclk_cat_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    data_type = 'float32'
    """
    if args.data_type == 'FP32':
        data_type = 'float32'
    elif args.data_type == 'FP16':
        data_type = 'float16'
    else:
        raise ValueError("Invalid model data type: %s" % args.data_type)
    """
    mid_mask = numpy.zeros((n_samples, maxlen_x)).astype(data_type)
    for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy

    uids = numpy.array([inp[0] for inp in input])
    mids = numpy.array([inp[1] for inp in input])
    cats = numpy.array([inp[2] for inp in input])

    if return_neg:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x), noclk_mid_his, noclk_cat_his

    else:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x)

def test(
        train_file = "local_train_splitByUser",
        test_file = "local_test_splitByUser",
        uid_voc = "uid_voc.pkl",
        mid_voc = "mid_voc.pkl",
        cat_voc = "cat_voc.pkl",
        batch_size = 128,
        maxlen = 100,
        model_path = "",
        data_type = 'FP32',
        seed = 2
):
    core = ov.Core()
    if data_type == "BF16":
        core.set_property("CPU", {"INFERENCE_PRECISION_HINT": "bf16"})
        print("Warning, set inference precision as BF16, please ensure that platform support BF16 ISA")
    elif data_type == "FP32":
        print("set inference precision as FP32")
        core.set_property("CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    compiled_model = core.compile_model(model_path, "CPU")
    test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
    test_auc, test_acc, eval_time, num_iters = eval(test_data, compiled_model)
    print('test_auc: %.4f  ---- test_accuracy: %.9f  ---- eval_time: %.3f' % (test_auc, test_acc, eval_time))
    approximate_accelerator_time=eval_time
    print("Total recommendations: %d" % (num_iters*batch_size))
    print("Approximate accelerator time in seconds is %.3f" % approximate_accelerator_time)
    print("Approximate accelerator performance in recommendations/second is %.3f" % (float(num_iters*batch_size)/float(approximate_accelerator_time)))

def eval(test_data, compiled_model):
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    eval_time = 0
    for src, tgt in test_data:
        nums += 1
        sys.stdout.flush()
        print("Prepare input data...")
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, return_neg=True)
        print("begin evaluation")
        start_time = time.time()
        inputs = {"Inputs/uid_batch_ph" : uids,
                  "Inputs/mid_batch_ph" : mids,
                  "Inputs/cat_batch_ph" : cats,
                  "Inputs/mid_his_batch_ph" : mid_his,
                  "Inputs/cat_his_batch_ph" : cat_his,
                  "Inputs/seq_len_ph" : sl,
                  "Inputs/mask" : mid_mask,
                }
        print("Run inference with openvino ...")
        results = compiled_model(inputs)
        prob = results[compiled_model.output(0)]
        end_time = time.time()
        print("evaluation time of one batch: %.3f" % (end_time - start_time))
        print("end evaluation")
        eval_time += end_time - start_time
        #loss_sum += loss
        #aux_loss_sum = aux_loss
        #accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p ,t in zip(prob_1, target_1):
            stored_arr.append([p, t])
        print("nums: ", nums)
        # break
    print("Calculate precision ...")
    test_auc = calc_auc(stored_arr)
    test_acc = calc_acc(stored_arr)
    
    return test_auc, test_acc, eval_time, nums

if __name__ == '__main__':
    SEED = args.seed
    numpy.random.seed(SEED)
    random.seed(SEED)
    test(model_path=args.model_path, seed=SEED, batch_size=args.batch_size, data_type=args.data_type)
