from tqdm import tqdm, trange
from itertools import cycle
import torch
from torch.nn.utils import clip_grad_norm_
from sklearn_crfsuite import metrics
from configs.bert_word_config import bert_config
import os
import codecs


def reformat_valid_label_ids(gold_label_ids, input_lens, gold_valid_ids_list, query_len_list=None):
    # new_gold_label_ids = []
    for index, label_ids in enumerate(gold_label_ids):
        if query_len_list is not None:
            gold_valid_ids_list.append(label_ids[query_len_list[index]:input_lens[index]])
        else:
            gold_valid_ids_list.append(label_ids[0:input_lens[index]])


def trainer(model, optimizer, train_data, valid_data, epochs, num_train_optimization_steps, each_epoch_train_steps,
            id2label, device, logger, args):
    bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
    train_dataloader = cycle(train_data)
    tr_loss = 0
    global_step = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    best_score = 0
    orig_text_list = []
    with codecs.open("data/orig_data_dev.txt", 'r', 'utf-8') as fr:
        for i, line in enumerate(fr):
            if i % 2 == 0:
                text = line.strip("\n")
                text_split = text.split(" ")
                orig_text_list.append(text_split)
    for step in bar:
        model.train()
        batch = next(train_dataloader)
        batch = tuple(t.to(device) for t in batch)
        # all_input_ids,all_word_seq_lens_baidu,all_word_seq_lens_thu,all_word_seq_lens_ltp,all_word_slice_baidu,all_word_slice_thu,all_word_slice_ltp,all_segment_ids,all_input_mask,all_label_ids,all_input_lens
        train_data_X, train_word_length_list_baidu, train_word_length_list_thu, train_word_length_list_ltp, train_word_slice_list_baidu, train_word_slice_list_thu, train_word_slice_list_ltp, train_input_segment_ids, train_input_masks, train_data_Y, train_input_lens = batch
        # input_ids, word_length_1, word_length_2, word_length_3, word_slice_1, word_slice_2, word_slice_3,token_type_ids=None,input_lens=None, attention_mask=None, labels=None
        _, loss = model(train_data_X, train_word_length_list_baidu, train_word_length_list_thu,
                        train_word_length_list_ltp, train_word_slice_list_baidu, train_word_slice_list_thu,
                        train_word_slice_list_ltp, train_input_segment_ids, train_input_lens, train_input_masks,
                        train_data_Y)
        tr_loss += loss.mean().item()
        train_loss = round(tr_loss / (nb_tr_steps + 1), 4)
        bar.set_description("train loss {}".format(train_loss))
        nb_tr_steps += 1
        loss.backward()
        # clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        if (step + 1) % each_epoch_train_steps == 0:
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            logger.info("***** Report result *****")
            logger.info("  %s = %s", 'global_step', str(global_step))
            logger.info("  %s = %s", 'train loss', str(train_loss))
            logger.info("***** Running evaluation *****")
            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            pred_valid_ids_list = []
            gold_valid_ids_list = []
            for valid_batch in valid_data:
                valid_batch = tuple(t.to(device) for t in valid_batch)
                dev_data_X, dev_word_length_list_baidu, dev_word_length_list_thu, dev_word_length_list_ltp, dev_word_slice_list_baidu, dev_word_slice_list_thu, dev_word_slice_list_ltp, dev_input_segment_ids, dev_input_masks, dev_data_Y, dev_input_lens = valid_batch
                with torch.no_grad():
                    pred_ids, valid_loss = model(dev_data_X, dev_word_length_list_baidu, dev_word_length_list_thu,
                                                 dev_word_length_list_ltp, dev_word_slice_list_baidu,
                                                 dev_word_slice_list_thu, dev_word_slice_list_ltp,
                                                 dev_input_segment_ids, dev_input_lens, dev_input_masks, dev_data_Y)

                    # pred_ids,_ = model.crf._obtain_labels(pred_logits,id2label,dev_input_lens)
                eval_loss += valid_loss.mean().item()
                valid_label_ids = dev_data_Y.to('cpu').numpy().tolist()
                pred_ids = pred_ids.to('cpu').numpy().tolist()
                reformat_valid_label_ids(valid_label_ids, dev_input_lens.to('cpu').numpy().tolist(),
                                         gold_valid_ids_list)
                reformat_valid_label_ids(pred_ids, dev_input_lens.to('cpu').numpy().tolist(), pred_valid_ids_list)
                # gold_valid_ids_list.extend(valid_label_ids)
                # print(valid_label_ids)
                # print(pred_ids)
                # pred_valid_ids_list.extend(pred_ids.to('cpu').numpy().tolist())
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            logger.info("  %s = %s", 'valid loss', str(eval_loss))

            gold_entity_list = gen_entity_from_label_id_list(orig_text_list, gold_valid_ids_list, id2label)
            pred_entity_list = gen_entity_from_label_id_list(orig_text_list, pred_valid_ids_list, id2label)
            valid_f1 = cal_mertric_from_two_list(pred_entity_list, gold_entity_list)

            # f1_score = metrics.flat_f1_score(gold_valid_ids_list, pred_valid_ids_list,average='macro')
            # logger.info("  %s = %s", 'valid f1 score', str(f1_score))
            if valid_f1 > best_score:
                best_score = valid_f1
                print("=" * 80)
                print("cur loss", eval_loss)
                print("best loss", best_score)
                print("Saving Model......")
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                if not os.path.exists(bert_config.get(args.model_checkpoint_dir)):
                    os.makedirs(bert_config.get(args.model_checkpoint_dir))
                output_model_file = os.path.join(bert_config.get(args.model_checkpoint_dir), "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                print("=" * 80)


def gen_entity_from_label_id_list(text_lists, label_id_list, id2slot_dict, is_orig=False):
    """
    B-LOC
    B-PER
    B-ORG
    I-LOC
    I-ORG
    I-PER
    :param label_id_list:
    :param id2slot_dict:
    :return:
    """
    entity_list = []
    # 存index
    buffer_list = []
    cur_label = ""
    for i, label_ids in enumerate(label_id_list):
        cur_entity_list = []
        if not is_orig:
            label_list = [id2slot_dict.get(label_ele) for label_ele in label_ids]
            label_list = label_list[1:-1]
        else:
            label_list = label_ids
        text_list = text_lists[i]
        if len(text_list) < len(label_list):
            print(text_list)
            print(label_list)
        # label_list = []
        # label_list
        # print(label_list)
        for j, label in enumerate(label_list):
            if not label.__contains__("-"):
                if len(buffer_list) == 0:
                    continue
                else:
                    # print(buffer_list)
                    # print(text_list)
                    buffer_char_list = [text_list[index] for index in buffer_list]
                    label_cur = label_list[j - 1].split("-")[1]
                    buffer_word = "".join(buffer_char_list)
                    cur_entity_list.append(buffer_word + label_cur)
                    buffer_list.clear()
            else:
                if len(buffer_list) == 0:
                    if label.startswith("B"):
                        # 必须以B开头，否则说明有问题，不能加入
                        buffer_list.append(j)
                else:
                    buffer_last_index = buffer_list[-1]
                    buffer_last_label = label_list[buffer_last_index]
                    split_label = buffer_last_label.split("-")
                    buffer_last_label_prefix, buffer_last_label_type = split_label[0], split_label[1]
                    cur_label_split = label.split("-")
                    cur_label_prefix, cur_label_type = cur_label_split[0], cur_label_split[1]
                    # B+B
                    if buffer_last_label_prefix == "B" and cur_label_prefix == "B":
                        cur_entity_list.append(text_list[buffer_list[-1]] + buffer_last_label_type)
                        buffer_list.clear()
                        buffer_list.append(j)
                    elif buffer_last_label_prefix == "I" and cur_label_prefix == "B":
                        buffer_char_list = [text_list[index] for index in buffer_list]
                        buffer_word = "".join(buffer_char_list)
                        cur_entity_list.append(buffer_word + buffer_last_label_type)
                        buffer_list.clear()
                        buffer_list.append(j)
                    elif buffer_last_label_prefix == "B" and cur_label_prefix == "I":
                        # analyze type
                        if buffer_last_label_type == cur_label_type:
                            buffer_list.append(j)
                        else:
                            cur_entity_list.append(text_list[buffer_list[-1]] + buffer_last_label_type)
                            buffer_list.clear()
                            # 这种情况出现在预测有问题，即一个I的label不应当作为一个实体的起始。
                            # buffer_list.append(j)
                    else:
                        # I + I
                        # analyze type
                        if buffer_last_label_type == cur_label_type:
                            buffer_list.append(j)
                        else:
                            cur_entity_list.append(text_list[buffer_list[-1]])
                            buffer_list.clear()
                            buffer_list.append(j)
        if buffer_list:
            buffer_char_list = [text_list[index] for index in buffer_list]
            buffer_word = "".join(buffer_char_list)
            cur_label = label_list[buffer_list[0]].split("-")[1]
            cur_entity_list.append(buffer_word + cur_label)
            buffer_list.clear()
        entity_list.append(cur_entity_list)
    return entity_list


def cal_mertric_from_two_list(prediction_list, true_list):
    tp, fp, fn = 0, 0, 0
    for pred_entity, true_entity in zip(prediction_list, true_list):
        pred_entity_set = set(pred_entity)
        true_entity_set = set(true_entity)
        tp += len(true_entity_set & pred_entity_set)
        fp += len(pred_entity_set - true_entity_set)
        fn += len(true_entity_set - pred_entity_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec)
    print("span_level pre micro_avg:{}".format(prec))
    print("span_level rec micro_avg:{}".format(rec))
    print("span_level f1 micro_avg:{}".format(f1))
    return f1


def predict_all_and_evaluate(model, test_data, id2label, device, logger, orig_test_file, args):
    model.eval()
    gold_valid_ids_list = []
    predict_ids_list = []
    for valid_batch in test_data:
        valid_batch = tuple(t.to(device) for t in valid_batch)
        dev_data_X, dev_word_length_list_baidu, dev_word_length_list_thu, dev_word_length_list_ltp, dev_word_slice_list_baidu, dev_word_slice_list_thu, dev_word_slice_list_ltp, dev_input_segment_ids, dev_input_masks, dev_data_Y, dev_input_lens = valid_batch
        with torch.no_grad():
            # pred_logits = model(dev_data_X,dev_word_length_list_baidu,dev_word_length_list_thu,dev_word_length_list_ltp,dev_word_slice_list_baidu,dev_word_slice_list_thu,dev_word_slice_list_ltp,dev_input_segment_ids,dev_input_lens,dev_input_masks,None)
            pred_ids = model(dev_data_X, dev_word_length_list_baidu, dev_word_length_list_thu, dev_word_length_list_ltp,
                             dev_word_slice_list_baidu, dev_word_slice_list_thu, dev_word_slice_list_ltp,
                             dev_input_segment_ids, dev_input_lens, dev_input_masks, None)

            # pred_ids,_ = model.crf._obtain_labels(pred_logits,id2label,dev_input_lens)
        valid_label_ids = dev_data_Y.to('cpu').numpy().tolist()
        gold_valid_ids_list.extend(valid_label_ids)
        # reformat_valid_label_ids(valid_label_ids,dev_input_lens.to('cpu').numpy().tolist(),gold_valid_ids_list)
        pred_ids = pred_ids.to('cpu').numpy().tolist()
        reformat_valid_label_ids(pred_ids, dev_input_lens.to('cpu').numpy().tolist(), predict_ids_list)
        # predict_ids_list.extend(pred_ids)
    orig_text_list = []
    orig_label_list = []
    with codecs.open(orig_test_file, 'r', 'utf-8') as fr:
        for i, line in enumerate(fr):
            if i % 2 == 0:
                text = line.strip("\n")
                text_split = text.split(" ")
                orig_text_list.append(text_split)
            else:
                text = line.strip("\n")
                text_split = text.split(" ")
                orig_label_list.append(text_split)
    gold_entity_list = gen_entity_from_label_id_list(orig_text_list, orig_label_list, id2label, True)
    pred_entity_list = gen_entity_from_label_id_list(orig_text_list, predict_ids_list, id2label)
    cal_mertric_from_two_list(pred_entity_list, gold_entity_list)

