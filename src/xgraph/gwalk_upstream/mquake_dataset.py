# Vendored from github.com/henryzhongsc/MQuAKE-Remastered at commit
# 349dcc50460c251ee09a6804aab444508a016485, eval/mquake_remastered/mquake_dataset.py,
# under CC BY 4.0 (see __init__.py for the attribution and the list of changes).
# process_kg and check_answer, methods of MQuAKE_Remastered upstream, are functions here:
# self.dataset and self.rand_list are arguments, and check_answer keeps no tally.


def process_kg(dataset, rand_list):
    kg_s_r_o = {}

    rels = set()
    ents = set()

    id2rel = {}
    rel2id = {}

    for d in dataset:
        if d['case_id'] not in rand_list:
            continue
        caseid = d['case_id']
        for edit in d["requested_rewrite"]:
            s = edit['subject']
            o = edit['target_new_str']
            o_pre = edit['target_true_str']

            rel_id = edit['relation_id']
            r = edit['prompt']

            if r not in rel2id.keys():
                rel2id[r] = rel_id
                id2rel[rel_id] = r
            else:
                assert rel2id[r] == rel_id

            rels.add(r)
            ents.add(s)
            ents.add(o)
            ents.add(o_pre)

            if s in kg_s_r_o.keys():
                if r in kg_s_r_o[s].keys():
                    obj = kg_s_r_o[s][r][0]
                    id_set = kg_s_r_o[s][r][1]
                    id_set.add(caseid)
                    kg_s_r_o[s][r] = [obj, id_set]
                else:
                    kg_s_r_o[s][r] = [o, set([caseid])]
            else:
                kg_s_r_o[s] = {r: [o, set([caseid])]}

    return kg_s_r_o, list(rels), list(ents), id2rel


def check_answer(edit_flag, instance, ans):
    if ans is None:
        ans = "no answer is provided. This is a failed case."
    # Define answer and answer_alias keys based on edit_flag
    answer = "answer"
    answer_alias = "answer_alias"
    if edit_flag:
        answer = "new_" + answer
        answer_alias = "new_" + answer_alias

    # Convert the answer and ans to upper case
    ans_upper = ans.upper()
    instance_answer_upper = instance[answer].upper()

    # Convert each alias to upper case for comparison
    instance_answer_alias_upper = [alias.upper() for alias in instance[answer_alias]]

    if ans_upper == instance_answer_upper or ans_upper in instance_answer_alias_upper:
        return True

    return False
