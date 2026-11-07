# Vendored from github.com/henryzhongsc/MQuAKE-Remastered at commit
# 349dcc50460c251ee09a6804aab444508a016485, eval/mquake_remastered/mquake_utils.py,
# under CC BY 4.0 (see __init__.py for the attribution and the list of changes). Changed
# lines are marked "xgraph:"; functions GWalk does not use are left out.

import logging
logger = logging.getLogger("main")

import torch
from tqdm import tqdm


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def get_sent_embeddings(sents, contriever, tok, BSZ=32):
    all_embs = []
    for i in tqdm(range(0, len(sents), BSZ), disable=True):
        sent_batch = sents[i:i + BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs


def retrieve_facts(query, fact_embs, contriever, tok, k=1, threshold=float('-inf')):
    inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to("cuda")
    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
    sim = (query_emb @ fact_embs.T)[0]
    knn = sim.topk(k, largest=True)

    if knn.values[0] < threshold:
        return None

    return knn.indices


def get_relation(subquestion, rels, rel_emb, contriever, tokenizer):
    rel_idx = retrieve_facts(subquestion, rel_emb, contriever, tokenizer, threshold=0.845)
    if rel_idx is None:
        return None
    rel = rels[rel_idx[0]]
    return rel


def call_model(prompt, stop, model, tokenzier, device='cuda', generate_length=50, temperature=1.0):
    # xgraph: upstream runs Hugging Face model.generate (do_sample=True, the stopping
    # criteria `stop`) and returns the decoded prompt and completion, stripped. `model` is
    # xgraph.gwalk.CaseModel, which sends the same request to vLLM's completions endpoint
    # and returns the same text; `stop` names a stop rule there.
    return model.generate(prompt, stop, generate_length, temperature)


def remove_extra_target_occurrences(gen, target, count):
    occurrences = gen.count(target)

    if occurrences <= count:
        return gen

    index = 0
    for _ in range(count + 1):
        index = gen.find(target, index) + len(target)

    return gen[:index - len(target) - 2]


def extract_entities(input_string):
    segments = input_string.split('->')

    entities = []

    for segment in segments:
        if '(' in segment and ')' in segment:
            entity = segment.strip()[1:-1]
            entities.append(entity)

    return entities


def get_ent_alias(dataset):
    ent2alias = {}
    for idx, d in enumerate(dataset):
        answer = d['answer']
        answer_alias = d['answer_alias']
        ent2alias[answer] = set(answer_alias)

        answer = d['new_answer']
        answer_alias = d['new_answer_alias']
        ent2alias[answer] = set(answer_alias)

        for hop in d['single_hops']:
            answer = hop['answer']
            answer_alias = hop['answer_alias']
            ent2alias[answer] = set(answer_alias)

        for hop in d['new_single_hops']:
            answer = hop['answer']
            answer_alias = hop['answer_alias']
            ent2alias[answer] = set(answer_alias)

    return ent2alias


def get_fact_form_kg(subject, rel, kg_s_r_o, caseid, track):
    if subject in kg_s_r_o.keys():
        if rel in kg_s_r_o[subject].keys():
            fact_object, caseids = list(kg_s_r_o[subject][rel])

            if True and caseid not in caseids and any(t == (subject, rel) for t in track):
                return "<no fact>", False, None

            fact = f'{rel.format(subject)} {fact_object}'
            return fact, True, fact_object

    return "<no fact>", False, None


def get_correct_track(d, edited, id2rel):
    tracks = []

    triples_name = 'new_triples' if edited else 'orig_triples'
    triples = d[triples_name]
    triples_labeled = d[triples_name + '_labeled']

    for i in range(len(triples)):
        tracks.append((triples_labeled[i][0], id2rel[triples[i][1]]))

    return tracks


def fit_subject_on_kg(subject, ent_emb, contriever, tokenizer, ents, kg_s_r_o, ent2alias):
    if subject in set(ents):
        return subject
    if len(ents) == 0:
        return subject
    indices = retrieve_facts(subject, ent_emb, contriever, tokenizer, k=min(10, len(ents)))
    for idx in indices:
        target = ents[idx]
        if target in ent2alias.keys() and subject in ent2alias[target]:
            return target
    return subject


def fetch_rel_subj2subq(subject, rel, relation2subq_prompt, sc_end_block, model, model_tokenizer, device):
    prompt = relation2subq_prompt + f"Given this relation: \"{rel}\" and this subject: \"{subject}\",\nThe corresponding question is"

    output = call_model(prompt, sc_end_block, model, model_tokenizer, temperature=0.2, generate_length=20, device=device)
    output = output.strip().split('\n\n')[5]
    output = output.strip().split('\n')[1]
    output = output.strip().split('\"')[1]

    return output


def extract_entity(questions, extract_entity_prompt, sc_end_block, model, model_tokenizer, device):
    for question in questions:
        task_prompt = extract_entity_prompt + f"Example 5:\nQuestion: \"{question}\"\nExtracted entity: ("
        output = call_model(task_prompt, sc_end_block, model, model_tokenizer, temperature=0.2, generate_length=20, device=device)

        output = output.strip().split('\n\n')[4]
        output = output.strip().split('\n')[-1][len("Extracted entity: ("):]
        entity = output[:-1].strip()

        if entity in question:
            entity_start_idx = question.find(entity)
            entity_end_idx = entity_start_idx + len(entity) - 1

            while entity_end_idx + 1 < len(question):
                next_char = question[entity_end_idx + 1]

                if next_char == '.' and entity_end_idx + 2 < len(question):
                    next_space_idx = question.find(" ", entity_end_idx + 3)
                    if next_space_idx == -1:
                        entity_end_idx = len(question) - 1
                    else:
                        entity_end_idx = next_space_idx - 1
                else:
                    break

            full_entity = question[entity_start_idx: entity_end_idx + 1].split("'")[0].replace('?', '')
            if full_entity in question:
                return full_entity

    return entity


def break_down_into_subquestions(d, subject, breakdown_prompt, sc_done, tokenizer, model, use_template=False, front_space=0):
    retval = []

    prompts = []
    for i in range(len(d['questions'])):  # xgraph: upstream's cases have 3
        prompt = breakdown_prompt + f"Given this problem:\n{d['questions'][i]}\nExtract relations in square parentheses into follows:\n\"{subject}->"
        prompts.append(prompt)

    res = []
    for i in range(len(d['questions'])):  # xgraph
        res.append(call_model(prompts[i], sc_done, model, tokenizer, temperature=0.2))

    for i in range(len(d['questions'])):  # xgraph
        temp = res[i].split("\n\n")[4]
        temp = temp.strip().split("\n")[-1]
        rels = extract_entities(temp)
        retval.append(rels)
    return retval
