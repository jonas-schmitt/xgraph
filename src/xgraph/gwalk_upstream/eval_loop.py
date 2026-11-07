# Vendored from github.com/henryzhongsc/MQuAKE-Remastered at commit
# 349dcc50460c251ee09a6804aab444508a016485, pipeline/gwalk/eval_loop.py, under CC BY 4.0
# (see __init__.py for the attribution and the list of changes). gwalk_case is the body of
# upstream's gwalk_eval_loop for one case `d`; the setup before that loop (prompts, bank,
# embeddings, aliases) is xgraph.gwalk's, and arrives as arguments. Added and changed lines
# of the loop are marked "xgraph:"; the model tokenizer arguments are None, since
# call_model no longer uses them.

from . import mquake_utils as utils
from .inference_mquake import REL2SUBQ as rel2subq


def gwalk_case(d, edit_flag, prompts, model, contriever, contriever_tokenizer, kg_s_r_o, rels, ents,
               id2rel, rel_emb, ent_emb, ent2alias, stopping_criteria_dict, device='cuda', mask=True,
               repair=None, complete=None, kg_fact=None, lookup_first=False, subquestions=None,
               catalogue=None):
    """Upstream's loop body for one case, with a trace of every question (xgraph).
    mask=False (xgraph) drops upstream's per-case mask of other cases' edits on the gold path.
    repair (xgraph), when given, is xgraph.gwalk.repair_answer: it may replace the text the
    loop reads after a hop's answer call. complete (xgraph), when given, is
    xgraph.gwalk.complete_answer: it may complete the answer line's empty slot with one more
    call of the model. kg_fact (xgraph), when given, is the lookup method of the bank's
    xgraph.gwalk.KGLookup: a hop the bank has no fact for is looked up in a KG store.
    lookup_first (xgraph): look each hop up in the bank and the KG store before the answer
    call, and ask the model only when neither has a fact. subquestions (xgraph): each
    relation template's subquestion, in place of upstream's REL2SUBQ. catalogue (xgraph),
    when given, is xgraph.gwalk.subject_catalogue for the bank: the relation templates, and
    their embeddings, a hop's relation is retrieved among for its subject."""
    task_prompt = prompts['fill_out_ga_w_blank2']
    breakdown_prompt = prompts['subq_breakdown']
    relation2subq_prompt = prompts['relation2subq_prompt2']
    extract_entity_prompt = prompts['extract_entity']

    start_subject = utils.extract_entity(d["questions"], extract_entity_prompt, stopping_criteria_dict['done'], model, None, device=device)
    breakdown_rels_list = utils.break_down_into_subquestions(d, start_subject, breakdown_prompt, stopping_criteria_dict['done'], None, model)
    trace = {'subject': start_subject, 'questions': []}  # xgraph
    track = utils.get_correct_track(d, edit_flag, id2rel) if mask else []  # xgraph: mask, once

    def lookup(subject, rel, hop):  # xgraph: the bank's fact, else the KG store's (kg_fact)
        fact_sent, contra_or_not, fact_object = utils.get_fact_form_kg(subject, rel, kg_s_r_o, d['case_id'], track)
        if not contra_or_not and kg_fact is not None and rel is not None:  # xgraph: opt-in
            found = kg_fact(subject, rel, id2rel)  # xgraph: the KG store's fact, if any
            if found is not None:  # xgraph
                fact_sent, fact_object, hop['kg_fact'] = found  # xgraph
                contra_or_not = True  # xgraph: it replaces the answer as a bank fact does
        return fact_sent, contra_or_not, fact_object

    llm_answers = []
    for q_id, q in enumerate(d["questions"]):
        model.question = q_id  # xgraph: the calls that follow belong to this question
        subject = str(start_subject)
        breakdown_rels = breakdown_rels_list[q_id]
        hops = []  # xgraph
        error = None if breakdown_rels else 'breakdown'  # xgraph: no relation parsed

        ans = None
        prompt = task_prompt + "\n\nQuestion: " + q + "\n"
        for i in range(len(breakdown_rels)):
            subject = utils.fit_subject_on_kg(subject, ent_emb, contriever, contriever_tokenizer, ents, kg_s_r_o, ent2alias)

            relation = breakdown_rels[i]
            hop_rels, hop_emb = (rels, rel_emb) if catalogue is None else catalogue(subject)  # xgraph
            rel = utils.get_relation(relation, hop_rels, hop_emb, contriever, contriever_tokenizer) if hop_rels else None  # xgraph
            hop = {'relation': relation, 'template': rel, 'subject': subject}  # xgraph
            hops.append(hop)  # xgraph
            if rel is not None:
                subquestion = (rel2subq if subquestions is None else subquestions).get(rel, rel).format(subject)  # xgraph
            else:
                # xgraph: upstream lets an IndexError from an unparsable reply end the
                # run; here it fails the hop, like the loop's own breaks below.
                try:
                    subquestion = utils.fetch_rel_subj2subq(subject, relation, relation2subq_prompt,
                                                            stopping_criteria_dict['end_block'],
                                                            model=model,
                                                            model_tokenizer=None, device=device)
                except IndexError:
                    hop['failed'] = error = 'rel2subq'
                    break
            hop['subquestion'] = subquestion  # xgraph
            prompt = prompt + "Subquestion: " + subquestion + "\n"

            looked = lookup(subject, rel, hop) if lookup_first else None  # xgraph: opt-in
            if looked is not None and looked[1]:  # xgraph: a fact decides the hop, no call
                fact_sent, _, fact_object = looked  # xgraph
                subject = utils.fit_subject_on_kg(fact_object, ent_emb, contriever, contriever_tokenizer, ents, kg_s_r_o, ent2alias)  # xgraph
                ans = subject  # xgraph
                prompt = (prompt + 'Retrieved fact: ' + fact_sent + '.\nRetrieved fact answers the subquestion, '
                          'so continue with this subject: ' + fact_object + '.\n')  # xgraph
                hop.update(retrieved_fact=fact_sent, contradicted=True, subject_after=subject, looked_up_first=True)  # xgraph
                continue  # xgraph

            asked = prompt  # xgraph: the answer call's prompt, for repair
            prompt = utils.call_model(prompt, stopping_criteria_dict['facts'], model, None)

            if prompt.strip().split('\n')[-1] == 'Retrieved fact:':
                prompt = prompt[:-len('\nRetrieved fact:')]
            prompt = utils.remove_extra_target_occurrences(prompt, "Question: ", 5)
            repaired = repair(asked, prompt) if repair else None  # xgraph: opt-in repair
            if repaired is not None:  # xgraph
                prompt, hop['repaired'] = repaired, True  # xgraph
            completed = complete(asked, prompt, model) if complete else None  # xgraph: opt-in
            if completed is not None:  # xgraph
                prompt, hop['completed'] = completed, True  # xgraph

            temp_split = prompt.strip().split('\n')
            if len(temp_split) < 2:
                hop['failed'] = error = 'answer'  # xgraph
                break  # failed case

            generated_answer = temp_split[-1][len("Generated answer: "):]
            hop['generated'] = generated_answer  # xgraph

            ga_seg = generated_answer.strip().split('. ')

            if len(ga_seg) >= 2:
                answer_object = ". ".join(ga_seg[1:])
            else:
                hop['failed'] = error = 'answer'  # xgraph
                break
            fact_sent, contra_or_not, fact_object = looked or lookup(subject, rel, hop)  # xgraph

            contra_promt = "Retrieved fact {} to generated answer, so continue with this subject: {}.\n"
            if contra_or_not:
                does_or_doesnot = "contradicts"
                inter_answer = fact_object
            else:
                does_or_doesnot = "does not contradict"
                inter_answer = answer_object

            contra_promt = contra_promt.format(does_or_doesnot, inter_answer)

            subject = utils.fit_subject_on_kg(inter_answer, ent_emb, contriever, contriever_tokenizer, ents, kg_s_r_o, ent2alias)

            ans = subject
            prompt = prompt + '\nRetrieved fact: ' + fact_sent + '.\n' + contra_promt
            hop.update(retrieved_fact=fact_sent, contradicted=contra_or_not, subject_after=subject)  # xgraph

        llm_answers.append(ans)
        # xgraph: upstream stops at the first question answered correctly (checked against
        # the gold answer); every question runs here, for per-question records.
        trace['questions'].append({'relations': breakdown_rels_list[q_id], 'hops': hops, 'answer': ans,
                                   'error': error, 'prompt': prompt})

    return trace
