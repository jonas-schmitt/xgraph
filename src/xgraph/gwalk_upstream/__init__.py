"""GWalk as released by the MQuAKE-Remastered authors, vendored for xgraph.gwalk.

Source: Shaochen Zhong, Yifan Lu, Lize Shao, Bhargav Bhushanam, Xiaocong Du, Yixin Wan,
Yucheng Shi, Daochen Zha, Yiwei Wang, Ninghao Liu, Kaixiong Zhou, Shuai Xu, Kai-Wei Chang,
Louis Feng, Vipin Chaudhary, Xia Hu: "MQuAKE-Remastered: Multi-Hop Knowledge Editing Can
Only Be Advanced with Reliable Evaluations", ICLR 2025; code at
https://github.com/henryzhongsc/MQuAKE-Remastered, commit
349dcc50460c251ee09a6804aab444508a016485 (2026-03-29). Licensed under the Creative Commons
Attribution 4.0 International License, https://creativecommons.org/licenses/by/4.0/. The
authors do not endorse this copy or its use.

Files and their upstream paths:

- prompts/extract_entity.txt, prompts/fill_out_ga_w_blank2.txt,
  prompts/relation2subq_prompt2.txt, prompts/subq_breakdown.txt: from
  data/mquake_remastered/prompts/, unchanged except that the repository stores every text
  file with LF line endings (.gitattributes): upstream, all but extract_entity.txt end
  their lines with CRLF. Upstream reads them in Python's text mode, which turns CRLF into
  LF, so the text the model sees is the same; tests/test_gwalk.py checks the files against
  upstream's SHA-256 with the CRLF endings put back.
- mquake_utils.py: from eval/mquake_remastered/mquake_utils.py.
- mquake_dataset.py: process_kg and check_answer from
  eval/mquake_remastered/mquake_dataset.py.
- eval_loop.py: the body of gwalk_eval_loop from pipeline/gwalk/eval_loop.py.
- inference_mquake.py: REL2SUBQ from pipeline/inference_mquake.py.

Changes (code lines that changed are marked "xgraph:" in the files):

1. call_model sends the request to vLLM's OpenAI-compatible completions endpoint through
   xgraph.gwalk.CaseModel instead of running Hugging Face model.generate; it returns the
   same text (prompt and completion, stripped). The stopping criteria become stop rules
   with the same effective stops on the served tokenizer (xgraph.gwalk.stop_rules), and
   each request gets its own seed (xgraph.gwalk.request_seed). The tokenizer argument the
   loop passes along is None, since call_model no longer uses it.
2. Functions GWalk does not use are left out: call_model_template, able_to_quit,
   call_model_batch, call_model_batch_template (mquake_utils.py); StoppingCriteriaSub,
   initialize_contriever and MODEL_to_SC (inference_mquake.py; xgraph.gwalk loads the
   contriever at a pinned revision); everything of MQuAKE_Remastered but process_kg and
   check_answer, which are functions of the dataset and rand_list instead of methods, and
   check_answer keeps no tally of correct answers.
3. gwalk_case is the body of gwalk_eval_loop's per-case loop; the setup before it (reading
   the prompts, the bank, the embeddings and aliases) and the dataset come from
   xgraph.gwalk, which reads the same dataset from a local JSON file and sets rand_list
   explicitly (upstream's Hugging Face split lists are empty in current revisions).
4. Every question of a case runs: upstream stops at the first question answered correctly
   against the gold answer, which only saves calls.
5. An IndexError from an unparsable subquestion reply (fetch_rel_subj2subq), which
   upstream lets end the whole run, fails only its hop, like the loop's own breaks on a
   malformed answer. (The subject extraction and breakdown parsers index blocks the
   prompt itself holds, so they cannot raise.)
6. gwalk_case returns a trace of each question (relations, hops, answer, the stage that
   failed, the final prompt; a breakdown naming no relation counts as failed) and tells
   the model which question its next calls belong to.
7. Requests ban the served model's thinking tags (<think>, </think>) with the strongest
   logit bias vLLM accepts (xgraph.gwalk.thinking_bias). Upstream's models have no such
   tags; a thinking model opens a thinking block in the middle of the few-shot text,
   which upstream's parsers then read as an answer or a relation.
8. A hop's answer call (the `facts` stop) also ends after its first complete line that
   the loop reads as a generated answer: one that starts with "Generated answer: " and
   holds ". " (xgraph.gwalk.answer_line_end). Upstream's models end that line and write
   "Retrieved fact:", the stop, which the loop cuts off, so for them nothing changes; a
   chat model instead goes on with a paragraph of commentary until the token limit, and
   the loop, which reads the hop's last line, takes the commentary for the answer or
   fails the hop. The call is cut on the returned text, since vLLM has no such stop, so
   its generated tokens count what the server generated past the line. A completion
   without such a line, or with only the stop after it, is left as upstream's.
9. gwalk_case takes mask (default True, upstream's behaviour); with mask=False the bank
   lookup gets an empty gold track, so another case's edit on the case's own gold path is
   not hidden. xgraph.gwalk --worlds uses it with per-world banks, where no such edit
   exists and the mask would only read the gold path.
10. gwalk_case takes repair (default None, upstream's behaviour); xgraph.gwalk
    --repair-answers passes xgraph.gwalk.repair_answer. After a hop's answer call, when
    the line the loop reads is no generated answer (it does not start with
    "Generated answer: " and hold ". "), repair may replace the text with the completion
    cut after its first line in the few-shot's form "<sentence> {}. <answer>" that comes
    before any line taking up another (sub)question, written as "Generated answer: " and
    that line; the hop is marked repaired. A chat model writes such a line below a bare
    "Generated answer:", in markdown bold, or above commentary, where upstream's loop
    fails the hop or reads the commentary as the answer. Nothing is guessed: a
    completion without such a line is left as it is.
11. gwalk_case takes complete (default None, upstream's behaviour); xgraph.gwalk
    --complete-answers passes xgraph.gwalk.complete_answer. After a hop's answer call
    (and repair, if given), when the line the loop reads is no generated answer and the
    completion's first line is "Generated answer: " and the model's template sentence
    with its slot left empty ("<sentence> {}."), the loop makes one more call of the
    case's model: the answer call's prompt continued with "Generated answer: <sentence>
    {}. ", the few-shot's form, with the answer call's token limit and temperature,
    ending at a newline (xgraph.gwalk.stop_rules' `line` rule; its calls count as stage
    "complete"). The completed line replaces the completion, and the hop is marked
    completed. A chat model ends the sentence at the slot where upstream's models write
    the object after it. A bare "Generated answer:" is not completed.
12. gwalk_case takes kg_fact (default None, upstream's behaviour); xgraph.gwalk --kg-db
    passes the lookup of the case's KG store (xgraph.gwalk.KGLookup.fact). After the bank
    lookup (get_fact_form_kg), when the bank gives no fact (it has none, or upstream's
    mask hides it) and the hop's relation template was retrieved, kg_fact(subject,
    template, id2rel) looks the hop up in the store; a fact it finds is taken as a bank
    fact is (the retrieved fact, "contradicts", its object as the next subject), and the
    hop records the triple's ids as kg_fact. Without a fact, the hop is upstream's.
    The bank lookup and this one are a local function of gwalk_case (lookup), and the
    gold track the mask reads is computed once per case, not once per hop: it depends on
    the case alone.
13. gwalk_case takes lookup_first (default False, upstream's behaviour); xgraph.gwalk
    --hop-order lookup-first passes True. Each hop is then looked up (the bank, then
    kg_fact) right after its subquestion, before the answer call: a hop with a fact takes
    the fact's object, fitted as upstream fits it, without calling the model, and the
    prompt gets "Retrieved fact: <fact>." and "Retrieved fact answers the subquestion, so
    continue with this subject: <object>."; the hop is marked looked_up_first. A hop
    without a fact makes upstream's answer call, and the lookup is not repeated after it.
14. break_down_into_subquestions breaks down every question of the case, not the first
    three: upstream's cases have exactly three, and xgraph.gwalk --conditions asks cases
    of one question. gwalk_case takes subquestions (default None, upstream's REL2SUBQ), a
    relation template's subquestion, which xgraph.gwalk's open catalogue (--relations)
    passes; and catalogue (default None, the bank's templates and embeddings), a function
    of the hop's subject giving the templates and embeddings its relation is retrieved
    among, which xgraph.gwalk --relations-of-subject passes
    (xgraph.gwalk.subject_catalogue); a hop given no template is retrieved from none, as
    when the contriever's best score falls below the threshold.
"""
