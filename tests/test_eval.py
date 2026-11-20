import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_score_eval_metrics(tmp_path):
    csv = tmp_path / "set.csv"
    csv.write_text(
        "id,batch,category,question,expected_answer,expected_doc_id,expected_article,answerable\n"
        "q1,b01,faktik,S1,A,lex:100,5-modda,true\n"
        "q2,b01,kop,S2,A,lex:100;lex:200,7-modda;57¹-modda,true\n"
        "q3,b01,javobsiz,S3,,,,false\n",
        encoding="utf-8",
    )
    run = tmp_path / "run"
    run.mkdir()
    res = {
        "results": [
            {"id": "q1", "status": "ok", "no_answer": False, "timing": {"total_ms": 2000},
             "citations": [{"chunk_id": "c2", "doc_id": "lex:100", "article": "5-modda", "doc_status": "amalda"}],
             "contexts": [{"chunk_id": "c1", "doc_id": "lex:100", "article": "4-modda"}, {"chunk_id": "c2", "doc_id": "lex:100", "article": "5-modda"}]},
            {"id": "q2", "status": "ok", "no_answer": False, "timing": {"total_ms": 4000},
             "citations": [{"chunk_id": "x", "doc_id": "lex:200", "article": "57¹-modda"}],
             "contexts": [{"chunk_id": "c3", "doc_id": "lex:200", "article": "57¹-modda"}]},
            {"id": "q3", "status": "no_answer", "no_answer": True, "timing": {"total_ms": 1000}, "citations": [], "contexts": []},
        ]
    }
    (run / "b01.json").write_text(json.dumps(res), encoding="utf-8")
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "score_eval.py"), "--csv", str(csv), "--run", str(run)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    s = json.loads((run / "score" / "summary.json").read_text("utf-8"))
    assert s["hit@5_doc"] == 1.0
    assert s["hit@5_article"] == 1.0
    assert s["mrr"] == 0.75  # (1/2 + 1/1) / 2
    assert s["recall@5"] == 0.75  # (1 + 1/2) / 2
    assert s["no_answer_accuracy"] == "1/1"
    assert s["citations_not_in_contexts"] == 1
    assert s["latency_p50_s"] == 2.0
