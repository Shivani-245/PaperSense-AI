
import json


def init_db():
    # Tables are already created in Supabase.
    pass


def save_paper(supabase, user_id, file_name, total_pages, extracted_text):
    response = supabase.table("papers").insert({
        "user_id": user_id,
        "file_name": file_name,
        "total_pages": total_pages,
        "extracted_text": extracted_text
    }).execute()

    return response.data[0]["id"]


def save_analysis(
    supabase,
    paper_id,
    basic_summary="",
    ai_summary="",
    abstract="",
    keywords=None
):
    supabase.table("analysis_results").upsert({
        "paper_id": paper_id,
        "basic_summary": basic_summary,
        "ai_summary": ai_summary,
        "abstract": abstract,
        "keywords": json.dumps(keywords or [])
    }, on_conflict="paper_id").execute()


def save_question(supabase, paper_id, question, answer):
    supabase.table("qa_history").insert({
        "paper_id": paper_id,
        "question": question,
        "answer": answer
    }).execute()


def get_recent_papers(supabase, user_id):
    response = (
        supabase.table("papers")
        .select("id, file_name, total_pages, uploaded_at")
        .eq("user_id", user_id)
        .order("uploaded_at", desc=True)
        .execute()
    )

    return [
        (p["id"], p["file_name"], p["total_pages"], p["uploaded_at"])
        for p in response.data
    ]


def get_paper_details(supabase, paper_id, user_id):
    paper_response = (
        supabase.table("papers")
        .select("*")
        .eq("id", paper_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )

    if not paper_response.data:
        return None

    paper = paper_response.data[0]

    analysis_response = (
        supabase.table("analysis_results")
        .select("*")
        .eq("paper_id", paper_id)
        .limit(1)
        .execute()
    )

    analysis = analysis_response.data[0] if analysis_response.data else {}

    return (
        paper["id"],
        paper["file_name"],
        paper["total_pages"],
        paper["extracted_text"],
        paper["uploaded_at"],
        analysis.get("basic_summary"),
        analysis.get("ai_summary"),
        analysis.get("abstract"),
        analysis.get("keywords")
    )


def get_question_history(supabase, paper_id):
    response = (
        supabase.table("qa_history")
        .select("question, answer, created_at")
        .eq("paper_id", paper_id)
        .order("created_at", desc=True)
        .execute()
    )

    return [
        (q["question"], q["answer"], q["created_at"])
        for q in response.data
    ]
