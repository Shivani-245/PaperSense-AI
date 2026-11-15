
import sqlite3

conn = sqlite3.connect("papersense.db")
cursor = conn.cursor()

print("\n===== PAPERSENSE DATABASE VERIFICATION =====")

print("Papers:", cursor.execute(
    "SELECT COUNT(*) FROM papers"
).fetchone()[0])

print("Analysis:", cursor.execute(
    "SELECT COUNT(*) FROM analysis_results"
).fetchone()[0])

print("AI Summaries:", cursor.execute(
    "SELECT COUNT(*) FROM analysis_results WHERE ai_summary IS NOT NULL AND ai_summary != ''"
).fetchone()[0])

print("Questions:", cursor.execute(
    "SELECT COUNT(*) FROM qa_history"
).fetchone()[0])

print("\n===== UPLOADED PAPERS =====")

for paper in cursor.execute(
    "SELECT id, file_name, total_pages FROM papers"
).fetchall():
    print(paper)

conn.close()
print("\nDatabase verification completed.")
