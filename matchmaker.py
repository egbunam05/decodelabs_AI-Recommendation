import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

print("--- INITIALIZING DIGITAL MATCHMAKER ---\n")


print("Step 1: Ingesting User State...")
user_skills = ["Python", "Cloud", "Automation"]
print(f"User Profile (Input Array): {user_skills}\n")


print("Step 2: Processing & Scoring...")


csv_data = """Job Role,Skills
Data Scientist,Python SQL Machine Learning Data Analysis Statistics
DevOps Engineer,AWS Docker Kubernetes CI/CD Linux Automation Cloud
Backend Developer,Java Python SQL APIs Databases Microservices
Frontend Developer,JavaScript HTML CSS React UI Design
Cloud Architect,AWS Azure Cloud Computing Networking Security Automation
Machine Learning Engineer,Python Machine Learning TensorFlow PyTorch Deep Learning
Systems Administrator,Linux Networking Bash Shell Scripting Hardware Troubleshooting"""


df = pd.read_csv(io.StringIO(csv_data))


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Skills'])
user_vector = vectorizer.transform([" ".join(user_skills)])


similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
df['Match Score'] = similarity_scores


print("Step 3 & 4: Sorting and Filtering...\n")

sorted_df = df.sort_values(by='Match Score', ascending=False)
top_n = 3
top_recommendations = sorted_df.head(top_n)

print("--- 🏆 TOP 3 CAREER PATH RECOMMENDATIONS 🏆 ---")
for index, row in top_recommendations.iterrows():
    match_percentage = round(row['Match Score'] * 100, 2)
    print(f"\nRole: {row['Job Role']}")
    print(f"Match Score: {match_percentage}%")
    print(f"Required Skills: {row['Skills']}")

print("\n--- END OF PIPELINE ---")
