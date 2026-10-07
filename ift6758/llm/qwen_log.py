def create_log(log_dir="runs"):
    """Fonction qui génère un nouveau fichier pour enregistrer les logs"""
    os.makedirs("logs", exist_ok=True)
def rag_log(doc,type,lenght,prompt_ID,task_ID):
    """Fonction qui est responsable de prendre en note les appels de RAG pour pouvoir analyser le résultat du RAG, laa tâche assoscié et sa provenance"""

def log_call(log_path,messages,result):
