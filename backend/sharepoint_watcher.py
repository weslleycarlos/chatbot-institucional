import time
import os
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- CONFIGURAÇÃO ---
# Caminho da pasta sincronizada do OneDrive/SharePoint
PASTA_SHAREPOINT = r"C:\Users\Weslley\Polícia Federal\Intranet Polícia Federal - Boletins de Serviço"

# URL da sua API Local (GovBot)
API_URL = "http://localhost:8000/upload"

# Senha de admin definida no backend/main.py
ADMIN_USER = "admin"
ADMIN_PASS = "senha_secreta_123"

class SharePointHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        
        filename = event.src_path
        extension = filename.split('.')[-1].lower()
        
        if extension in ['pdf', 'docx']:
            print(f"📥 Novo arquivo detectado: {filename}")
            self.enviar_para_api(filename)

    def enviar_para_api(self, filepath):
        """Envia o arquivo detectado para a API do Chatbot"""
        try:
            # Espera um pouco para garantir que o OneDrive terminou de baixar
            time.sleep(2) 
            
            with open(filepath, 'rb') as f:
                files = {'file': f}
                # Usa Basic Auth para autenticar na API
                response = requests.post(
                    API_URL, 
                    files=files, 
                    auth=(ADMIN_USER, ADMIN_PASS)
                )
                
            if response.status_code == 200:
                print(f"✅ Arquivo indexado com sucesso!")
            else:
                print(f"❌ Erro ao indexar: {response.text}")
                
        except Exception as e:
            print(f"Erro de conexão: {e}")

if __name__ == "__main__":
    print(f"👀 Monitorando pasta: {PASTA_SHAREPOINT}")
    print("Mova arquivos para lá (via SharePoint ou Explorer) para indexar automaticamente.")
    
    event_handler = SharePointHandler()
    observer = Observer()
    observer.schedule(event_handler, PASTA_SHAREPOINT, recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Instale as dependências: pip install watchdog requests