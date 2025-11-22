import time
import os
import requests
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
from pathlib import Path
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO ---
# Caminho da pasta sincronizada do OneDrive/SharePoint
PASTA_SHAREPOINT = r"C:\Users\Weslley\Polícia Federal\Intranet Polícia Federal - Boletins de Serviço"

# URL da sua API Local (GovBot)
API_URL = "http://127.0.0.1:8000/upload"

# Senha de admin definida no backend/main.py
ADMIN_USER = "admin"
ADMIN_PASS = "senha_secreta_123"

# Configurações de processamento
DEBOUNCE_SEGUNDOS = 10  # Tempo mínimo entre processamentos do mesmo arquivo
TIMEOUT_UPLOAD = 120    # Timeout para upload de arquivos grandes (segundos)
MAX_RETRIES = 3         # Número máximo de tentativas de upload

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sharepoint_watcher.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Extensões permitidas
EXTENSOES_PERMITIDAS = {'pdf', 'docx'}

# Controle de processamento
processando_arquivos = set()
ultimo_processamento = {}  # {filepath_str: datetime}
lock = threading.Lock()    # Thread safety


class SharePointHandler(FileSystemEventHandler):
    
    def _deve_ignorar_arquivo(self, filepath):
        """Verifica se o arquivo deve ser ignorado"""
        nome = filepath.name
        
        # Arquivos temporários do OneDrive/Office
        if nome.startswith('~') or nome.startswith('.~'):
            return True
        
        # Arquivos temporários do Windows
        if nome.startswith('~$'):
            return True
            
        # Arquivos ocultos
        if nome.startswith('.'):
            return True
            
        # Arquivos .tmp
        if nome.endswith('.tmp'):
            return True
            
        return False
    
    def _debounce_ok(self, filepath):
        """Verifica se passou tempo suficiente desde o último processamento"""
        filepath_str = str(filepath)
        agora = datetime.now()
        
        with lock:
            if filepath_str in ultimo_processamento:
                tempo_desde_ultimo = agora - ultimo_processamento[filepath_str]
                if tempo_desde_ultimo < timedelta(seconds=DEBOUNCE_SEGUNDOS):
                    logger.debug(f"⏳ Debounce ativo para: {filepath.name}")
                    return False
            
            ultimo_processamento[filepath_str] = agora
        
        return True

    def on_created(self, event):
        """Chamado quando um novo arquivo é criado"""
        if event.is_directory:
            return
        
        filepath = Path(event.src_path)
        extension = filepath.suffix.lower()[1:]
        
        if extension not in EXTENSOES_PERMITIDAS:
            return
            
        if self._deve_ignorar_arquivo(filepath):
            return
        
        # Aguarda um pouco para o arquivo ser completamente criado
        time.sleep(2)
        
        logger.info(f"📥 Novo arquivo detectado: {filepath.name}")
        self.processar_arquivo(filepath)

    def on_modified(self, event):
        """Chamado quando um arquivo é modificado"""
        if event.is_directory:
            return
            
        filepath = Path(event.src_path)
        extension = filepath.suffix.lower()[1:]
        
        if extension not in EXTENSOES_PERMITIDAS:
            return
            
        if self._deve_ignorar_arquivo(filepath):
            return
        
        # Verifica se já está processando
        with lock:
            if filepath in processando_arquivos:
                return
        
        # Verifica debounce
        if not self._debounce_ok(filepath):
            return
        
        # Aguarda para evitar processar durante download
        time.sleep(3)
        
        logger.info(f"📝 Arquivo modificado detectado: {filepath.name}")
        self.processar_arquivo(filepath)

    def processar_arquivo(self, filepath):
        """Processa o arquivo e envia para a API"""
        # Evita processamento duplicado com thread safety
        with lock:
            if filepath in processando_arquivos:
                return
            processando_arquivos.add(filepath)
        
        try:
            # Verifica se o arquivo ainda existe
            if not filepath.exists():
                logger.warning(f"⚠️ Arquivo não existe mais: {filepath.name}")
                return
            
            # Verifica se o arquivo está completamente baixado
            if not self.arquivo_pronto(filepath):
                logger.warning(f"⏳ Arquivo ainda não está pronto: {filepath.name}")
                return
            
            # Verifica tamanho mínimo
            tamanho = filepath.stat().st_size
            if tamanho == 0:
                logger.warning(f"⚠️ Arquivo vazio ignorado: {filepath.name}")
                return
            
            logger.info(f"📤 Enviando {filepath.name} ({tamanho / 1024:.1f} KB)...")
                
            # Envia para a API com retry
            sucesso = self.enviar_com_retry(filepath)
            
            if sucesso:
                logger.info(f"✅ Arquivo indexado com sucesso: {filepath.name}")
            else:
                logger.error(f"❌ Falha ao indexar após {MAX_RETRIES} tentativas: {filepath.name}")
                
        except FileNotFoundError:
            logger.warning(f"⚠️ Arquivo foi removido durante processamento: {filepath.name}")
        except Exception as e:
            logger.error(f"💥 Erro ao processar {filepath.name}: {str(e)}")
        finally:
            with lock:
                processando_arquivos.discard(filepath)

    def arquivo_pronto(self, filepath, timeout=30):
        """Verifica se o arquivo está completamente baixado"""
        for i in range(timeout):
            try:
                # Verifica se o arquivo existe
                if not filepath.exists():
                    return False
                
                # Tenta abrir o arquivo em modo exclusivo
                with open(filepath, 'rb') as f:
                    # Lê um pequeno trecho para garantir acesso
                    f.read(1024)
                
                # Verifica se o tamanho está estável
                tamanho_atual = filepath.stat().st_size
                time.sleep(1)
                
                # Verifica novamente se existe antes de pegar o tamanho
                if not filepath.exists():
                    return False
                    
                tamanho_novo = filepath.stat().st_size
                
                if tamanho_atual == tamanho_novo and tamanho_atual > 0:
                    return True
                    
            except (IOError, OSError, PermissionError) as e:
                logger.debug(f"Arquivo em uso, aguardando... ({i+1}/{timeout})")
                time.sleep(1)
                continue
            except FileNotFoundError:
                return False
                
        return False

    def enviar_com_retry(self, filepath):
        """Envia o arquivo para a API com tentativas de retry"""
        for tentativa in range(1, MAX_RETRIES + 1):
            try:
                sucesso = self.enviar_para_api(filepath)
                if sucesso:
                    return True
                    
                if tentativa < MAX_RETRIES:
                    wait_time = tentativa * 5  # Backoff progressivo
                    logger.warning(f"🔄 Tentativa {tentativa} falhou. Aguardando {wait_time}s...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                logger.error(f"💥 Erro na tentativa {tentativa}: {str(e)}")
                if tentativa < MAX_RETRIES:
                    time.sleep(tentativa * 5)
                    
        return False

    def enviar_para_api(self, filepath):
        """Envia o arquivo para a API do GovBot"""
        try:
            # Verifica se arquivo ainda existe
            if not filepath.exists():
                logger.error(f"❌ Arquivo não existe: {filepath.name}")
                return False
            
            with open(filepath, 'rb') as arquivo:
                # Determina o content-type correto
                ext = filepath.suffix.lower()[1:]
                if ext == 'pdf':
                    content_type = 'application/pdf'
                elif ext == 'docx':
                    content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                else:
                    content_type = 'application/octet-stream'
                
                files = {'file': (filepath.name, arquivo, content_type)}
                
                response = requests.post(
                    API_URL, 
                    files=files, 
                    auth=(ADMIN_USER, ADMIN_PASS),
                    timeout=TIMEOUT_UPLOAD
                )
                
            if response.status_code == 200:
                resultado = response.json()
                chunks = resultado.get('chunks', 0)
                status = resultado.get('status', 'desconhecido')
                
                if status == 'sucesso':
                    logger.info(f"📊 Resultado: {chunks} chunks processados")
                    return True
                else:
                    logger.error(f"🔴 API retornou erro: {resultado.get('detalhes', 'Erro desconhecido')}")
                    return False
                    
            elif response.status_code == 401:
                logger.error("🔐 Erro de autenticação: Verifique usuário e senha")
                return False
            else:
                logger.error(f"🔴 API retornou status {response.status_code}: {response.text[:200]}")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.error("🔌 Erro de conexão: API do GovBot não está respondendo")
            return False
        except requests.exceptions.Timeout:
            logger.error(f"⏰ Timeout: Upload demorou mais de {TIMEOUT_UPLOAD}s")
            return False
        except FileNotFoundError:
            logger.error(f"❌ Arquivo foi removido: {filepath.name}")
            return False
        except Exception as e:
            logger.error(f"💥 Erro inesperado ao enviar arquivo: {str(e)}")
            return False


def verificar_pasta_sharepoint():
    """Verifica se a pasta do SharePoint existe e está acessível"""
    pasta = Path(PASTA_SHAREPOINT)
    
    if not pasta.exists():
        logger.error(f"❌ Pasta não encontrada: {PASTA_SHAREPOINT}")
        logger.info("💡 Dica: Verifique se o OneDrive está sincronizado")
        return False
    
    if not pasta.is_dir():
        logger.error(f"❌ Caminho não é uma pasta: {PASTA_SHAREPOINT}")
        return False
    
    # Conta arquivos válidos
    arquivos_validos = [
        f for f in pasta.rglob('*') 
        if f.is_file() and f.suffix.lower()[1:] in EXTENSOES_PERMITIDAS
    ]
    
    logger.info(f"📁 Pasta monitorada: {pasta}")
    logger.info(f"📊 Arquivos PDF/DOCX encontrados: {len(arquivos_validos)}")
    return True


def verificar_conexao_api():
    """Verifica se a API está respondendo"""
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            versao = data.get('version', 'desconhecida')
            logger.info(f"✅ API do GovBot online (versão: {versao})")
            return True
        else:
            logger.warning(f"⚠️ API retornou status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("❌ API não está rodando. Execute: python main.py")
        return False
    except Exception as e:
        logger.error(f"❌ Erro ao conectar com a API: {str(e)}")
        return False


def processar_arquivos_existentes():
    """Processa arquivos que já existem na pasta (opcional)"""
    pasta = Path(PASTA_SHAREPOINT)
    arquivos = [
        f for f in pasta.rglob('*') 
        if f.is_file() and f.suffix.lower()[1:] in EXTENSOES_PERMITIDAS
    ]
    
    if not arquivos:
        return
        
    logger.info(f"📂 Encontrados {len(arquivos)} arquivos existentes")
    resposta = input("Deseja processar arquivos existentes? (s/N): ").strip().lower()
    
    if resposta == 's':
        handler = SharePointHandler()
        for arquivo in arquivos:
            logger.info(f"📤 Processando: {arquivo.name}")
            handler.processar_arquivo(arquivo)
            time.sleep(2)  # Evita sobrecarregar a API


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 SharePoint Watcher - GovBot")
    logger.info("=" * 60)
    
    # Verificações iniciais
    if not verificar_pasta_sharepoint():
        logger.error("❌ Não foi possível acessar a pasta do SharePoint.")
        input("Pressione Enter para sair...")
        exit(1)
    
    api_online = verificar_conexao_api()
    if not api_online:
        logger.warning("⚠️ API offline. O watcher aguardará a conexão...")
    
    # Opção de processar arquivos existentes
    if api_online:
        processar_arquivos_existentes()
    
    logger.info("-" * 60)
    logger.info("👀 Monitoramento ATIVO")
    logger.info("📝 Logs salvos em: sharepoint_watcher.log")
    logger.info("⏹️  Pressione Ctrl+C para parar")
    logger.info("-" * 60)
    
    # Inicia o monitoramento
    event_handler = SharePointHandler()
    observer = Observer()
    observer.schedule(event_handler, PASTA_SHAREPOINT, recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Parando monitoramento...")
        observer.stop()
    
    observer.join()
    logger.info("👋 SharePoint Watcher finalizado")