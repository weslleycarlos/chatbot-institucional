import os
from dotenv import load_dotenv
from urllib.parse import quote

load_dotenv()

def configurar_proxy():
    """
    Detecta e configura proxy automaticamente se estiver no Windows da empresa.
    Suporta proxy com e sem autenticação.
    Escape de caracteres especiais em senhas.
    Isso evita o erro 'Connection Refused' ao tentar falar com APIs externas.
    """
    # Tenta pegar do ambiente do sistema
    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    
    # Se não tiver, tenta variables de ambiente de proxy com autenticação
    proxy_user = os.environ.get("PROXY_USER") or os.getenv("PROXY_USER")
    proxy_pass = os.environ.get("PROXY_PASS") or os.getenv("PROXY_PASS")
    proxy_host = os.environ.get("PROXY_HOST") or "proxy.dpf.gov.br"
    proxy_port = os.environ.get("PROXY_PORT") or "8080"

    if not http_proxy:
        # Construir proxy URL com autenticação se disponível
        if proxy_user and proxy_pass:
            # IMPORTANTE: Escape de caracteres especiais na senha
            # $, @, #, %, &, etc. precisam ser escapados em URLs
            proxy_pass_escaped = quote(proxy_pass, safe='')
            proxy_user_escaped = quote(proxy_user, safe='\\')  # Mantém barra invertida
            
            http_proxy = f"http://{proxy_user_escaped}:{proxy_pass_escaped}@{proxy_host}:{proxy_port}"
            https_proxy = f"http://{proxy_user_escaped}:{proxy_pass_escaped}@{proxy_host}:{proxy_port}"
            
            print(f"📡 Proxy configurado: {proxy_host}:{proxy_port}")
            print(f"   Com autenticação: {proxy_user}")
            print(f"   (Caracteres especiais da senha foram escapados)")
        else:
            http_proxy = f"http://{proxy_host}:{proxy_port}"
            https_proxy = f"http://{proxy_host}:{proxy_port}"
            print(f"📡 Proxy configurado: {proxy_host}:{proxy_port}")
            print(f"   (sem autenticação)")
        
        os.environ["HTTP_PROXY"] = http_proxy
        os.environ["HTTPS_PROXY"] = https_proxy
    else:
        print(f"📡 Proxy detectado do sistema: {http_proxy}")

# Chame esta função no início do seu main.py