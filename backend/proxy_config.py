import os

def configurar_proxy():
    """
    Detecta e configura proxy automaticamente se estiver no Windows da empresa.
    Isso evita o erro 'Connection Refused' ao tentar falar com o Google.
    """
    # Tenta pegar do ambiente do sistema
    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

    if not http_proxy:
        # Se não tiver definido, às vezes é necessário forçar (exemplo fictício)
        # os.environ["HTTP_PROXY"] = "http://seu.proxy.aqui:8080"
        # os.environ["HTTPS_PROXY"] = "http://seu.proxy.aqui:8080"
        pass
    else:
        print(f"📡 Proxy detectado: {http_proxy}")

# Chame esta função no início do seu main.py