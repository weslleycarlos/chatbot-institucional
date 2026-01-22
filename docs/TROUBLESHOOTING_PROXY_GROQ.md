# Guia de Troubleshooting - Proxy 407 e GROQ

## Problema
Ao tentar usar GROQ com proxy corporativo, recebemos erro `407 Proxy Authentication Required`.

## Causas Possíveis

1. **Credenciais incorretas**
   - Usuário: `pf\weslley.wcm` (verifique se é com domínio ou sem)
   - Senha: Pode ter expirado ou estar incorreta

2. **Formato de autenticação incompatível**
   - Seu proxy pode usar NTLM (Windows Authentication)
   - httpx/groq pode não suportar NTLM diretamente

3. **Escaping incorreto de caracteres especiais**
   - Caracteres como `@`, `$` na senha precisam ser URL-encoded

## Soluções

### Solução 1: Trocar para Gemini (Recomendado - Funciona Hoje)
Gemini já funciona. Configure no `.env`:
```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=seu_api_key_aqui
```

### Solução 2: Testar Credenciais do Proxy Manualmente
Abra PowerShell e teste:
```powershell
# Teste 1: Conectar ao proxy sem auth
[System.Net.ServicePointManager]::DefaultProxy = New-Object System.Net.WebProxy("http://proxy.dpf.gov.br:8080")
$response = Invoke-WebRequest -Uri "http://httpbin.org/ip" -TimeoutSec 5

# Teste 2: Com credenciais
$cred = New-Object System.Management.Automation.PSCredential('pf\weslley.wcm', (ConvertTo-SecureString 'SUA_SENHA' -AsPlainText -Force))
[System.Net.ServicePointManager]::DefaultProxy.Credentials = $cred
$response = Invoke-WebRequest -Uri "http://httpbin.org/ip" -TimeoutSec 5 -Proxy "http://proxy.dpf.gov.br:8080" -ProxyCredential $cred
```

Se isso funcionar, temos proof-of-concept das credenciais.

### Solução 3: Usar VPN (Se Disponível)
Se você tiver acesso a VPN corporativa, conecte à VPN e deixe o proxy desativado:
```env
# Deixar vazio ou remover:
PROXY_HOST=
PROXY_PORT=
PROXY_USER=
PROXY_PASS=

# Então GROQ funcionará:
LLM_PROVIDER=groq
```

### Solução 4: Usar GROQ com Proxy Alternativo
Se souber de outro proxy ou gateway, tente:
```env
PROXY_HOST=outro.proxy.com
PROXY_PORT=3128
PROXY_USER=user
PROXY_PASS=pass
LLM_PROVIDER=groq
```

### Solução 5: Contatar TI para Exceção de Firewall
Solicite ao seu TI uma exceção para as seguintes URLs (necessárias para GROQ):
- `api.groq.com:443` (HTTPS)
- `api-inference.huggingface.co:443` (para embeddings)

## Status Atual

- Gemini: ✅ **FUNCIONANDO**
- OpenAI: ✅ **FUNCIONANDO** (se API key configurada)
- GROQ: ⏳ **AGUARDANDO** configuração de proxy corporativo

## Próximos Passos

1. Escolha qual solução usar acima
2. Atualize `.env` conforme escolha
3. Reinicie a aplicação: `iniciar_dev.bat`
4. Teste no chat: `/admin` ou `/chat`

## Referências

- GROQ: https://console.groq.com
- Google Gemini: https://ai.google.dev
- OpenAI: https://platform.openai.com

---

**Nota**: O GROQ é completamente opcional. A aplicação funciona perfeitamente com Gemini ou OpenAI como LLM primário.
