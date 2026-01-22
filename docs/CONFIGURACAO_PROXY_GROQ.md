# Configuração de Proxy para GROQ (e outras APIs)

## 🔍 Diagnóstico do Seu Erro

Você recebe:
```
httpcore.ProxyError: 407 Proxy Authentication Required
```

Isso significa que **sua rede requer autenticação no proxy** para acessar api.groq.com.

## ✅ Solução: Configurar Proxy com Autenticação

### **Passo 1: Editar `backend/.env`**

Adicione as credenciais do proxy:

```env
# === CONFIGURAÇÃO DE PROXY ===
PROXY_HOST=proxy.dpf.gov.br        # Seu host proxy (já pré-configurado)
PROXY_PORT=8080                    # Porta (8080 é padrão)
PROXY_USER=seu_usuario             # Seu usuário de rede/AD
PROXY_PASS=sua_senha               # Sua senha
```

**Exemplo completo:**
```env
PROXY_HOST=proxy.dpf.gov.br
PROXY_PORT=8080
PROXY_USER=weslley.wcm
PROXY_PASS=SuaSenha@123
```

### **Passo 2: Como Descobrir Suas Credenciais de Proxy**

#### Opção A: Windows (Automático)
Se você estiver logado em um Windows corporativo, às vezes o proxy usa suas credenciais:
```powershell
# Abra PowerShell e execute:
netsh winhttp show proxy
```

#### Opção B: Configurações do Internet Explorer/Edge
1. Abra `Internet Options` (ou Edge Settings)
2. Vá para `Network > Proxy`
3. Veja as credenciais salvas

#### Opção C: Pergunte ao TI
Entre em contato com seu departamento de TI para obter:
- Host do proxy (ex: `proxy.dpf.gov.br`)
- Porta (ex: `8080`)
- Usuário (ex: `seu_usuario`)
- Senha

### **Passo 3: Teste a Configuração**

```bash
cd backend
.\venv\Scripts\activate.ps1
python teste_groq.py
```

Deve mostrar:
```
✅ Teste de Conexão (SIMPLES):
✅ Resposta recebida com sucesso!
```

## 🔧 Como Funciona a Solução

**Arquivo: `backend/proxy_config.py`**

Agora detecta proxy com autenticação:
```python
if proxy_user and proxy_pass:
    http_proxy = f"http://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}"
```

Isso adiciona as credenciais à URL de proxy:
```
http://user:pass@proxy.host:port
```

Quando o Python tenta conectar, apresenta as credenciais automaticamente.

## 🌍 Por que Funciona no Navegador?

- Chrome/Edge salvam as credenciais de proxy
- Usam autenticação NTLM/Kerberos integrada do Windows
- Python precisa de configuração explícita

## 🧪 Alternativas se Proxy Não Funcionar

### **1. Usar VPN da Empresa**
Se sua rede tiver VPN:
```bash
# Conectar na VPN antes de rodar
# Depois o proxy pode ser desnecessário
python -m uvicorn main:app --reload
```

### **2. Usar Gemini no lugar de GROQ**
Gemini pode já estar funcionando (não requer proxy adicional):
```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
```

Teste:
```bash
python -c "from main import _get_llm_instance; llm = _get_llm_instance(); print(llm.invoke('OK'))"
```

## 📋 Resumo da Configuração

| Componente | Valor | Onde |
|-----------|-------|------|
| Proxy Host | proxy.dpf.gov.br | `.env` ou Windows |
| Proxy Port | 8080 | `.env` ou Windows |
| Proxy User | seu_usuario | `.env` (novo) |
| Proxy Pass | sua_senha | `.env` (novo) |
| Script | proxy_config.py | Lê do `.env` |

## 🚀 Próximos Passos

1. **Configure as credenciais de proxy no `.env`**
2. **Execute o teste**: `python teste_groq.py`
3. **Se funcionar, use GROQ normalmente**:
   ```env
   LLM_PROVIDER=groq
   LLM_MODEL=mixtral-8x7b-32768
   ```

---

**Dúvidas?** O erro `407 Proxy Authentication Required` significa que você está bem perto de funcionar, apenas faltam as credenciais!
