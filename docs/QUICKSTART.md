# QUICK START - GovBot

## 30 Segundos para Funcionar

### Pré-requisitos
- Python 3.10+
- Node.js 16+
- Git

### Passos

**1. Verificar Configuração** (1min)
```bash
cd backend
python verificar_config.py
# Deve mostrar: [OK] Configuracao basica pronta para iniciar!
```

**2. Iniciar Backend** (2min)
```bash
cd backend
.\venv\Scripts\activate  # ou: source venv/bin/activate
uvicorn main:app --reload
# Deve mostrar: Uvicorn running on http://127.0.0.1:8000
```

**3. Em novo terminal: Iniciar Frontend** (2min)
```bash
cd frontend
npm install  # só na primeira vez
npm run dev
# Deve mostrar: VITE ready in XXX ms
```

**4. Abrir Navegador**
```
http://localhost:5173
```

## Pronto! 🚀

- **Chat**: http://localhost:5173/ (fazer perguntas)
- **Admin**: http://localhost:5173/admin (login: admin / senha: veja .env)

---

## Troubleshooting Rápido

| Problema | Solução |
|----------|---------|
| `ModuleNotFoundError: langchain` | `pip install -r requirements.txt` |
| `GOOGLE_API_KEY not found` | Verificar `backend/.env` |
| `Uvicorn not found` | `pip install uvicorn` |
| `npm: command not found` | Instalar Node.js |
| Front não conecta API | Verificar `VITE_API_URL` no `.env` |
| 407 Proxy Error | Ver [TROUBLESHOOTING_PROXY_GROQ.md](./docs/TROUBLESHOOTING_PROXY_GROQ.md) |

---

## Próximos Passos Recomendados

1. **Testar Chat**
   - Upload um PDF em `/admin`
   - Faça uma pergunta sobre o documento em `/chat`

2. **Mudar LLM (opcional)**
   - Edite `backend/.env`: `LLM_PROVIDER=openai`
   - Restart do backend

3. **Colocar em Produção**
   - Ver [STATUS.md](./STATUS.md)

4. **Adicionar Mais Docs**
   - Admin panel → Upload
   - Suporta PDF e DOCX

---

## Documentação Completa

- [STATUS.md](./STATUS.md) - Visão geral do projeto
- [SESSAO_TRABALHO.md](./SESSAO_TRABALHO.md) - O que foi feito nesta sessão
- [docs/TROUBLESHOOTING_PROXY_GROQ.md](./docs/TROUBLESHOOTING_PROXY_GROQ.md) - Proxy help
- [README.md](./README.md) - Documentação original

---

**Tempo total**: ~5-10 minutos
**Dificuldade**: ⭐ Muito Fácil
