# Correcao: Erro "list object has no attribute split/lower"

## Problema
Ao usar `gemini-3-flash-preview`, o sistema retorna erros:
- `'list' object has no attribute 'split'`
- `'list' object has no attribute 'lower'`

## Causa
O modelo Gemini 3 Flash Preview às vezes retorna `response.content` como uma **lista** em vez de uma **string**.

Isso ocorria em 3 lugares:
1. **Linha 351**: Gerando variações de pergunta - tentava fazer `.split('\n')`
2. **Linha 372**: Gerando HyDE (documento hipotético)
3. **Linha 467**: Filtrando sources - tentava fazer `.lower()`

## Solucao Implementada

Adicionado verificação de tipo em todos os lugares onde `response.content` é usado:

```python
# Antes (problema):
variations = response.content.split('\n')

# Depois (corrigido):
content = response.content if isinstance(response.content, str) else '\n'.join(str(c) for c in response.content) if isinstance(response.content, list) else str(response.content)
variations = content.split('\n')
```

### Locais Corrigidos

1. **Variações de pergunta** (linha ~352)
   - Converte lista para string unindo com `\n`

2. **Documento hipotético HyDE** (linha ~372)
   - Converte lista para string unindo sem separador

3. **Filtragem de sources** (linha ~470)
   - Converte lista para string para fazer `.lower()`

4. **Retorno da resposta** (linha ~505)
   - Converte lista para string usando `str()`

## Como Usar

### Opção 1: Usar Gemini 2.5 Flash (Recomendado)
```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
```

### Opção 2: Usar Gemini 3 Flash Preview (Agora Corrigido)
```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3-flash-preview
```

Ambos devem funcionar agora sem erros de `'list' object`.

## Testes

Para validar a correção:

```bash
# Terminal 1
cd backend
uvicorn main:app --reload

# Terminal 2
cd frontend
npm run dev

# Abrir: http://localhost:5173/chat
# Fazer uma pergunta
```

Se receber resposta normal sem erro de `'list' object`, a correção funcionou!

## Arquivos Modificados

- `backend/main.py` - Adicionada conversão de tipo para `response.content` em 4 locais

---

**Data**: Janeiro 2026
**Status**: ✅ Corrigido
