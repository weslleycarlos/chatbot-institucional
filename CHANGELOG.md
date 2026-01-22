# Changelog

Todas as mudanças importantes neste projeto serão documentadas aqui.

## [1.0.0] - 2025-01-22

### ✨ Adicionado
- **RAG Otimizado**: Query variations geradas localmente (~50ms vs 3-5s com LLM)
- **Suporte Multi-Gemini**: Compatibilidade com Gemini 2.5, 2.0 e 3 Flash Preview
- **HyDE Opcional**: Feature de Hypothetical Document Embeddings configurável
- **Cleanup Automático**: Remoção de markdown das respostas
- **Admin Panel**: Interface para gerenciar documentos com autenticação
- **Chat Interface**: UI responsiva para chat em tempo real
- **Proxy Support**: Suporte para redes corporativas com proxy autenticado
- **SharePoint Watcher**: Monitor automático de pastas sincronizadas

### 🔧 Corrigido
- **Gemini 3 Flash Compatibility**: Tratamento de resposta em formato lista
- **Encoding Issues**: Suporte correto para português
- **Proxy Auth**: Configuração simplificada de autenticação

### ⚡ Otimizado
- **Pipeline RAG**: Redução de ~40% no tempo de resposta (7-10s → 4-6s)
- **Query Variations**: Algoritmo local substitui chamadas ao LLM
- **Embeddings**: Suporte a modelo offline (sentence-transformers)

### 📚 Documentado
- README.md completo com todas as features
- .env.example com configurações explicadas
- API documentation com exemplos curl
- Troubleshooting guide
- Contributing guide
- Architecture diagram

### 🧹 Limpeza
- Removidos arquivos de diagnóstico desnecessários
- Remcódigos de teste temporários
- Otimizado .gitignore
- Estrutura de diretórios organizada

## [Próximas Features]

- [ ] Suporte para múltiplas línguas
- [ ] Rate limiting para endpoints de chat
- [ ] Integração com OAuth
- [ ] Docker Compose setup
- [ ] CI/CD com GitHub Actions
- [ ] Testes automatizados (pytest, Jest)
- [ ] WebSocket para chat em tempo real
- [ ] Cache de respostas frecuentes
- [ ] Analytics e logging
- [ ] Suporte a embeddings customizados

## Notas

- Versão mínima: Python 3.9+, Node.js 16+
- API Key obrigatória: Google Gemini
- Licença: MIT
