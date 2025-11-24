import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Bot, User, FileText, Settings, RotateCcw, AlertCircle, Loader2 } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || "http://10.61.172.6:8000";

// Limite de caracteres para o input
const MAX_INPUT_LENGTH = 2000;

// Gerador de IDs únicos
let messageIdCounter = 0;
const generateId = () => `msg_${Date.now()}_${++messageIdCounter}`;

export default function ChatbotPage() {
  const [messages, setMessages] = useState([
    { 
      id: generateId(), 
      role: 'assistant', 
      text: 'Olá! Sou o assistente virtual da Intranet. Posso consultar portarias, manuais e leis internas. Como posso ajudar?',
      isWelcome: true
    }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  // Foca no input ao carregar
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Remove fontes duplicadas
  const deduplicateSources = (sources) => {
    if (!sources || sources.length === 0) return [];
    
    const seen = new Set();
    return sources.filter(s => {
      const key = `${s.name}-${s.page}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  };

  // Nova conversa
  const handleNewChat = () => {
    if (messages.length <= 1) return;
    
    if (window.confirm('Iniciar uma nova conversa? O histórico atual será perdido.')) {
      setMessages([{ 
        id: generateId(), 
        role: 'assistant', 
        text: 'Olá! Sou o assistente virtual da Intranet. Posso consultar portarias, manuais e leis internas. Como posso ajudar?',
        isWelcome: true
      }]);
      setInput('');
      inputRef.current?.focus();
    }
  };

  // Enviar mensagem
  const sendMessage = useCallback(async () => {
    const trimmedInput = input.trim();
    if (!trimmedInput || isTyping) return;
    
    const userMsg = { id: generateId(), role: 'user', text: trimmedInput };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    // Controller para timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmedInput }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!res.ok) {
        throw new Error(`Erro ${res.status}`);
      }
      
      const data = await res.json();
      
      setMessages(prev => [...prev, {
        id: generateId(),
        role: 'assistant',
        text: data.answer,
        sources: deduplicateSources(data.sources)
      }]);
      
    } catch (err) {
      clearTimeout(timeoutId);
      
      let errorMessage = "Desculpe, ocorreu um erro ao processar sua pergunta.";
      
      if (err.name === 'AbortError') {
        errorMessage = "A requisição demorou muito. Por favor, tente novamente.";
      } else if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
        errorMessage = "Não foi possível conectar ao servidor. Verifique se a API está rodando.";
      }
      
      setMessages(prev => [...prev, { 
        id: generateId(), 
        role: 'assistant', 
        text: errorMessage,
        isError: true
      }]);
    }
    
    setIsTyping(false);
    inputRef.current?.focus();
  }, [input, isTyping]);

  // Handler do input
  const handleInputChange = (e) => {
    const value = e.target.value;
    if (value.length <= MAX_INPUT_LENGTH) {
      setInput(value);
    }
  };

  // Handler do teclado
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 font-sans">
      {/* Header */}
      <div className="fixed top-0 w-full h-16 bg-slate-900 text-white flex items-center justify-between px-4 md:px-6 z-10 shadow-md">
        <div className="font-bold flex items-center gap-2">
          <div className="p-2 bg-blue-600 rounded-lg">
            <Bot size={20}/>
          </div>
          <span className="hidden sm:inline">GovBot - Assistente Virtual</span>
          <span className="sm:hidden">GovBot</span>
        </div>
        
        <div className="flex items-center gap-2">
          {/* Botão Nova Conversa */}
          <button 
            onClick={handleNewChat}
            disabled={messages.length <= 1}
            className="flex items-center gap-2 px-3 py-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 disabled:hover:bg-slate-700 rounded-lg transition-colors text-sm"
            title="Nova conversa"
          >
            <RotateCcw size={16}/>
            <span className="hidden md:inline">Nova Conversa</span>
          </button>
          
          {/* Botão Admin */}
          <a 
            href="/admin" 
            className="flex items-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors text-sm"
          >
            <Settings size={16}/>
            <span className="hidden md:inline">Admin</span>
          </a>
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col pt-16">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-4 md:space-y-6">
          {messages.map(msg => (
            <div 
              key={msg.id} 
              className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`flex gap-2 md:gap-3 max-w-[95%] md:max-w-[75%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                
                {/* Avatar */}
                <div className={`w-8 h-8 md:w-10 md:h-10 rounded-full flex items-center justify-center shrink-0 shadow-sm ${
                  msg.role === 'user' 
                    ? 'bg-slate-200' 
                    : msg.isError 
                      ? 'bg-red-500 text-white'
                      : 'bg-blue-600 text-white'
                }`}>
                  {msg.role === 'user' 
                    ? <User size={16} className="text-slate-500"/> 
                    : msg.isError 
                      ? <AlertCircle size={16}/> 
                      : <Bot size={16}/>
                  }
                </div>

                <div className="flex flex-col gap-1">
                  {/* Mensagem */}
                  <div className={`p-3 md:p-4 rounded-2xl shadow-sm text-sm md:text-base leading-relaxed whitespace-pre-line ${
                    msg.role === 'user' 
                      ? 'bg-white border border-slate-200 text-slate-800 rounded-tr-none' 
                      : msg.isError
                        ? 'bg-red-50 border border-red-200 text-red-800 rounded-tl-none'
                        : 'bg-white border border-blue-100 text-slate-800 rounded-tl-none'
                  }`}>
                    {msg.text}
                  </div>
                  
                  {/* Sources */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-1 flex flex-wrap gap-1.5 md:gap-2">
                      {msg.sources.map((s, i) => (
                        <div 
                          key={`${s.name}-${s.page}-${i}`} 
                          className="flex items-center gap-1 md:gap-1.5 bg-blue-50 text-blue-700 px-2 md:px-3 py-1 md:py-1.5 rounded-full border border-blue-100 text-xs font-medium"
                          title={s.name}
                        >
                          <FileText size={10} className="md:w-3 md:h-3"/> 
                          <span className="truncate max-w-[100px] md:max-w-[150px]">{s.name}</span>
                          {s.page !== 'N/A' && (
                            <span className="opacity-70 text-[10px] md:text-xs">p.{s.page}</span>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
          
          {/* Typing Indicator */}
          {isTyping && (
            <div className="flex gap-2 md:gap-3 items-center">
              <div className="w-8 h-8 md:w-10 md:h-10 bg-blue-600 rounded-full flex items-center justify-center">
                <Bot size={16} className="text-white"/>
              </div>
              <div className="flex items-center gap-2 bg-white px-4 py-3 rounded-2xl rounded-tl-none border border-slate-200">
                <Loader2 size={16} className="animate-spin text-blue-600"/>
                <span className="text-sm text-slate-500">Pensando...</span>
              </div>
            </div>
          )}
          
          <div ref={chatEndRef} />
        </div>
        
        {/* Input Area */}
        <div className="p-3 md:p-6 bg-white border-t border-slate-200">
          <div className="max-w-4xl mx-auto">
            <div className="flex items-end gap-2">
              <div className="flex-1 relative">
                <textarea 
                  ref={inputRef}
                  className="w-full bg-slate-50 border border-slate-200 rounded-xl p-3 md:p-4 pr-4 shadow-inner focus:bg-white focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all resize-none text-sm md:text-base"
                  placeholder="Pergunte sobre portarias, leis ou serviços..."
                  value={input}
                  onChange={handleInputChange}
                  onKeyDown={handleKeyDown}
                  disabled={isTyping}
                  rows={1}
                  style={{ 
                    minHeight: '48px',
                    maxHeight: '120px',
                    height: 'auto'
                  }}
                  onInput={(e) => {
                    e.target.style.height = 'auto';
                    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
                  }}
                />
                
                {/* Contador de caracteres */}
                {input.length > MAX_INPUT_LENGTH * 0.8 && (
                  <div className={`absolute bottom-1 right-2 text-xs ${
                    input.length >= MAX_INPUT_LENGTH ? 'text-red-500' : 'text-slate-400'
                  }`}>
                    {input.length}/{MAX_INPUT_LENGTH}
                  </div>
                )}
              </div>
              
              {/* Botão Enviar */}
              <button 
                onClick={sendMessage} 
                disabled={!input.trim() || isTyping}
                className="p-3 md:p-4 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:hover:bg-blue-600 transition-all shrink-0"
                title="Enviar mensagem (Enter)"
              >
                <Send size={20}/>
              </button>
            </div>
            
            {/* Dica */}
            <p className="text-xs text-slate-400 mt-2 text-center hidden md:block">
              Pressione Enter para enviar • Shift+Enter para nova linha
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}