import React, { useState, useEffect, useRef } from 'react';
import { Send, BookOpen, Bot, User, Settings, Upload, Lock, Unlock, FileText, CheckCircle, AlertCircle, Trash2, Menu, X } from 'lucide-react';

// Aponta para o teu backend Python (FastAPI)
const API_URL = "http://localhost:8000";

export default function App() {
  // Estados de Navegação
  const [view, setView] = useState('chat'); // 'chat' | 'admin'
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Estados de Autenticação
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [adminPass, setAdminPass] = useState('');
  const [showLoginModal, setShowLoginModal] = useState(false);
  
  // Estados do Chat
  const [messages, setMessages] = useState([
    { id: 1, role: 'assistant', text: 'Olá! Sou o assistente virtual da Intranet. Posso consultar portarias, manuais e leis internas. Como posso ajudar?' }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef(null);
  
  // Estados do Admin
  const [uploadStatus, setUploadStatus] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  // Auto-scroll no chat
  useEffect(() => {
    if (view === 'chat') chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping, view]);

  // --- FUNÇÕES ---

  const handleLogin = (e) => {
    e.preventDefault();
    // Validação preliminar no frontend. A segurança real é feita pelo Backend ao tentar fazer upload.
    if (adminPass) {
      setIsAuthenticated(true);
      setShowLoginModal(false);
      setView('admin');
      setAdminPass(''); // Limpa o campo por segurança, mas idealmente guardarias o token/senha em memória para reuso
    }
  };

  // Precisamos guardar a senha temporariamente para enviar nas requisições de upload
  // Num app real, usaríamos um token JWT, mas para Basic Auth guardamos em ref ou estado
  const savedPassword = useRef(''); 

  const confirmLogin = (e) => {
    e.preventDefault();
    if(adminPass) {
        savedPassword.current = adminPass; // Guarda para usar no header
        setIsAuthenticated(true);
        setShowLoginModal(false);
        setView('admin');
        setAdminPass('');
    }
  }

  const sendMessage = async () => {
    if (!input.trim()) return;
    
    const userMsg = { id: Date.now(), role: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMsg.text })
      });
      
      if (!res.ok) throw new Error('Falha na API');
      
      const data = await res.json();
      
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        text: data.answer,
        sources: data.sources
      }]);
    } catch (err) {
      setMessages(prev => [...prev, { 
        id: Date.now(), 
        role: 'assistant', 
        text: "Desculpe, não consegui conectar ao servidor. Verifique se o 'main.py' está a correr." 
      }]);
    }
    setIsTyping(false);
  };

  const handleUpload = async (e) => {
    const file = e.target.files ? e.target.files[0] : null;
    if (!file) return;

    setUploadStatus('Enviando...');
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Cria o cabeçalho de Autenticação Basic
      // O utilizador:senha deve bater com o que está no main.py
      const credentials = btoa(`admin:${savedPassword.current}`);
      
      const res = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        headers: {
          'Authorization': `Basic ${credentials}`
        },
        body: formData
      });

      if (res.status === 401) {
        setUploadStatus('Erro: Senha de Administrador Incorreta!');
        setIsAuthenticated(false); 
        setShowLoginModal(true);
      } else if (res.ok) {
        const data = await res.json();
        setUploadStatus(`Sucesso! ${data.chunks_criados} trechos indexados.`);
      } else {
        setUploadStatus('Erro no processamento do arquivo.');
      }
    } catch (err) {
      setUploadStatus('Erro de conexão com o servidor.');
    }
  };

  const handleClearBase = async () => {
    if(!window.confirm("Tem a certeza? Isto apagará todo o conhecimento da IA.")) return;

    try {
        const credentials = btoa(`admin:${savedPassword.current}`);
        const res = await fetch(`${API_URL}/limpar_base`, {
            method: 'DELETE',
            headers: { 'Authorization': `Basic ${credentials}` }
        });
        if(res.ok) alert("Base limpa com sucesso.");
        else alert("Erro ao limpar base.");
    } catch (e) {
        alert("Erro de conexão.");
    }
  }

  // --- RENDERIZAÇÃO ---

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 font-sans overflow-hidden">
      
      {/* Sidebar (Desktop) */}
      <div className="hidden md:flex w-64 bg-slate-900 text-white flex-col p-4 shadow-xl z-10">
        <div className="font-bold text-xl mb-8 flex items-center gap-2 tracking-tight">
          <div className="p-2 bg-blue-600 rounded-lg"><Bot size={20} className="text-white"/></div>
          GovBot
        </div>
        
        <nav className="space-y-2 flex-1">
            <button onClick={() => setView('chat')} className={`w-full p-3 rounded-lg flex gap-3 items-center transition-all ${view === 'chat' ? 'bg-blue-600 shadow-lg shadow-blue-900/50' : 'hover:bg-slate-800 text-slate-400 hover:text-white'}`}>
            <Bot size={20}/> Chat
            </button>
            
            <button onClick={() => isAuthenticated ? setView('admin') : setShowLoginModal(true)} className={`w-full p-3 rounded-lg flex gap-3 items-center transition-all ${view === 'admin' ? 'bg-blue-600 shadow-lg shadow-blue-900/50' : 'hover:bg-slate-800 text-slate-400 hover:text-white'}`}>
            {isAuthenticated ? <Unlock size={20} className="text-green-400"/> : <Lock size={20}/>} 
            Área Admin
            </button>
        </nav>

        <div className="text-xs text-slate-500 pt-4 border-t border-slate-800">
            v1.0 • Intranet Edition
        </div>
      </div>

      {/* Mobile Header */}
      <div className="md:hidden fixed top-0 w-full h-16 bg-slate-900 text-white flex items-center justify-between px-4 z-20 shadow-md">
          <span className="font-bold flex items-center gap-2"><Bot size={20}/> GovBot</span>
          <button onClick={() => setMobileMenuOpen(!mobileMenuOpen)}>
              {mobileMenuOpen ? <X/> : <Menu/>}
          </button>
      </div>

      {/* Mobile Menu Overlay */}
      {mobileMenuOpen && (
          <div className="fixed inset-0 bg-slate-900 z-10 pt-20 px-4 space-y-4 md:hidden">
              <button onClick={() => {setView('chat'); setMobileMenuOpen(false)}} className="block w-full text-left p-4 bg-slate-800 rounded text-white">Chat</button>
              <button onClick={() => {isAuthenticated ? setView('admin') : setShowLoginModal(true); setMobileMenuOpen(false)}} className="block w-full text-left p-4 bg-slate-800 rounded text-white">Admin</button>
          </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col relative pt-16 md:pt-0">
        
        {/* LOGIN MODAL */}
        {showLoginModal && (
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-white p-8 rounded-2xl shadow-2xl w-full max-w-md transform transition-all scale-100">
              <div className="text-center mb-6">
                  <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3 text-blue-600">
                      <Lock size={24}/>
                  </div>
                  <h3 className="font-bold text-2xl text-slate-800">Acesso Restrito</h3>
                  <p className="text-slate-500 text-sm mt-1">Esta área permite alterar a base de conhecimento.</p>
              </div>
              
              <form onSubmit={confirmLogin}>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">Senha de Administrador</label>
                <input 
                  type="password" 
                  className="w-full border border-slate-300 p-3 rounded-lg mb-6 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                  value={adminPass}
                  onChange={e => setAdminPass(e.target.value)}
                  placeholder="Digite a senha..."
                  autoFocus
                />
                <div className="flex gap-3">
                  <button type="button" onClick={() => setShowLoginModal(false)} className="flex-1 py-3 text-slate-600 bg-slate-100 rounded-lg hover:bg-slate-200 font-medium transition-colors">Cancelar</button>
                  <button type="submit" className="flex-1 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium shadow-lg shadow-blue-600/30 transition-all">Acessar</button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* VIEW: CHAT */}
        {view === 'chat' && (
          <div className="flex-1 flex flex-col max-w-5xl mx-auto w-full h-full">
            
            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
              {messages.map(msg => (
                <div key={msg.id} className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`flex gap-3 max-w-[90%] md:max-w-[75%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                    
                    <div className={`w-8 h-8 md:w-10 md:h-10 rounded-full flex items-center justify-center shrink-0 shadow-sm ${msg.role === 'user' ? 'bg-slate-200' : 'bg-blue-600 text-white'}`}>
                        {msg.role === 'user' ? <User size={18} className="text-slate-500"/> : <Bot size={18}/>}
                    </div>

                    <div className="flex flex-col gap-1">
                        <div className={`p-4 rounded-2xl shadow-sm text-sm md:text-base leading-relaxed whitespace-pre-line ${
                            msg.role === 'user' 
                            ? 'bg-white border border-slate-200 text-slate-800 rounded-tr-none' 
                            : 'bg-white border border-blue-100 text-slate-800 rounded-tl-none'
                        }`}>
                            {msg.text}
                        </div>
                        
                        {/* Sources (RAG) */}
                        {msg.sources && msg.sources.length > 0 && (
                        <div className="mt-1 flex flex-wrap gap-2">
                            {msg.sources.map((s, i) => (
                                <div key={i} className="flex items-center gap-1.5 bg-blue-50 text-blue-700 px-3 py-1.5 rounded-full border border-blue-100 text-xs font-medium hover:bg-blue-100 cursor-default transition-colors">
                                    <FileText size={12}/> 
                                    <span className="truncate max-w-[150px]">{s.name}</span>
                                    {s.page !== 'N/A' && <span className="opacity-70">Pág {s.page}</span>}
                                </div>
                            ))}
                        </div>
                        )}
                    </div>
                  </div>
                </div>
              ))}
              {isTyping && (
                 <div className="flex gap-3 items-center text-slate-400 text-sm ml-2">
                    <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center"><Bot size={16} className="text-white"/></div>
                    <div className="flex gap-1 bg-white px-4 py-3 rounded-2xl rounded-tl-none border border-slate-200">
                        <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></span>
                        <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-100"></span>
                        <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-200"></span>
                    </div>
                 </div>
              )}
              <div ref={chatEndRef} />
            </div>
            
            {/* Input Area */}
            <div className="p-4 bg-white border-t border-slate-200 shadow-lg md:shadow-none z-10">
              <div className="max-w-4xl mx-auto relative flex items-center gap-2">
                <input 
                  className="flex-1 bg-slate-50 border border-slate-200 rounded-xl p-4 pr-12 shadow-inner focus:bg-white focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                  placeholder="Pergunte sobre portarias, leis ou serviços..."
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && sendMessage()}
                  disabled={isTyping}
                />
                <button 
                    onClick={sendMessage} 
                    disabled={!input.trim() || isTyping}
                    className="absolute right-2 p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:hover:bg-blue-600 transition-all shadow-md"
                >
                    <Send size={20}/>
                </button>
              </div>
              <div className="text-center mt-2 text-[10px] text-slate-400 uppercase tracking-wider">
                A IA pode cometer erros. Verifique sempre as fontes citadas.
              </div>
            </div>
          </div>
        )}

        {/* VIEW: ADMIN */}
        {view === 'admin' && (
          <div className="flex-1 overflow-y-auto bg-slate-50 p-4 md:p-8">
            <div className="max-w-3xl mx-auto w-full space-y-8">
              
              <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-bold text-slate-800 flex items-center gap-2"><Settings className="text-slate-400"/> Painel de Controle</h2>
                    <p className="text-slate-500">Gerencie a base de conhecimento da Intranet.</p>
                </div>
                <div className="text-xs bg-green-100 text-green-700 px-3 py-1 rounded-full border border-green-200 font-medium flex items-center gap-1">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    Sessão Autenticada
                </div>
              </div>
              
              {/* Upload Card */}
              <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200 space-y-6">
                <div className="border-b border-slate-100 pb-4">
                    <h3 className="font-semibold text-lg text-slate-800">Upload de Documentos</h3>
                    <p className="text-sm text-slate-500">Adicione PDFs ou DOCX para que a IA aprenda o conteúdo.</p>
                </div>
                
                <div 
                    className={`border-2 border-dashed rounded-xl p-10 text-center transition-all cursor-pointer ${dragActive ? 'border-blue-500 bg-blue-50' : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50'}`}
                    onDragEnter={() => setDragActive(true)}
                    onDragLeave={() => setDragActive(false)}
                    onDrop={(e) => {e.preventDefault(); setDragActive(false);}}
                >
                  <input type="file" id="fileUpload" onChange={handleUpload} className="hidden" accept=".pdf,.docx"/>
                  <label htmlFor="fileUpload" className="cursor-pointer flex flex-col items-center gap-4">
                      <div className="w-16 h-16 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center">
                          <Upload size={32}/>
                      </div>
                      <div>
                          <p className="font-medium text-slate-700 text-lg">Clique para escolher um arquivo</p>
                          <p className="text-sm text-slate-400">Suporta PDF e DOCX (Max 10MB)</p>
                      </div>
                  </label>
                </div>

                {uploadStatus && (
                  <div className={`p-4 rounded-lg text-sm font-medium flex items-center gap-2 animate-fade-in ${uploadStatus.includes('Erro') ? 'bg-red-50 text-red-700 border border-red-100' : 'bg-green-50 text-green-700 border border-green-100'}`}>
                    {uploadStatus.includes('Erro') ? <AlertCircle size={18}/> : <CheckCircle size={18}/>}
                    {uploadStatus}
                  </div>
                )}
              </div>

              {/* Danger Zone */}
              <div className="bg-red-50 p-6 rounded-2xl border border-red-100 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                <div>
                    <h3 className="font-bold text-red-800 flex items-center gap-2"><AlertCircle size={20}/> Zona de Perigo</h3>
                    <p className="text-sm text-red-600 mt-1">Apagar a base de dados remove todo o conhecimento indexado até agora.</p>
                </div>
                <button onClick={handleClearBase} className="flex items-center gap-2 px-5 py-2.5 bg-white border border-red-200 text-red-700 rounded-lg hover:bg-red-100 font-medium transition shadow-sm">
                  <Trash2 size={18}/> Limpar Tudo
                </button>
              </div>

            </div>
          </div>
        )}
      </div>
    </div>
  );
}