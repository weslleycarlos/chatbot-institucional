import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Lock, FileText, CheckCircle, AlertCircle, Trash2, Database, RefreshCw, ArrowLeft, Settings } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || "http://10.61.172.6:8000";

export default function AdminPage() {
  const navigate = useNavigate();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [adminPass, setAdminPass] = useState('');
  const [savedPassword, setSavedPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  
  const [uploadStatus, setUploadStatus] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState(false);
  const [baseStats, setBaseStats] = useState({ totalChunks: 0, totalDocuments: 0 });

  // ✅ Carrega dados quando autenticado
  useEffect(() => {
    if (isAuthenticated && savedPassword) {
      loadDocuments();
      getBaseStats();
    }
  }, [isAuthenticated, savedPassword]);

  const getAuthHeaders = () => {
    if (!savedPassword) return {};
    const credentials = btoa(`admin:${savedPassword}`);
    return { 'Authorization': `Basic ${credentials}` };
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoginError('');
    
    if (!adminPass) {
      setLoginError('❌ Digite uma senha');
      return;
    }

    try {
      const testCredentials = btoa(`admin:${adminPass}`);
      const res = await fetch(`${API_URL}/documentos`, {
        headers: { 'Authorization': `Basic ${testCredentials}` }
      });
      
      console.log('🔐 Status da resposta:', res.status);
      
      if (res.ok) {
        setSavedPassword(adminPass);
        setIsAuthenticated(true);
        setAdminPass('');
        setLoginError('');
        // ✅ Removido: loadDocuments() e getBaseStats() - agora no useEffect
      } else if (res.status === 401) {
        setLoginError('❌ Senha incorreta!');
        setIsAuthenticated(false);
        setSavedPassword('');
      } else {
        setLoginError('❌ Erro no servidor');
      }
    } catch (error) {
      console.error('Erro de conexão:', error);
      setLoginError('❌ Erro de conexão com o servidor');
    }
  };

  const handleLogout = () => {
    setSavedPassword('');
    setIsAuthenticated(false);
    setAdminPass('');
    setLoginError('');
    setDocuments([]);
    setBaseStats({ totalChunks: 0, totalDocuments: 0 });
  };

  const loadDocuments = async () => {
    if (!savedPassword) return;
    
    setIsLoadingDocs(true);
    try {
      const res = await fetch(`${API_URL}/documentos`, {
        headers: getAuthHeaders()
      });
      
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documentos || []);
      } else if (res.status === 401) {
        handleLogout();
      }
    } catch (error) {
      console.error('Erro ao carregar documentos:', error);
    }
    setIsLoadingDocs(false);
  };

  const getBaseStats = async () => {
    if (!savedPassword) return;
    
    try {
      const res = await fetch(`${API_URL}/documentos`, {
        headers: getAuthHeaders()
      });
      
      if (res.ok) {
        const data = await res.json();
        setBaseStats({
          totalChunks: data.total_chunks || 0,
          totalDocuments: data.documentos ? data.documentos.length : 0
        });
      }
    } catch (error) {
      console.error('Erro ao carregar estatísticas:', error);
    }
  };

  const handleClearBase = async () => {
    if(!window.confirm("🚨 TEM CERTEZA ABSOLUTA?\n\nIsso apagará TODOS os documentos!\nEsta ação NÃO pode ser desfeita!")) return;

    try {
      const res = await fetch(`${API_URL}/limpar_base`, {
        method: 'DELETE',
        headers: getAuthHeaders()
      });
      
      if(res.ok) {
        const data = await res.json();
        alert("✅ " + data.status);
        setDocuments([]);
        setBaseStats({ totalChunks: 0, totalDocuments: 0 });
        setUploadStatus("Base reiniciada com sucesso!");
      } else if (res.status === 401) {
        alert("❌ Erro: Sessão expirada!");
        handleLogout();
      } else {
        alert("❌ Erro ao limpar base.");
      }
    } catch (e) {
      alert("❌ Erro de conexão.");
    }
  };

  const handleClearUploads = async () => {
    if(!window.confirm("Limpar apenas arquivos uploadados?\nA base de conhecimento será mantida.")) return;

    try {
      const res = await fetch(`${API_URL}/limpar_uploads`, {
        method: 'DELETE',
        headers: getAuthHeaders()
      });
      
      if(res.ok) {
        const data = await res.json();
        alert("✅ " + data.status);
        setUploadStatus("Arquivos temporários removidos!");
      } else if (res.status === 401) {
        alert("❌ Erro: Sessão expirada!");
        handleLogout();
      } else {
        alert("❌ Erro ao limpar uploads.");
      }
    } catch (e) {
      alert("❌ Erro de conexão.");
    }
  };

  // ✅ Função de upload extraída para reutilização
  const processUpload = async (file) => {
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf') && !file.name.toLowerCase().endsWith('.docx')) {
      setUploadStatus('❌ Erro: Apenas PDF e DOCX são permitidos!');
      return;
    }

    setUploadStatus('📤 Enviando arquivo...');
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: formData
      });

      if (res.status === 401) {
        setUploadStatus('❌ Erro: Sessão expirada!');
        handleLogout();
      } else if (res.ok) {
        const data = await res.json();
        setUploadStatus(`✅ Sucesso! ${data.chunks} trechos indexados de "${data.documento}"`);
        loadDocuments();
        getBaseStats();
      } else {
        const errorData = await res.json();
        setUploadStatus(`❌ Erro: ${errorData.detalhes || 'Processamento falhou'}`);
      }
    } catch (err) {
      setUploadStatus('❌ Erro de conexão com o servidor.');
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files ? e.target.files[0] : null;
    await processUpload(file);
    if (e.target.value) e.target.value = '';
  };

  // ✅ Handler para Drag & Drop
  const handleDrop = async (e) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    await processUpload(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragActive(false);
  };

  // Página de Login
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center p-4">
        <div className="bg-white p-8 rounded-2xl shadow-2xl w-full max-w-md">
          <div className="text-center mb-6">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4 text-blue-600">
              <Lock size={32}/>
            </div>
            <h1 className="font-bold text-2xl text-slate-800">Acesso Administrativo</h1>
            <p className="text-slate-500 text-sm mt-2">GovBot - Painel de Controle</p>
          </div>
          
          <form onSubmit={handleLogin}>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Senha de Administrador
            </label>
            <input 
              type="password" 
              className="w-full border border-slate-300 p-3 rounded-lg mb-3 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
              value={adminPass}
              onChange={e => {
                setAdminPass(e.target.value);
                setLoginError('');
              }}
              placeholder="Digite a senha..."
              autoFocus
            />
            
            {loginError && (
              <div className="mb-4 p-3 bg-red-50 text-red-700 rounded-lg border border-red-200 text-sm flex items-center gap-2">
                <AlertCircle size={16}/>
                {loginError}
              </div>
            )}
            
            <div className="flex gap-3">
              <button 
                type="button" 
                onClick={() => navigate('/')}
                className="flex-1 py-3 text-slate-600 bg-slate-100 rounded-lg hover:bg-slate-200 font-medium transition-colors"
              >
                Voltar ao Chat
              </button>
              <button 
                type="submit" 
                className="flex-1 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium shadow-lg shadow-blue-600/30 transition-all"
              >
                Acessar
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  }

  // Página do Admin (quando autenticado)
  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <div className="bg-slate-900 text-white p-6 shadow-md">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button 
              onClick={() => navigate('/')}
              className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
            >
              <ArrowLeft size={20}/>
            </button>
            <div>
              <h1 className="text-2xl font-bold flex items-center gap-2">
                <Settings className="text-blue-400"/>
                Painel de Administração
              </h1>
              <p className="text-slate-400 text-sm">GovBot - Gestão de Conhecimento</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="text-sm bg-green-100 text-green-700 px-3 py-1 rounded-full border border-green-200 font-medium">
              Autenticado
            </div>
            <button 
              onClick={handleLogout}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors text-sm font-medium"
            >
              Sair
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="p-6 max-w-6xl mx-auto space-y-8">
        
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center text-blue-600">
                <Database size={24}/>
              </div>
              <div>
                <p className="text-2xl font-bold text-slate-800">{baseStats.totalChunks}</p>
                <p className="text-sm text-slate-500">Trechos Indexados</p>
              </div>
            </div>
          </div>
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center text-green-600">
                <FileText size={24}/>
              </div>
              <div>
                <p className="text-2xl font-bold text-slate-800">{baseStats.totalDocuments}</p>
                <p className="text-sm text-slate-500">Documentos</p>
              </div>
            </div>
          </div>
        </div>
        
        {/* Upload Card */}
        <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200 space-y-6">
          <div className="border-b border-slate-100 pb-4">
            <h3 className="font-semibold text-lg text-slate-800">Upload de Documentos</h3>
            <p className="text-sm text-slate-500">Adicione PDFs ou DOCX para que a IA aprenda o conteúdo.</p>
          </div>
          
          {/* ✅ Drag & Drop corrigido */}
          <div 
            className={`border-2 border-dashed rounded-xl p-10 text-center transition-all cursor-pointer ${dragActive ? 'border-blue-500 bg-blue-50' : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50'}`}
            onDragEnter={handleDragOver}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input type="file" id="fileUpload" onChange={handleUpload} className="hidden" accept=".pdf,.docx"/>
            <label htmlFor="fileUpload" className="cursor-pointer flex flex-col items-center gap-4">
              <div className="w-16 h-16 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center">
                <Upload size={32}/>
              </div>
              <div>
                <p className="font-medium text-slate-700 text-lg">
                  {dragActive ? 'Solte o arquivo aqui!' : 'Clique ou arraste um arquivo'}
                </p>
                <p className="text-sm text-slate-400">Suporta PDF e DOCX (Max 10MB)</p>
              </div>
            </label>
          </div>

          {uploadStatus && (
            <div className={`p-4 rounded-lg text-sm font-medium flex items-center gap-2 ${uploadStatus.includes('❌') ? 'bg-red-50 text-red-700 border border-red-100' : 'bg-green-50 text-green-700 border border-green-100'}`}>
              {uploadStatus.includes('❌') ? <AlertCircle size={18}/> : <CheckCircle size={18}/>}
              {uploadStatus}
            </div>
          )}
        </div>

        {/* Documentos Indexados */}
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-lg text-slate-800">Documentos Indexados</h3>
            <button onClick={loadDocuments} className="p-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors">
              <RefreshCw size={18}/>
            </button>
          </div>
          
          {isLoadingDocs ? (
            <div className="text-center py-8 text-slate-500">
              <RefreshCw size={24} className="animate-spin mx-auto mb-2"/>
              <p>Carregando documentos...</p>
            </div>
          ) : documents.length > 0 ? (
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {documents.map((doc, index) => (
                <div key={index} className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg border border-slate-200">
                  <FileText size={16} className="text-slate-400"/>
                  <span className="text-sm font-medium text-slate-700 truncate">{doc}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-slate-500">
              <FileText size={32} className="mx-auto mb-2 opacity-50"/>
              <p>Nenhum documento indexado</p>
              <p className="text-sm">Faça upload de arquivos para começar</p>
            </div>
          )}
        </div>

        {/* Gestão da Base */}
        <div className="bg-yellow-50 p-6 rounded-2xl border border-yellow-200">
          <h3 className="font-bold text-yellow-800 flex items-center gap-2 mb-4"><Database size={20}/> Gestão da Base</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <button 
              onClick={handleClearUploads}
              className="flex items-center gap-3 p-4 bg-white border border-yellow-300 text-yellow-700 rounded-lg hover:bg-yellow-100 font-medium transition shadow-sm text-left"
            >
              <Trash2 size={18}/>
              <div>
                <div className="font-semibold">Limpar Uploads</div>
                <div className="text-xs text-yellow-600">Remove arquivos temporários</div>
              </div>
            </button>
            
            <button 
              onClick={() => {loadDocuments(); getBaseStats();}}
              className="flex items-center gap-3 p-4 bg-white border border-blue-300 text-blue-700 rounded-lg hover:bg-blue-50 font-medium transition shadow-sm text-left"
            >
              <RefreshCw size={18}/>
              <div>
                <div className="font-semibold">Atualizar Dados</div>
                <div className="text-xs text-blue-600">Recarrega estatísticas</div>
              </div>
            </button>
          </div>
        </div>

        {/* Danger Zone */}
        <div className="bg-red-50 p-6 rounded-2xl border border-red-100">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h3 className="font-bold text-red-800 flex items-center gap-2"><AlertCircle size={20}/> Zona de Perigo</h3>
              <p className="text-sm text-red-600 mt-1">Apagar a base de dados remove todo o conhecimento indexado até agora.</p>
            </div>
            <button 
              onClick={handleClearBase} 
              className="flex items-center gap-2 px-5 py-2.5 bg-white border border-red-300 text-red-700 rounded-lg hover:bg-red-100 font-medium transition shadow-sm whitespace-nowrap"
            >
              <Trash2 size={18}/> Limpar Tudo
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}